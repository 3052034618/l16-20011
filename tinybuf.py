"""
tinybuf - 精简版二进制序列化协议编解码器
==========================================

线格式 (Wire Format) 设计
-------------------------
每个字段在线上由两部分组成:
  [Tag] [Value]

  Tag = (field_number << 3) | wire_type
  Tag 本身用 varint 编码, 因此小字段号(1-15)只需 1 字节。

  Wire Type:
    0 = VARINT   — int32, int64, uint32, uint64, sint32, sint64, bool, enum
    1 = FIXED64  — 固定 8 字节 (本实现暂不使用)
    2 = LEN      — string, bytes, 嵌套消息, 重复字段打包
    5 = FIXED32  — 固定 4 字节 (本实现暂不使用)

Varint 编码原理
---------------
每个字节的最高位 (MSB, bit 7) 是 "续接标志":
  - 1 = 后面还有更多字节
  - 0 = 这是最后一个字节
低 7 位承载实际数据, 小端序排列 (little-endian group)。

  例子: 150 = 0b10010110
    低 7 位 = 0010110, 高位 = 1  → 第一字节: 1_0010110 = 0x96
    剩余    = 0000001            → 第二字节: 0_0000001 = 0x01
    线上: 0x96 0x01

  优势: 0-127 只需 1 字节, 128-16383 只需 2 字节, 依此类推。
  小整数占绝大多数时, 压缩效果显著。

ZigZag 编码原理
---------------
直接用 varint 编码有符号整数, 负数会被当作极大的无符号数
(如 -1 → 0xFFFFFFFFFFFFFFFF → 10 字节)。
ZigZag 把有符号整数映射到无符号整数, 使绝对值小的数 (无论正负)
都映射到小的无符号值:

    n (signed)  →  (n << 1) ^ (n >> 63)   (64-bit)

    0  → 0        -1 → 1
    1  → 2        -2 → 3
    2  → 4        -3 → 5

  这样 -1 编码为 varint(1) = 1 字节, 而非 10 字节。

字段标签与向前兼容
-----------------
解码时先读 Tag (varint), 从中提取 wire_type 和 field_number。
若 field_number 在当前 schema 中未知, 只需根据 wire_type 跳过
固定长度即可:
  - VARINT: 读一个 varint 然后丢弃
  - LEN:    读 varint 得到长度 N, 再跳过 N 字节
这样新版本消息中的新字段不会破坏旧版本解码器 → 向前兼容。

嵌套消息的长度界定
-----------------
嵌套消息的 wire_type = LEN (2), Value 部分先是 varint 编码的
字节长度 L, 再跟 L 字节的子消息体。解码器读 L 后, 从流中切出
L 字节, 在子缓冲区上递归解码, 自然知道子消息在哪里结束——
不会读多, 也不会读少。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field as dc_field
from enum import IntEnum
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

T = TypeVar("T", bound="Message")

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_LEN = 2
WIRE_FIXED32 = 5


class FieldType(IntEnum):
    INT32 = 0
    INT64 = 1
    UINT32 = 2
    UINT64 = 3
    SINT32 = 4
    SINT64 = 5
    BOOL = 6
    STRING = 7
    BYTES = 8
    MESSAGE = 9
    FIXED32 = 10
    FIXED64 = 11


_WIRE_MAP = {
    FieldType.INT32: WIRE_VARINT,
    FieldType.INT64: WIRE_VARINT,
    FieldType.UINT32: WIRE_VARINT,
    FieldType.UINT64: WIRE_VARINT,
    FieldType.SINT32: WIRE_VARINT,
    FieldType.SINT64: WIRE_VARINT,
    FieldType.BOOL: WIRE_VARINT,
    FieldType.STRING: WIRE_LEN,
    FieldType.BYTES: WIRE_LEN,
    FieldType.MESSAGE: WIRE_LEN,
    FieldType.FIXED32: WIRE_FIXED32,
    FieldType.FIXED64: WIRE_FIXED64,
}

_INT32_RANGE = (-2**31, 2**31 - 1)
_INT64_RANGE = (-2**63, 2**63 - 1)
_UINT32_RANGE = (0, 2**32 - 1)
_UINT64_RANGE = (0, 2**64 - 1)


# ---------------------------------------------------------------------------
# Varint
# ---------------------------------------------------------------------------

def encode_varint(value: int) -> bytes:
    """
    将无符号整数编码为 varint。

    每字节: bit7=续接标志(1=还有后续, 0=末尾), bit6-0=数据。
    数据按小端分组排列 (最低 7 位先出)。
    """
    if value < 0:
        value &= 0xFFFFFFFFFFFFFFFF
    buf = bytearray()
    while value > 0x7F:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.append(value & 0x7F)
    return bytes(buf)


def decode_varint(stream: BytesIO) -> int:
    """
    从字节流中解码一个 varint, 返回无符号整数值。
    """
    result = 0
    shift = 0
    while True:
        raw = stream.read(1)
        if not raw:
            raise EOFError("Unexpected end of stream while reading varint")
        b = raw[0]
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
        if shift >= 64:
            raise ValueError("Varint too long (>64 bits)")
    return result


# ---------------------------------------------------------------------------
# ZigZag
# ---------------------------------------------------------------------------

def zigzag_encode(value: int, bits: int = 64) -> int:
    """
    ZigZag 编码: 将有符号整数映射到无符号整数。

      n → (n << 1) ^ (n >> (bits-1))

    bits=64 用于 sint64, bits=32 用于 sint32。
    效果: 0→0, -1→1, 1→2, -2→3, 2→4, ...
    绝对值越小的数, 编码后的值也越小 → varint 压缩效果好。
    """
    return (value << 1) ^ (value >> (bits - 1))


def zigzag_decode(value: int) -> int:
    """
    ZigZag 解码: 将无符号整数还原为有符号整数。

      n → (n >>> 1) ^ -(n & 1)

    偶数还原为非负数, 奇数还原为负数。
    """
    return (value >> 1) ^ -(value & 1)


# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------

def make_tag(field_number: int, wire_type: int) -> int:
    return (field_number << 3) | wire_type


def parse_tag(tag: int) -> Tuple[int, int]:
    return tag >> 3, tag & 0x07


# ---------------------------------------------------------------------------
# FieldDescriptor & Message
# ---------------------------------------------------------------------------

@dataclass
class FieldDescriptor:
    number: int
    name: str
    field_type: FieldType
    repeated: bool = False
    message_cls: Optional[Type["Message"]] = None


class Message:
    _field_descriptors: Dict[int, FieldDescriptor] = {}
    _field_by_name: Dict[str, FieldDescriptor] = {}

    def __init__(self, **kwargs: Any):
        for desc in self.__class__._field_descriptors.values():
            if desc.repeated:
                setattr(self, desc.name, list(kwargs.get(desc.name, [])))
            else:
                setattr(self, desc.name, kwargs.get(desc.name, self._default(desc)))

    @staticmethod
    def _default(desc: FieldDescriptor) -> Any:
        if desc.field_type == FieldType.STRING:
            return ""
        if desc.field_type == FieldType.BYTES:
            return b""
        if desc.field_type == FieldType.BOOL:
            return False
        if desc.field_type in (FieldType.INT32, FieldType.INT64,
                               FieldType.SINT32, FieldType.SINT64,
                               FieldType.UINT32, FieldType.UINT64,
                               FieldType.FIXED32, FieldType.FIXED64):
            return 0
        if desc.field_type == FieldType.MESSAGE:
            return desc.message_cls()
        return None

    @classmethod
    def define_field(cls, number: int, name: str, field_type: FieldType,
                     repeated: bool = False,
                     message_cls: Optional[Type["Message"]] = None) -> None:
        desc = FieldDescriptor(number, name, field_type, repeated, message_cls)
        if not hasattr(cls, '_field_descriptors') or cls._field_descriptors is Message._field_descriptors:
            cls._field_descriptors = {}
            cls._field_by_name = {}
        cls._field_descriptors[number] = desc
        cls._field_by_name[name] = desc

    def __repr__(self) -> str:
        parts = []
        for desc in self.__class__._field_descriptors.values():
            parts.append(f"{desc.name}={getattr(self, desc.name)!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        for desc in self._field_descriptors.values():
            if getattr(self, desc.name) != getattr(other, desc.name):
                return False
        return True


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def _encode_value(desc: FieldDescriptor, value: Any) -> bytes:
    ft = desc.field_type
    if ft in (FieldType.INT32, FieldType.INT64):
        if value < 0:
            value &= 0xFFFFFFFFFFFFFFFF
        return encode_varint(value)
    if ft in (FieldType.UINT32, FieldType.UINT64):
        return encode_varint(value)
    if ft == FieldType.SINT32:
        return encode_varint(zigzag_encode(value, 32))
    if ft == FieldType.SINT64:
        return encode_varint(zigzag_encode(value, 64))
    if ft == FieldType.BOOL:
        return encode_varint(1 if value else 0)
    if ft == FieldType.FIXED32:
        return struct.pack("<I", value & 0xFFFFFFFF)
    if ft == FieldType.FIXED64:
        return struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF)
    if ft == FieldType.STRING:
        encoded = value.encode("utf-8")
        return encode_varint(len(encoded)) + encoded
    if ft == FieldType.BYTES:
        return encode_varint(len(value)) + value
    if ft == FieldType.MESSAGE:
        payload = encode(value)
        return encode_varint(len(payload)) + payload
    raise ValueError(f"Unknown field type: {ft}")


def _encode_field(desc: FieldDescriptor, value: Any) -> bytes:
    wire_type = _WIRE_MAP[desc.field_type]
    tag_bytes = encode_varint(make_tag(desc.number, wire_type))
    value_bytes = _encode_value(desc, value)
    return tag_bytes + value_bytes


def encode(msg: Message) -> bytes:
    buf = bytearray()
    for desc in msg.__class__._field_descriptors.values():
        value = getattr(msg, desc.name)
        if desc.repeated:
            for item in value:
                buf.extend(_encode_field(desc, item))
        else:
            if _is_default_value(desc, value):
                continue
            buf.extend(_encode_field(desc, value))
    return bytes(buf)


def _is_default_value(desc: FieldDescriptor, value: Any) -> bool:
    if desc.field_type == FieldType.MESSAGE and value is not None:
        if encode(value) == b"":
            return True
        return False
    if value is None:
        return True
    if desc.field_type == FieldType.STRING and value == "":
        return True
    if desc.field_type == FieldType.BYTES and value == b"":
        return True
    if desc.field_type == FieldType.BOOL and value is False:
        return True
    if isinstance(value, int) and value == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def _skip_field(stream: BytesIO, wire_type: int) -> None:
    """
    根据 wireType 跳过未知字段, 保证向前兼容:
      VARINT:  读一个 varint 丢弃
      LEN:     读 varint 长度 N, 跳过 N 字节
      FIXED32: 跳过 4 字节
      FIXED64: 跳过 8 字节
    """
    if wire_type == WIRE_VARINT:
        decode_varint(stream)
    elif wire_type == WIRE_LEN:
        length = decode_varint(stream)
        stream.read(length)
    elif wire_type == WIRE_FIXED32:
        stream.read(4)
    elif wire_type == WIRE_FIXED64:
        stream.read(8)
    else:
        raise ValueError(f"Unknown wire type: {wire_type}")


def _decode_value(desc: FieldDescriptor, stream: BytesIO,
                  wire_type: int) -> Any:
    ft = desc.field_type
    if ft in (FieldType.INT32,):
        raw = decode_varint(stream)
        if raw > 0x7FFFFFFF:
            raw -= 0x100000000
        return raw
    if ft in (FieldType.INT64,):
        raw = decode_varint(stream)
        if raw > 0x7FFFFFFFFFFFFFFF:
            raw -= 0x10000000000000000
        return raw
    if ft in (FieldType.UINT32, FieldType.UINT64):
        return decode_varint(stream)
    if ft == FieldType.SINT32:
        return zigzag_decode(decode_varint(stream))
    if ft == FieldType.SINT64:
        return zigzag_decode(decode_varint(stream))
    if ft == FieldType.BOOL:
        return bool(decode_varint(stream))
    if ft == FieldType.FIXED32:
        return struct.unpack("<I", stream.read(4))[0]
    if ft == FieldType.FIXED64:
        return struct.unpack("<Q", stream.read(8))[0]
    if ft == FieldType.STRING:
        length = decode_varint(stream)
        return stream.read(length).decode("utf-8")
    if ft == FieldType.BYTES:
        length = decode_varint(stream)
        return stream.read(length)
    if ft == FieldType.MESSAGE:
        length = decode_varint(stream)
        sub_buf = stream.read(length)
        return decode(desc.message_cls, sub_buf)
    raise ValueError(f"Unknown field type: {ft}")


def decode(cls: Type[T], data: bytes) -> T:
    """
    从字节流解码消息。遇到未知 field_number 时根据 wire_type
    跳过对应字节, 保证向前兼容。
    """
    stream = BytesIO(data)
    kwargs: Dict[str, Any] = {}

    for desc in cls._field_descriptors.values():
        if desc.repeated:
            kwargs[desc.name] = []

    while True:
        try:
            tag_val = decode_varint(stream)
        except EOFError:
            break
        field_number, wire_type = parse_tag(tag_val)
        desc = cls._field_descriptors.get(field_number)

        if desc is None:
            _skip_field(stream, wire_type)
            continue

        expected_wire = _WIRE_MAP[desc.field_type]
        if wire_type != expected_wire:
            _skip_field(stream, wire_type)
            continue

        value = _decode_value(desc, stream, wire_type)

        if desc.repeated:
            kwargs.setdefault(desc.name, []).append(value)
        else:
            kwargs[desc.name] = value

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Demo / Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- 1. Varint 演示 ----
    print("=" * 60)
    print("Varint 编码演示")
    print("=" * 60)
    for n in [0, 1, 127, 128, 150, 300, 2**20 - 1]:
        encoded = encode_varint(n)
        decoded = decode_varint(BytesIO(encoded))
        print(f"  {n:>10} → {encoded.hex(' '):<20} ({len(encoded)} bytes) → {decoded}")

    # ---- 2. ZigZag 演示 ----
    print()
    print("=" * 60)
    print("ZigZag 编码演示 (sint64)")
    print("=" * 60)
    for n in [0, -1, 1, -2, 2, -100, 100]:
        zz = zigzag_encode(n, 64)
        varint_bytes = encode_varint(zz)
        restored = zigzag_decode(zz)
        print(f"  {n:>6} → zigzag={zz:>4} → varint={varint_bytes.hex(' '):<14} → {restored}")

    # ---- 3. 定义消息结构 ----

    class Address(Message):
        pass

    Address.define_field(1, "city", FieldType.STRING)
    Address.define_field(2, "zip", FieldType.STRING)

    class Person(Message):
        pass

    Person.define_field(1, "id", FieldType.UINT32)
    Person.define_field(2, "name", FieldType.STRING)
    Person.define_field(3, "score", FieldType.SINT32)
    Person.define_field(4, "address", FieldType.MESSAGE, message_cls=Address)
    Person.define_field(5, "tags", FieldType.STRING, repeated=True)
    Person.define_field(6, "active", FieldType.BOOL)
    Person.define_field(7, "payload", FieldType.BYTES)
    Person.define_field(8, "big_id", FieldType.SINT64)

    # ---- 4. 编码 ----
    addr = Address(city="Beijing", zip="100000")
    person = Person(
        id=42,
        name="Alice",
        score=-99,
        address=addr,
        tags=["engineer", "golang"],
        active=True,
        payload=b"\x01\x02\x03\x04",
        big_id=-1,
    )

    data = encode(person)
    print()
    print("=" * 60)
    print("编码结果")
    print("=" * 60)
    print(f"  原始消息: {person}")
    print(f"  字节长度: {len(data)} bytes")
    print(f"  十六进制: {data.hex(' ')}")

    # ---- 5. 解码 ----
    restored = decode(Person, data)
    print()
    print("=" * 60)
    print("解码结果")
    print("=" * 60)
    print(f"  还原消息: {restored}")
    print(f"  id      = {restored.id} (expect 42)")
    print(f"  name    = {restored.name!r} (expect 'Alice')")
    print(f"  score   = {restored.score} (expect -99)")
    print(f"  address = {restored.address} (expect city=Beijing)")
    print(f"  tags    = {restored.tags} (expect ['engineer','golang'])")
    print(f"  active  = {restored.active} (expect True)")
    print(f"  payload = {restored.payload.hex()} (expect 01020304)")
    print(f"  big_id  = {restored.big_id} (expect -1)")

    # ---- 6. 向前兼容演示 ----
    print()
    print("=" * 60)
    print("向前兼容演示: 解码含未知字段的数据")
    print("=" * 60)

    buf = bytearray()
    buf.extend(_encode_field(
        FieldDescriptor(1, "id", FieldType.UINT32), 42))
    buf.extend(_encode_field(
        FieldDescriptor(99, "future_field", FieldType.STRING), "unknown"))
    buf.extend(_encode_field(
        FieldDescriptor(100, "future_int", FieldType.UINT64), 123456))
    buf.extend(_encode_field(
        FieldDescriptor(2, "name", FieldType.STRING), "Bob"))
    buf.extend(_encode_field(
        FieldDescriptor(101, "future_msg", FieldType.MESSAGE, message_cls=Address),
        Address(city="Shanghai", zip="200000")))

    compat = decode(Person, bytes(buf))
    print(f"  解码结果: {compat}")
    print(f"  id   = {compat.id} (expect 42)")
    print(f"  name = {compat.name!r} (expect 'Bob')")
    print("  未知字段 99, 100, 101 被安全跳过 ✓")

    # ---- 7. 断言 ----
    assert restored == person, "Round-trip failed!"
    assert restored.id == 42
    assert restored.name == "Alice"
    assert restored.score == -99
    assert restored.address.city == "Beijing"
    assert restored.tags == ["engineer", "golang"]
    assert restored.active is True
    assert restored.payload == b"\x01\x02\x03\x04"
    assert restored.big_id == -1
    assert compat.id == 42
    assert compat.name == "Bob"

    print()
    print("✅ 所有测试通过!")
