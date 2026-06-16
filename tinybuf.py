"""
tinybuf - 精简版二进制序列化协议编解码器 (v2)
===============================================

线格式 (Wire Format) 设计
-------------------------
每个字段在线上由两部分组成:
  [Tag] [Value]

  Tag = (field_number << 3) | wire_type
  Tag 本身用 varint 编码, 因此小字段号(1-15)只需 1 字节。

  Wire Type:
    0 = VARINT   — int32, int64, uint32, uint64, sint32, sint64, bool, enum
    1 = FIXED64  — 固定 8 字节
    2 = LEN      — string, bytes, 嵌套消息, packed重复数值字段
    5 = FIXED32  — 固定 4 字节

Packed 编码
-----------
对 repeated 的 VARINT/FIXED32/FIXED64 字段, 可把一组元素合并成一个
LEN 块 (wire_type=2): 先 varint 写出总字节长度, 再依次把所有元素的
编码拼在一起。解码器读一个长度 L 后, 在 L 字节的子缓冲里循环解码直到耗尽。
这样就省去了 N 个重复 Tag 的开销。解码端同时接受 "普通重复 (各自带Tag)"
和 "packed (一个LEN块)" 两种写法, 实现上把这两种 wire_type 都导向同一字段。

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
  - VARINT:  读一个 varint 然后丢弃
  - LEN:     读 varint 得到长度 N, 再跳过 N 字节
  - FIXED32: 跳过 4 字节
  - FIXED64: 跳过 8 字节
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
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

T = TypeVar("T", bound="Message")

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_LEN = 2
WIRE_FIXED32 = 5


# ---------------------------------------------------------------------------
# 自定义异常 — 让解码错误可理解、可捕获
# ---------------------------------------------------------------------------

class DecodeError(Exception):
    """tinybuf 解码错误基类。"""


class TruncatedDataError(DecodeError):
    """字节流在字符串/bytes/嵌套消息/fixed字段中间被截断。"""


class VarintOverflowError(DecodeError):
    """varint 超过 64 位, 或字段整数超过声明类型的位宽。"""


class WireTypeMismatchError(DecodeError):
    """Tag 声明的 wire_type 与该字段能接受的类型不符。"""


class SchemaError(Exception):
    """schema 定义错误 (字段号冲突、遗漏参数等)。"""


# ---------------------------------------------------------------------------
# FieldType
# ---------------------------------------------------------------------------

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


_WIRE_MAP: Dict[FieldType, int] = {
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

# 哪些字段类型支持 packed 编码 (其元素的 wire_type 是 VARINT/FIXED32/FIXED64)
_PACKABLE = {
    FieldType.INT32, FieldType.INT64,
    FieldType.UINT32, FieldType.UINT64,
    FieldType.SINT32, FieldType.SINT64,
    FieldType.BOOL,
    FieldType.FIXED32, FieldType.FIXED64,
}

_MAX_VARINT_BYTES = 10  # 64 bits / 7 ≈ 9.14, 故最多 10 字节


# ---------------------------------------------------------------------------
# Varint — 带更明确的异常
# ---------------------------------------------------------------------------

def encode_varint(value: int) -> bytes:
    """
    将无符号整数编码为 varint。

    每字节: bit7=续接标志(1=还有后续, 0=末尾), bit6-0=数据。
    数据按小端分组排列 (最低 7 位先出)。
    输入若为负数, 按补码扩展到 64 位无符号数 (与 protobuf 行为一致)。
    """
    if value < 0:
        value &= 0xFFFFFFFFFFFFFFFF
    buf = bytearray()
    while value > 0x7F:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.append(value & 0x7F)
    return bytes(buf)


def _assert_read(stream: BytesIO, n: int, field: str) -> bytes:
    """从流中读 n 字节, 若读不满则抛出 TruncatedDataError。"""
    data = stream.read(n)
    if len(data) != n:
        raise TruncatedDataError(
            f"Truncated {field}: expected {n} bytes, got {len(data)}"
        )
    return data


def decode_varint(stream: BytesIO, ctx: str = "varint") -> int:
    """
    从字节流中解码一个 varint, 返回无符号整数值。
    ctx 用于在异常消息中标识当前位置 (例如哪个字段)。
    """
    result = 0
    shift = 0
    while True:
        raw = stream.read(1)
        if not raw:
            raise TruncatedDataError(
                f"Truncated {ctx}: unexpected end of stream while reading varint"
            )
        b = raw[0]
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
        if shift >= 64:
            raise VarintOverflowError(
                f"{ctx}: varint exceeds 64 bits (more than {_MAX_VARINT_BYTES} bytes)"
            )
    return result


# ---------------------------------------------------------------------------
# ZigZag — 同时处理 32/64 位
# ---------------------------------------------------------------------------

def zigzag_encode(value: int, bits: int = 64) -> int:
    """
    ZigZag 编码: 将有符号整数映射到无符号整数。
      n → (n << 1) ^ (n >> (bits-1))
    映射后先按 bits 位宽截断, 避免 Python 任意精度整数带来的偏移。
    """
    mask63 = (1 << (bits - 1)) - 1
    sign = (value >> (bits - 1)) & 1
    return ((value << 1) ^ (-sign)) & ((1 << bits) - 1)


def zigzag_decode(value: int, bits: int = 64) -> int:
    """
    ZigZag 解码: 将无符号整数还原为有符号整数。
      n → (n >>> 1) ^ -(n & 1)
    结果按 bits 位宽做符号扩展。
    """
    unsigned = value & ((1 << bits) - 1)
    half = unsigned >> 1
    if unsigned & 1:
        half = -half - 1
    return half


# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------

def make_tag(field_number: int, wire_type: int) -> int:
    if field_number < 1:
        raise SchemaError(f"Field number must be >= 1, got {field_number}")
    return (field_number << 3) | wire_type


def parse_tag(tag: int) -> Tuple[int, int]:
    return tag >> 3, tag & 0x07


# ---------------------------------------------------------------------------
# 有符号/无符号整数按位宽截断 & 符号扩展 — 解决边界还原问题
# ---------------------------------------------------------------------------

def _to_signed(value: int, bits: int) -> int:
    """把 value (可能是无符号表示) 按 bits 位宽解释为有符号整数。"""
    if bits <= 0:
        raise ValueError("bits must be positive")
    value &= (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    if value & sign_bit:
        return value - (1 << bits)
    return value


def _to_unsigned(value: int, bits: int) -> int:
    """把 value (可能是负数) 按 bits 位宽转换为补码表示的无符号整数。"""
    return value & ((1 << bits) - 1)


# ---------------------------------------------------------------------------
# Field 描述对象 — 供装饰器风格 schema 使用
# ---------------------------------------------------------------------------

@dataclass
class _FieldSpec:
    """装饰器内临时保存的字段声明。"""
    number: int
    field_type: FieldType
    repeated: bool = False
    packed: bool = False
    message_cls: Optional[Type["Message"]] = None


class Field:
    """
    在类体中声明字段的描述符。示例::

        @msg_schema
        class Person(Message):
            id = Field(1, FieldType.UINT32)
            name = Field(2, FieldType.STRING)
            scores = Field(3, FieldType.SINT32, repeated=True, packed=True)
    """

    __slots__ = ("_spec",)

    def __init__(self, number: int, field_type: FieldType,
                 repeated: bool = False,
                 packed: bool = False,
                 message_cls: Optional[Type["Message"]] = None):
        if packed and (not repeated or field_type not in _PACKABLE):
            raise SchemaError(
                f"packed=True requires repeated=True and a packable scalar type "
                f"(VARINT/FIXED32/FIXED64), got repeated={repeated}, type={field_type.name}"
            )
        self._spec = _FieldSpec(number, field_type, repeated, packed, message_cls)

    def __set_name__(self, owner: Type["Message"], name: str) -> None:
        # 装饰器 @msg_schema 会遍历类 __dict__ 收集 Field,
        # 这里我们什么都不做, 只留 hook 以备扩展。
        pass


# ---------------------------------------------------------------------------
# FieldDescriptor & Message — 加 packed; 加装饰器注册
# ---------------------------------------------------------------------------

@dataclass
class FieldDescriptor:
    number: int
    name: str
    field_type: FieldType
    repeated: bool = False
    packed: bool = False
    message_cls: Optional[Type["Message"]] = None

    def packable(self) -> bool:
        return self.repeated and self.field_type in _PACKABLE


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
    def _register_field(cls, spec: _FieldSpec, name: str) -> None:
        if not hasattr(cls, '_field_descriptors') or cls._field_descriptors is Message._field_descriptors:
            cls._field_descriptors = {}
            cls._field_by_name = {}
        if spec.number in cls._field_descriptors:
            raise SchemaError(
                f"Duplicate field number {spec.number} in {cls.__name__}: "
                f"{cls._field_descriptors[spec.number].name} vs {name}"
            )
        if name in cls._field_by_name:
            raise SchemaError(f"Duplicate field name '{name}' in {cls.__name__}")
        desc = FieldDescriptor(spec.number, name, spec.field_type,
                               spec.repeated, spec.packed, spec.message_cls)
        cls._field_descriptors[spec.number] = desc
        cls._field_by_name[name] = desc

    @classmethod
    def define_field(cls, number: int, name: str, field_type: FieldType,
                     repeated: bool = False, packed: bool = False,
                     message_cls: Optional[Type["Message"]] = None) -> None:
        """旧风格 API, 保持向后兼容。"""
        spec = _FieldSpec(number, field_type, repeated, packed, message_cls)
        if packed and (not repeated or field_type not in _PACKABLE):
            raise SchemaError(
                "packed=True requires repeated=True and packable scalar type"
            )
        cls._register_field(spec, name)

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


def msg_schema(cls: Type[T]) -> Type[T]:
    """
    装饰器: 遍历类体中声明的 Field 描述符, 注册到 _field_descriptors。

    子类的字段会自动包含父类字段 (子类号不能和父类冲突)。
    """
    if not issubclass(cls, Message):
        raise SchemaError(f"@msg_schema can only decorate Message subclasses, got {cls}")

    # 1) 先从基类拷贝已有字段 (保证继承可用), 只拷贝最近一层自己的
    inherited: Dict[int, FieldDescriptor] = {}
    for base in cls.__mro__[1:]:
        base_fields = getattr(base, "_field_descriptors", {})
        if base_fields is Message._field_descriptors:
            continue
        for n, desc in base_fields.items():
            if n in inherited:
                continue
            inherited[n] = desc

    # 2) 为当前类创建独立的字段表 (先继承, 再覆盖)
    cls._field_descriptors = {}
    cls._field_by_name = {}
    for n, desc in inherited.items():
        cls._field_descriptors[n] = desc
        cls._field_by_name[desc.name] = desc

    # 3) 扫描类体中的 Field 描述符
    fields_found: List[Tuple[str, _FieldSpec]] = []
    for name, attr in list(vars(cls).items()):
        if isinstance(attr, Field):
            fields_found.append((name, attr._spec))
            # 从类字典里移除描述符, 避免挡住实例属性
            try:
                delattr(cls, name)
            except AttributeError:
                pass

    # 4) 按字段号排序注册 (稳定顺序)
    fields_found.sort(key=lambda x: x[1].number)
    for name, spec in fields_found:
        # 自动推断 message_cls: 若 type=MESSAGE 但未指定, 报错
        if spec.field_type == FieldType.MESSAGE and spec.message_cls is None:
            raise SchemaError(
                f"Field '{cls.__name__}.{name}' (number={spec.number}) is MESSAGE "
                f"type but no message_cls= was given"
            )
        if spec.field_type != FieldType.MESSAGE and spec.message_cls is not None:
            raise SchemaError(
                f"Field '{cls.__name__}.{name}' has message_cls= but is not MESSAGE type"
            )
        cls._register_field(spec, name)

    return cls


# ---------------------------------------------------------------------------
# 单元素编码/解码 (可复用在 packed 和 非 packed 场景)
# ---------------------------------------------------------------------------

def _encode_single_value(field_type: FieldType,
                         message_cls: Optional[Type[Message]],
                         value: Any) -> bytes:
    """只编码 Value 部分 (不含 Tag / 长度前缀)。"""
    if field_type == FieldType.INT32:
        return encode_varint(_to_unsigned(value, 32))
    if field_type == FieldType.INT64:
        return encode_varint(_to_unsigned(value, 64))
    if field_type == FieldType.UINT32:
        return encode_varint(value & 0xFFFFFFFF)
    if field_type == FieldType.UINT64:
        return encode_varint(value & 0xFFFFFFFFFFFFFFFF)
    if field_type == FieldType.SINT32:
        return encode_varint(zigzag_encode(value, 32))
    if field_type == FieldType.SINT64:
        return encode_varint(zigzag_encode(value, 64))
    if field_type == FieldType.BOOL:
        return encode_varint(1 if value else 0)
    if field_type == FieldType.FIXED32:
        return struct.pack("<I", value & 0xFFFFFFFF)
    if field_type == FieldType.FIXED64:
        return struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF)
    if field_type == FieldType.STRING:
        encoded = value.encode("utf-8")
        return encode_varint(len(encoded)) + encoded
    if field_type == FieldType.BYTES:
        return encode_varint(len(value)) + value
    if field_type == FieldType.MESSAGE:
        payload = encode(message_cls() if value is None else value)
        return encode_varint(len(payload)) + payload
    raise ValueError(f"Unknown field type: {field_type}")


def _encode_single_varint_value(field_type: FieldType, value: Any) -> bytes:
    """只把 "元素自身的编码" 打出来, 不带长度前缀 (用于 packed VARINT)。"""
    if field_type == FieldType.INT32:
        return encode_varint(_to_unsigned(value, 32))
    if field_type == FieldType.INT64:
        return encode_varint(_to_unsigned(value, 64))
    if field_type == FieldType.UINT32:
        return encode_varint(value & 0xFFFFFFFF)
    if field_type == FieldType.UINT64:
        return encode_varint(value & 0xFFFFFFFFFFFFFFFF)
    if field_type == FieldType.SINT32:
        return encode_varint(zigzag_encode(value, 32))
    if field_type == FieldType.SINT64:
        return encode_varint(zigzag_encode(value, 64))
    if field_type == FieldType.BOOL:
        return encode_varint(1 if value else 0)
    raise ValueError(f"Not a varint-based scalar: {field_type}")


def _decode_single_value(desc: FieldDescriptor, stream: BytesIO,
                         ctx: str) -> Any:
    """根据字段类型从流中解码一个值 (调用方负责读 Tag / 长度)。"""
    ft = desc.field_type
    if ft == FieldType.INT32:
        raw = decode_varint(stream, ctx)
        if raw > 0xFFFFFFFF:
            raise VarintOverflowError(f"{ctx}: value {raw} exceeds uint32 range")
        return _to_signed(raw, 32)
    if ft == FieldType.INT64:
        raw = decode_varint(stream, ctx)
        return _to_signed(raw, 64)
    if ft == FieldType.UINT32:
        raw = decode_varint(stream, ctx)
        if raw > 0xFFFFFFFF:
            raise VarintOverflowError(f"{ctx}: value {raw} exceeds uint32 range")
        return raw
    if ft == FieldType.UINT64:
        return decode_varint(stream, ctx)
    if ft == FieldType.SINT32:
        raw = decode_varint(stream, ctx)
        if raw > 0xFFFFFFFF:
            raise VarintOverflowError(f"{ctx}: value {raw} exceeds uint32 range")
        return zigzag_decode(raw, 32)
    if ft == FieldType.SINT64:
        raw = decode_varint(stream, ctx)
        return zigzag_decode(raw, 64)
    if ft == FieldType.BOOL:
        raw = decode_varint(stream, ctx)
        if raw not in (0, 1):
            # 允许非 0/1 的 varint, 按非 0 为真处理 (和 protobuf 一致)
            pass
        return raw != 0
    if ft == FieldType.FIXED32:
        raw = _assert_read(stream, 4, ctx)
        return struct.unpack("<I", raw)[0]
    if ft == FieldType.FIXED64:
        raw = _assert_read(stream, 8, ctx)
        return struct.unpack("<Q", raw)[0]
    if ft == FieldType.STRING:
        length = decode_varint(stream, f"{ctx}.length")
        if length < 0:
            raise DecodeError(f"{ctx}: negative string length {length}")
        raw = _assert_read(stream, length, f"{ctx}.payload")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecodeError(f"{ctx}: invalid UTF-8: {exc}") from exc
    if ft == FieldType.BYTES:
        length = decode_varint(stream, f"{ctx}.length")
        if length < 0:
            raise DecodeError(f"{ctx}: negative bytes length {length}")
        return _assert_read(stream, length, f"{ctx}.payload")
    if ft == FieldType.MESSAGE:
        length = decode_varint(stream, f"{ctx}.length")
        if length < 0:
            raise DecodeError(f"{ctx}: negative message length {length}")
        sub_buf = _assert_read(stream, length, f"{ctx}.payload")
        return decode(desc.message_cls, sub_buf, stack=[desc.message_cls.__name__])
    raise ValueError(f"Unknown field type: {ft}")


def _decode_packed_values(desc: FieldDescriptor, payload: bytes,
                          ctx: str) -> List[Any]:
    """把一个 LEN 块 (packed) 解码成该字段的元素列表。"""
    stream = BytesIO(payload)
    items: List[Any] = []
    ft = desc.field_type
    if ft in (FieldType.FIXED32, FieldType.FIXED64):
        size = 4 if ft == FieldType.FIXED32 else 8
        total = len(payload)
        if total % size != 0:
            raise TruncatedDataError(
                f"{ctx}: packed {ft.name} block length {total} not divisible by {size}"
            )
        while True:
            raw = stream.read(size)
            if not raw:
                break
            fmt = "<I" if size == 4 else "<Q"
            items.append(struct.unpack(fmt, raw)[0])
        return items
    # VARINT 类 (int32/64, uint32/64, sint32/64, bool)
    idx = 0
    while True:
        try:
            raw = decode_varint(stream, f"{ctx}.element[{idx}]")
        except TruncatedDataError:
            if stream.tell() == len(payload):
                break
            raise
        if ft == FieldType.INT32:
            if raw > 0xFFFFFFFF:
                raise VarintOverflowError(
                    f"{ctx}.element[{idx}]: value {raw} exceeds uint32 range"
                )
            items.append(_to_signed(raw, 32))
        elif ft == FieldType.INT64:
            items.append(_to_signed(raw, 64))
        elif ft == FieldType.UINT32:
            if raw > 0xFFFFFFFF:
                raise VarintOverflowError(
                    f"{ctx}.element[{idx}]: value {raw} exceeds uint32 range"
                )
            items.append(raw)
        elif ft == FieldType.UINT64:
            items.append(raw)
        elif ft == FieldType.SINT32:
            if raw > 0xFFFFFFFF:
                raise VarintOverflowError(
                    f"{ctx}.element[{idx}]: value {raw} exceeds uint32 range"
                )
            items.append(zigzag_decode(raw, 32))
        elif ft == FieldType.SINT64:
            items.append(zigzag_decode(raw, 64))
        elif ft == FieldType.BOOL:
            items.append(raw != 0)
        idx += 1
    return items


# ---------------------------------------------------------------------------
# Encoder — 支持 packed
# ---------------------------------------------------------------------------

def _encode_value(desc: FieldDescriptor, value: Any) -> bytes:
    return _encode_single_value(desc.field_type, desc.message_cls, value)


def _encode_field(desc: FieldDescriptor, value: Any) -> bytes:
    wire_type = _WIRE_MAP[desc.field_type]
    tag_bytes = encode_varint(make_tag(desc.number, wire_type))
    value_bytes = _encode_value(desc, value)
    return tag_bytes + value_bytes


def _encode_packed_field(desc: FieldDescriptor, values: Sequence[Any]) -> bytes:
    """把 repeated 数值字段打包成单个 LEN 块。"""
    body = bytearray()
    ft = desc.field_type
    for v in values:
        if ft == FieldType.FIXED32:
            body.extend(struct.pack("<I", v & 0xFFFFFFFF))
        elif ft == FieldType.FIXED64:
            body.extend(struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF))
        else:
            body.extend(_encode_single_varint_value(ft, v))
    tag = encode_varint(make_tag(desc.number, WIRE_LEN))
    length = encode_varint(len(body))
    return bytes(tag + length + body)


def encode(msg: Message) -> bytes:
    buf = bytearray()
    for desc in msg.__class__._field_descriptors.values():
        value = getattr(msg, desc.name)
        if desc.repeated:
            items = list(value)
            if not items:
                continue
            if desc.packed:
                buf.extend(_encode_packed_field(desc, items))
            else:
                for item in items:
                    buf.extend(_encode_field(desc, item))
        else:
            if _is_default_value(desc, value):
                continue
            buf.extend(_encode_field(desc, value))
    return bytes(buf)


def _is_default_value(desc: FieldDescriptor, value: Any) -> bool:
    if value is None:
        return True
    if desc.field_type == FieldType.MESSAGE and value is not None:
        if encode(value) == b"":
            return True
        return False
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
# Decoder — 更严格的异常; packed + 普通重复通吃
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
        decode_varint(stream, "skip.varint")
    elif wire_type == WIRE_LEN:
        length = decode_varint(stream, "skip.len.length")
        _assert_read(stream, length, "skip.len.payload")
    elif wire_type == WIRE_FIXED32:
        _assert_read(stream, 4, "skip.fixed32")
    elif wire_type == WIRE_FIXED64:
        _assert_read(stream, 8, "skip.fixed64")
    else:
        raise WireTypeMismatchError(f"Unknown wire type {wire_type} while skipping field")


def decode(cls: Type[T], data: bytes, stack: Optional[List[str]] = None) -> T:
    """
    从字节流解码消息。
    - 遇到未知 field_number 时根据 wire_type 跳过 → 向前兼容
    - repeated 数值字段:
        * wire_type == VARINT/FIXED32/FIXED64 → 普通重复写法, 追加一个元素
        * wire_type == LEN                    → packed 写法, 把子块内所有元素追加
    """
    if stack is None:
        stack = [cls.__name__]
    stream = BytesIO(data)
    kwargs: Dict[str, Any] = {}
    for desc in cls._field_descriptors.values():
        if desc.repeated:
            kwargs[desc.name] = []

    expected_wire_defaults: Dict[int, int] = {
        n: _WIRE_MAP[d.field_type] for n, d in cls._field_descriptors.items()
    }

    while True:
        pos = stream.tell()
        try:
            tag_val = decode_varint(stream, f"{cls.__name__}.tag@offset={pos}")
        except TruncatedDataError:
            # 刚好读到末尾是正常终止
            if stream.tell() == len(data) and pos == len(data):
                break
            raise

        field_number, wire_type = parse_tag(tag_val)
        desc = cls._field_descriptors.get(field_number)

        if desc is None:
            _skip_field(stream, wire_type)
            continue

        default_wire = expected_wire_defaults[field_number]
        ctx = f"{'.'.join(stack)}.{desc.name}@offset={pos}"

        # ---- repeated + packable: 接受 packed (LEN) 和普通两种 ----
        if desc.repeated and desc.packable():
            if wire_type == WIRE_LEN:
                # packed 写法
                length = decode_varint(stream, f"{ctx}.packed.length")
                payload = _assert_read(stream, length, f"{ctx}.packed.payload")
                kwargs[desc.name].extend(_decode_packed_values(desc, payload, ctx))
                continue
            elif wire_type == default_wire:
                # 普通重复写法
                value = _decode_single_value(desc, stream, ctx)
                kwargs[desc.name].append(value)
                continue
            else:
                raise WireTypeMismatchError(
                    f"{ctx}: wire type {wire_type} not accepted by repeated "
                    f"{desc.field_type.name} field (expected {default_wire} or {WIRE_LEN})"
                )

        # ---- 其他字段: 必须匹配默认 wire_type ----
        if wire_type != default_wire:
            # 不匹配就跳过 (向前兼容: 旧字段的新编码格式旧解析器忽略)
            try:
                _skip_field(stream, wire_type)
            except WireTypeMismatchError:
                raise WireTypeMismatchError(
                    f"{ctx}: wire type {wire_type} does not match "
                    f"{desc.field_type.name} (expected {default_wire})"
                )
            continue

        value = _decode_single_value(desc, stream, ctx)

        if desc.repeated:
            kwargs.setdefault(desc.name, []).append(value)
        else:
            kwargs[desc.name] = value

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Demo / Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0

    def check(name: str, cond: Any, detail: str = "") -> None:
        global passed, failed
        if cond:
            passed += 1
            print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
        else:
            failed += 1
            print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))

    # ---- 1. Varint 边界演示 ----
    print("=" * 60)
    print("[1/7] Varint 边界演示")
    print("=" * 60)
    for n in [0, 1, 127, 128, 150, 300, 2**20 - 1, 2**63 - 1]:
        encoded = encode_varint(n)
        decoded = decode_varint(BytesIO(encoded))
        print(f"  {n:>20} → {encoded.hex(' '):<26} ({len(encoded)} B) → {decoded}")
        check(f"varint round-trip {n}", decoded == n)

    # ---- 2. ZigZag 边界演示 ----
    print()
    print("=" * 60)
    print("[2/7] ZigZag + 整数边界还原 (int32/int64)")
    print("=" * 60)
    for n in [0, -1, 1, -2, 2, -100, 100,
              -2**31, 2**31 - 1,  # int32 边界
              -2**63, 2**63 - 1]:  # int64 边界
        zz64 = zigzag_encode(n, 64)
        zz32 = zigzag_encode(n, 32)
        back64 = zigzag_decode(zz64, 64)
        back32 = zigzag_decode(zz32, 32)
        print(f"  {n:>20} → zz64={zz64:<20} zz32={zz32:<10} → back64={back64} back32={back32}")

    # INT32 负数通过 varint + _to_signed 还原
    int32_samples = [-1, -2, -2**31, 2**31 - 1, 0, 1, -123456789]
    print()
    print("  INT32 边界 round-trip (encode_varint + _to_signed):")
    for v in int32_samples:
        encoded = encode_varint(_to_unsigned(v, 32))
        raw = decode_varint(BytesIO(encoded))
        restored = _to_signed(raw, 32)
        check(f"  int32 {v:>12}", restored == v,
              f"→ varint={encoded.hex(' ')}, restored={restored}")

    # INT64 边界
    int64_samples = [-1, -2**63, 2**63 - 1, -123456789012345]
    print("  INT64 边界 round-trip:")
    for v in int64_samples:
        encoded = encode_varint(_to_unsigned(v, 64))
        raw = decode_varint(BytesIO(encoded))
        restored = _to_signed(raw, 64)
        check(f"  int64 {v:>20}", restored == v,
              f"→ varint={encoded.hex(' ')}, restored={restored}")

    # ---- 3. 装饰器风格 schema ----
    print()
    print("=" * 60)
    print("[3/7] @msg_schema 装饰器定义消息")
    print("=" * 60)

    @msg_schema
    class Address(Message):
        city = Field(1, FieldType.STRING)
        zip_code = Field(2, FieldType.STRING)

    @msg_schema
    class Person(Message):
        id = Field(1, FieldType.UINT32)
        name = Field(2, FieldType.STRING)
        score = Field(3, FieldType.SINT32)
        address = Field(4, FieldType.MESSAGE, message_cls=Address)
        tags = Field(5, FieldType.STRING, repeated=True)
        active = Field(6, FieldType.BOOL)
        payload = Field(7, FieldType.BYTES)
        big_id = Field(8, FieldType.SINT64)
        scores = Field(9, FieldType.INT32, repeated=True, packed=True)
        flags = Field(10, FieldType.BOOL, repeated=True, packed=True)

    print(f"  Person._field_descriptors:")
    for n, desc in sorted(Person._field_descriptors.items()):
        extra = ""
        if desc.repeated:
            extra += " repeated"
        if desc.packed:
            extra += " packed"
        print(f"    #{n:<3} {desc.name:<10} {desc.field_type.name:<8}{extra}")

    # ---- 4. 基础编解码 + INT32/INT64 负数 ----
    print()
    print("=" * 60)
    print("[4/7] 基础编码 / 解码 round-trip (含 packed 字段)")
    print("=" * 60)

    addr = Address(city="Beijing", zip_code="100000")
    person = Person(
        id=42,
        name="Alice",
        score=-99,
        address=addr,
        tags=["engineer", "golang"],
        active=True,
        payload=b"\x01\x02\x03\x04",
        big_id=-1,
        scores=[-1, -2**31, 2**31 - 1, 0, 100, -100],
        flags=[True, False, True, True],
    )

    data = encode(person)
    print(f"  字节长度: {len(data)} bytes")
    print(f"  十六进制: {data.hex(' ')}")

    restored = decode(Person, data)
    check("整体相等", restored == person)
    check(f"id={restored.id}", restored.id == 42)
    check(f"name={restored.name!r}", restored.name == "Alice")
    check(f"score={restored.score}", restored.score == -99)
    check(f"address.city={restored.address.city}", restored.address.city == "Beijing")
    check(f"tags={restored.tags}", restored.tags == ["engineer", "golang"])
    check(f"active={restored.active}", restored.active is True)
    check(f"payload={restored.payload.hex()}", restored.payload == b"\x01\x02\x03\x04")
    check(f"big_id={restored.big_id}", restored.big_id == -1)
    check(f"scores (int32 packed, 含边界)={restored.scores}",
          restored.scores == [-1, -2**31, 2**31 - 1, 0, 100, -100])
    check(f"flags (bool packed)={restored.flags}",
          restored.flags == [True, False, True, True])

    # ---- 5. packed / 非 packed 两种写法互通 ----
    print()
    print("=" * 60)
    print("[5/7] packed ↔ 普通重复 双写互通")
    print("=" * 60)

    # 定义两个几乎一样的消息: 一个 packed=True, 一个 packed=False
    @msg_schema
    class MetricsPacked(Message):
        values = Field(1, FieldType.SINT64, repeated=True, packed=True)

    @msg_schema
    class MetricsLoose(Message):
        values = Field(1, FieldType.SINT64, repeated=True)

    samples = [0, -1, 100, -200, 2**30, -2**40]

    packed_src = MetricsPacked(values=samples)
    packed_bytes = encode(packed_src)
    print(f"  Packed 编码: {packed_bytes.hex(' ')}")

    loose_src = MetricsLoose(values=samples)
    loose_bytes = encode(loose_src)
    print(f"  Loose  编码: {loose_bytes.hex(' ')}")
    print(f"  packed vs loose 尺寸: {len(packed_bytes)} vs {len(loose_bytes)} bytes "
          f"(packed 节省 {len(loose_bytes) - len(packed_bytes)} B)")

    # 双向互读
    via_loose = decode(MetricsLoose, packed_bytes)  # packed bytes → loose schema
    via_packed = decode(MetricsPacked, loose_bytes)  # loose bytes → packed schema
    direct_packed = decode(MetricsPacked, packed_bytes)
    direct_loose = decode(MetricsLoose, loose_bytes)
    check("packed→loose 解码", via_loose.values == samples, f"{via_loose.values}")
    check("loose→packed 解码", via_packed.values == samples, f"{via_packed.values}")
    check("packed 自洽", direct_packed.values == samples)
    check("loose 自洽", direct_loose.values == samples)

    # ---- 6. 向前兼容 + 异常 ----
    print()
    print("=" * 60)
    print("[6/7] 向前兼容 / 异常类型演示")
    print("=" * 60)

    buf = bytearray()
    buf.extend(_encode_field(
        FieldDescriptor(1, "id", FieldType.UINT32), 42))
    buf.extend(_encode_field(
        FieldDescriptor(99, "future_str", FieldType.STRING), "unknown"))
    buf.extend(_encode_field(
        FieldDescriptor(100, "future_int", FieldType.UINT64), 123456))
    buf.extend(_encode_field(
        FieldDescriptor(2, "name", FieldType.STRING), "Bob"))
    buf.extend(_encode_field(
        FieldDescriptor(101, "future_msg", FieldType.MESSAGE, message_cls=Address),
        Address(city="Shanghai", zip_code="200000")))
    # packed 的未知数值字段 (int32 packed) — 用 LEN 跳过
    packed_desc = FieldDescriptor(102, "nums", FieldType.INT32, repeated=True, packed=True)
    buf.extend(_encode_packed_field(packed_desc, [1, 2, 3, 4, 5]))

    compat = decode(Person, bytes(buf))
    check("兼容 id=42", compat.id == 42)
    check("兼容 name='Bob'", compat.name == "Bob")
    check("兼容 scores 空列表", compat.scores == [])

    # 异常演示
    print()
    print("  异常类型测试:")

    # 截断的字符串 (长度 10 但只有 3 字节内容)
    try:
        bad = encode_varint(make_tag(2, WIRE_LEN)) + encode_varint(10) + b"abc"
        decode(Person, bad)
        check("截断字符串抛异常", False, "没有抛异常!")
    except TruncatedDataError as exc:
        check("截断字符串 → TruncatedDataError", True, str(exc))

    # 截断的嵌套消息
    try:
        bad_msg = encode_varint(make_tag(4, WIRE_LEN)) + encode_varint(5) + b"\x01"
        decode(Person, bad_msg)
        check("截断嵌套消息抛异常", False, "没有抛异常!")
    except TruncatedDataError as exc:
        check("截断嵌套消息 → TruncatedDataError", True, str(exc))

    # 过长 varint (11 字节全带续位)
    try:
        too_long = b"\x80" * 11 + b"\x00"
        decode_varint(BytesIO(too_long), "test")
        check("超长 varint 抛异常", False, "没有抛异常!")
    except VarintOverflowError as exc:
        check("超长 varint → VarintOverflowError", True, str(exc))

    # 非法 UTF-8
    try:
        bad_utf8 = (encode_varint(make_tag(2, WIRE_LEN))
                    + encode_varint(5)
                    + b"\xff\xfe\xfd\xfc\xfb")
        decode(Person, bad_utf8)
        check("非法 UTF-8 抛异常", False, "没有抛异常!")
    except DecodeError as exc:
        check("非法 UTF-8 → DecodeError", True, str(exc))

    # INT32 varint 超过 32 位 (一个合法但超宽的 varint: 2^33)
    try:
        big = encode_varint(make_tag(9, FieldType.UINT32.value)) + encode_varint(1 << 33)
        # 实际上 scores 是 repeated packed INT32, wire_type 需要匹配,
        # 我们改走一个非 packed 的 INT32 单字段来测试
        @msg_schema
        class SingleI32(Message):
            x = Field(1, FieldType.INT32)
        bad = encode_varint(make_tag(1, WIRE_VARINT)) + encode_varint(1 << 33)
        decode(SingleI32, bad)
        check("INT32 超宽抛异常", False, "没有抛异常!")
    except VarintOverflowError as exc:
        check("INT32 超宽 → VarintOverflowError", True, str(exc))

    # ---- 7. 空消息 / 默认值省略 ----
    print()
    print("=" * 60)
    print("[7/7] 空消息 & 默认值省略")
    print("=" * 60)

    empty_addr = Address()
    empty_bytes = encode(empty_addr)
    check("空 Address → 零字节", empty_bytes == b"",
          f"got {len(empty_bytes)} B")

    back = decode(Address, b"")
    check("零字节 → 默认 Address", back.city == "" and back.zip_code == "")

    @msg_schema
    class EdgeVals(Message):
        i32 = Field(1, FieldType.INT32)
        i64 = Field(2, FieldType.INT64)
        u32 = Field(3, FieldType.UINT32)
        u64 = Field(4, FieldType.UINT64)
        s32 = Field(5, FieldType.SINT32)
        s64 = Field(6, FieldType.SINT64)

    edge = EdgeVals(
        i32=-1, i64=-2**63, u32=0xFFFFFFFF, u64=0xFFFFFFFFFFFFFFFF,
        s32=-2**31, s64=-2**63,
    )
    edge_bytes = encode(edge)
    print(f"  边界值消息: {edge}")
    print(f"  编码: {edge_bytes.hex(' ')} ({len(edge_bytes)} B)")
    edge_back = decode(EdgeVals, edge_bytes)
    check(f"i32 -1", edge_back.i32 == -1, f"got {edge_back.i32}")
    check(f"i64 -2^63", edge_back.i64 == -2**63, f"got {edge_back.i64}")
    check(f"u32 max", edge_back.u32 == 0xFFFFFFFF, f"got {edge_back.u32}")
    check(f"u64 max", edge_back.u64 == 0xFFFFFFFFFFFFFFFF, f"got {edge_back.u64}")
    check(f"s32 min", edge_back.s32 == -2**31, f"got {edge_back.s32}")
    check(f"s64 min", edge_back.s64 == -2**63, f"got {edge_back.s64}")

    # 再反方向: 最大值
    edge2 = EdgeVals(
        i32=2**31 - 1, i64=2**63 - 1, u32=0, u64=0,
        s32=2**31 - 1, s64=2**63 - 1,
    )
    edge2_back = decode(EdgeVals, encode(edge2))
    check(f"i32 max 2^31-1", edge2_back.i32 == 2**31 - 1, f"got {edge2_back.i32}")
    check(f"i64 max 2^63-1", edge2_back.i64 == 2**63 - 1, f"got {edge2_back.i64}")
    check(f"s32 max 2^31-1", edge2_back.s32 == 2**31 - 1, f"got {edge2_back.s32}")
    check(f"s64 max 2^63-1", edge2_back.s64 == 2**63 - 1, f"got {edge2_back.s64}")

    print()
    print("=" * 60)
    print(f"  总计: {passed} 通过, {failed} 失败")
    print("=" * 60)
    if failed:
        raise SystemExit(1)
    print("🎉 全部通过!")
