"""
tinybuf - 精简版二进制序列化协议编解码器 (v3)
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
    2 = LEN      — string, bytes, 嵌套消息, packed重复数值字段, map entry
    5 = FIXED32  — 固定 4 字节

Packed 编码
-----------
对 repeated 的 VARINT/FIXED32/FIXED64 字段, 可把一组元素合并成一个
LEN 块 (wire_type=2): 先 varint 写出总字节长度, 再依次把所有元素的
编码拼在一起。解码器读一个长度 L 后, 在 L 字节的子缓冲里循环解码直到耗尽。
这样就省去了 N 个重复 Tag 的开销。解码端同时接受 "普通重复 (各自带Tag)"
和 "packed (一个LEN块)" 两种写法, 实现上把这两种 wire_type 都导向同一字段。

Map 字段
--------
map<K,V> 在编码时等价于 repeated Message { K key=1; V value=2; },
每个 entry 是一个独立的 LEN 字段。解码器把所有 entry 还原成一个 dict。
老版本 decoder 看到这个字段号但没定义 map 时, 会按嵌套消息跳过,
不影响向前兼容。

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
from typing import Any, Callable, Dict, ItemsView, List, Optional, Sequence, Tuple, Type, TypeVar, Union

T = TypeVar("T", bound="Message")

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_LEN = 2
WIRE_FIXED32 = 5

_WIRE_NAME = {
    WIRE_VARINT: "VARINT",
    WIRE_FIXED64: "FIXED64",
    WIRE_LEN: "LEN",
    WIRE_FIXED32: "FIXED32",
}


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------

class DecodeError(Exception):
    """tinybuf 解码错误基类。"""


class TruncatedDataError(DecodeError):
    """字节流在字符串/bytes/嵌套消息/fixed字段/varint中间被截断。"""


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

# 哪些字段类型支持 packed 编码
_PACKABLE = {
    FieldType.INT32, FieldType.INT64,
    FieldType.UINT32, FieldType.UINT64,
    FieldType.SINT32, FieldType.SINT64,
    FieldType.BOOL,
    FieldType.FIXED32, FieldType.FIXED64,
}

# map 的 key 只能是标量 (VARINT 或 LEN 字符串)
_MAP_KEY_TYPES = {
    FieldType.INT32, FieldType.INT64,
    FieldType.UINT32, FieldType.UINT64,
    FieldType.SINT32, FieldType.SINT64,
    FieldType.BOOL,
    FieldType.STRING,
    FieldType.FIXED32, FieldType.FIXED64,
}

_MAX_VARINT_BYTES = 10  # 64 bits / 7 ≈ 9.14, 故最多 10 字节


# ---------------------------------------------------------------------------
# Varint — 更严格的 10 字节上限校验
# ---------------------------------------------------------------------------

def encode_varint(value: int) -> bytes:
    """
    将无符号整数编码为 varint。
    每字节: bit7=续接标志(1=还有后续, 0=末尾), bit6-0=数据。
    输入若为负数, 按补码扩展到 64 位无符号数。
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


def decode_varint(stream: BytesIO, ctx: str = "varint",
                  allow_64bit_spill: bool = False) -> int:
    """
    从字节流中解码一个 varint, 返回无符号整数值。

    严格的边界校验:
      - 超过 10 字节 → VarintOverflowError (不管最后是否有终止位)
      - 第 10 字节的高 3 位必须为 0 (64-bit varint 最多只能用到 bit 63)
        除非 allow_64bit_spill=True (用于 skip 等不需要严格校验的场景)
      - 读不满就终止 → TruncatedDataError
    """
    result = 0
    shift = 0
    for byte_index in range(_MAX_VARINT_BYTES):
        raw = stream.read(1)
        if not raw:
            if byte_index == 0:
                raise TruncatedDataError(
                    f"Truncated {ctx}: unexpected end of stream at start of varint"
                )
            raise TruncatedDataError(
                f"Truncated {ctx}: varint cut off at byte {byte_index + 1}"
            )
        b = raw[0]
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            # 第 10 字节 (byte_index==9) 检查高 3 位是否越界
            if byte_index == 9 and not allow_64bit_spill and (b & 0xFE) != 0:
                raise VarintOverflowError(
                    f"{ctx}: 10th byte of varint has bits beyond 64-bit range "
                    f"(byte=0x{b:02x}, only bit 0 allowed for 64-bit varint)"
                )
            return result
        shift += 7
    # 读完 10 字节, 但最后一字节仍设了续位 → 超过 10 字节
    raise VarintOverflowError(
        f"{ctx}: varint exceeds {_MAX_VARINT_BYTES} bytes (>64 bits)"
    )


# ---------------------------------------------------------------------------
# ZigZag — 同时处理 32/64 位
# ---------------------------------------------------------------------------

def zigzag_encode(value: int, bits: int = 64) -> int:
    """ZigZag 编码: 将有符号整数映射到无符号整数, 按 bits 位宽截断。"""
    sign = (value >> (bits - 1)) & 1
    return ((value << 1) ^ (-sign)) & ((1 << bits) - 1)


def zigzag_decode(value: int, bits: int = 64) -> int:
    """ZigZag 解码: 将无符号整数还原为有符号整数, 按 bits 位宽符号扩展。"""
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
# 有符号/无符号整数按位宽截断 & 符号扩展
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
# Field 描述对象 — 支持 map
# ---------------------------------------------------------------------------

@dataclass
class _FieldSpec:
    number: int
    field_type: FieldType
    repeated: bool = False
    packed: bool = False
    message_cls: Optional[Type["Message"]] = None
    is_map: bool = False
    key_type: Optional[FieldType] = None
    value_type: Optional[FieldType] = None
    value_message_cls: Optional[Type["Message"]] = None


class Field:
    """
    在类体中声明字段的描述符。示例::

        @msg_schema
        class Person(Message):
            id = Field(1, FieldType.UINT32)
            scores = Field(9, FieldType.INT32, repeated=True, packed=True)
            attributes = Field(10, map=(FieldType.STRING, FieldType.INT32))
            data_map = Field(11, map=(FieldType.STRING, FieldType.STRING))
    """

    __slots__ = ("_spec",)

    def __init__(self, number: int, field_type: Optional[FieldType] = None,
                 repeated: bool = False,
                 packed: bool = False,
                 message_cls: Optional[Type["Message"]] = None,
                 map: Optional[Tuple[FieldType, FieldType]] = None,
                 value_message_cls: Optional[Type["Message"]] = None):
        if map is not None:
            if field_type is not None:
                raise SchemaError("Cannot specify both field_type= and map=")
            if not isinstance(map, tuple) or len(map) != 2:
                raise SchemaError("map= must be a tuple of (key_type, value_type)")
            key_type, value_type = map
            if key_type not in _MAP_KEY_TYPES:
                raise SchemaError(
                    f"Map key type {key_type} not allowed; must be scalar type"
                )
            if value_type == FieldType.MESSAGE and value_message_cls is None:
                raise SchemaError(
                    "map value type is MESSAGE but no value_message_cls= given"
                )
            if value_type != FieldType.MESSAGE and value_message_cls is not None:
                raise SchemaError(
                    "value_message_cls= given but map value type is not MESSAGE"
                )
            if packed or repeated:
                raise SchemaError(
                    "map= field cannot have repeated=True or packed=True"
                )
            self._spec = _FieldSpec(
                number=number, field_type=FieldType.MESSAGE,
                repeated=True, is_map=True,
                key_type=key_type, value_type=value_type,
                value_message_cls=value_message_cls,
            )
        else:
            if field_type is None:
                raise SchemaError("Must provide field_type= or map=")
            if packed and (not repeated or field_type not in _PACKABLE):
                raise SchemaError(
                    f"packed=True requires repeated=True and a packable scalar type "
                    f"(VARINT/FIXED32/FIXED64), got repeated={repeated}, type={field_type.name}"
                )
            self._spec = _FieldSpec(
                number=number, field_type=field_type,
                repeated=repeated, packed=packed,
                message_cls=message_cls, is_map=False,
            )


# ---------------------------------------------------------------------------
# FieldDescriptor — 支持 map
# ---------------------------------------------------------------------------

@dataclass
class FieldDescriptor:
    number: int
    name: str
    field_type: FieldType
    repeated: bool = False
    packed: bool = False
    message_cls: Optional[Type["Message"]] = None
    is_map: bool = False
    key_type: Optional[FieldType] = None
    value_type: Optional[FieldType] = None
    value_message_cls: Optional[Type["Message"]] = None

    def packable(self) -> bool:
        return self.repeated and not self.is_map and self.field_type in _PACKABLE


# ---------------------------------------------------------------------------
# Message — 支持 introspection / format_schema
# ---------------------------------------------------------------------------

class Message:
    _field_descriptors: Dict[int, FieldDescriptor] = {}
    _field_by_name: Dict[str, FieldDescriptor] = {}

    def __init__(self, **kwargs: Any):
        for desc in self.__class__._field_descriptors.values():
            if desc.is_map:
                setattr(self, desc.name, dict(kwargs.get(desc.name, {})))
            elif desc.repeated:
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
        desc = FieldDescriptor(
            spec.number, name, spec.field_type,
            spec.repeated, spec.packed, spec.message_cls,
            spec.is_map, spec.key_type, spec.value_type, spec.value_message_cls,
        )
        cls._field_descriptors[spec.number] = desc
        cls._field_by_name[name] = desc

    @classmethod
    def define_field(cls, number: int, name: str, field_type: FieldType,
                     repeated: bool = False, packed: bool = False,
                     message_cls: Optional[Type["Message"]] = None,
                     map: Optional[Tuple[FieldType, FieldType]] = None,
                     value_message_cls: Optional[Type["Message"]] = None) -> None:
        """旧风格 API, 保持向后兼容, 也支持 map。"""
        if map is not None:
            if not isinstance(map, tuple) or len(map) != 2:
                raise SchemaError("map must be a tuple of (key_type, value_type)")
            key_type, value_type = map
            if key_type not in _MAP_KEY_TYPES:
                raise SchemaError(f"Map key type {key_type} not allowed")
            if value_type == FieldType.MESSAGE and value_message_cls is None:
                raise SchemaError(
                    "map value type is MESSAGE but no value_message_cls= given"
                )
            spec = _FieldSpec(
                number=number, field_type=FieldType.MESSAGE,
                repeated=True, is_map=True,
                key_type=key_type, value_type=value_type,
                value_message_cls=value_message_cls,
            )
        else:
            if packed and (not repeated or field_type not in _PACKABLE):
                raise SchemaError(
                    "packed=True requires repeated=True and packable scalar type"
                )
            spec = _FieldSpec(
                number=number, field_type=field_type,
                repeated=repeated, packed=packed, message_cls=message_cls,
            )
        cls._register_field(spec, name)

    @classmethod
    def list_fields(cls) -> ItemsView[int, FieldDescriptor]:
        """返回 (field_number, FieldDescriptor) 的迭代视图, 按字段号升序。"""
        return items_view_sorted(cls._field_descriptors)

    @classmethod
    def format_schema(cls, indent: int = 0) -> str:
        """
        导出人类可读的 IDL 风格 schema 文本。
        嵌套消息的结构也会递归展开。
        """
        lines: List[str] = []
        prefix = "  " * indent
        lines.append(f"{prefix}message {cls.__name__} {{")
        for n, desc in sorted(cls._field_descriptors.items()):
            inner_prefix = "  " * (indent + 1)
            if desc.is_map:
                k = desc.key_type.name
                v = (desc.value_message_cls.__name__
                     if desc.value_type == FieldType.MESSAGE
                     else desc.value_type.name)
                lines.append(f"{inner_prefix}map<{k}, {v}> {desc.name} = {n};")
            else:
                parts = []
                if desc.repeated and desc.packed:
                    parts.append("repeated packed")
                elif desc.repeated:
                    parts.append("repeated")
                if desc.field_type == FieldType.MESSAGE:
                    parts.append(desc.message_cls.__name__)
                else:
                    parts.append(desc.field_type.name)
                parts.append(desc.name)
                parts.append(f"= {n};")
                lines.append(f"{inner_prefix}{' '.join(parts)}")
        lines.append(f"{prefix}}}")
        # 递归添加嵌套 message 的 schema
        seen: set = set()
        for desc in cls._field_descriptors.values():
            mc = None
            if desc.field_type == FieldType.MESSAGE and not desc.is_map:
                mc = desc.message_cls
            elif desc.is_map and desc.value_type == FieldType.MESSAGE:
                mc = desc.value_message_cls
            if mc is not None and mc is not cls and mc not in seen:
                seen.add(mc)
                lines.append("")
                lines.append(mc.format_schema(indent))
        return "\n".join(lines)

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


def items_view_sorted(d: Dict[int, FieldDescriptor]) -> ItemsView[int, FieldDescriptor]:
    """返回按 key 排序的 ItemsView 外观 (用于 list_fields)。"""
    return ItemsView(dict(sorted(d.items())))


# ---------------------------------------------------------------------------
# 自动生成的 map entry 消息
# ---------------------------------------------------------------------------

def _make_map_entry_class(desc: FieldDescriptor) -> Type[Message]:
    """
    为 map 字段动态生成 entry 子消息类型:
      message MapEntry { K key = 1; V value = 2; }
    """
    ktype = desc.key_type
    vtype = desc.value_type
    entry_cls_name = f"_MapEntry_{desc.name}_{ktype.name}_{vtype.name}"

    @msg_schema
    class _MapEntry(Message):
        key = Field(1, ktype)
        if vtype == FieldType.MESSAGE:
            value = Field(2, vtype, message_cls=desc.value_message_cls)
        else:
            value = Field(2, vtype)

    _MapEntry.__name__ = entry_cls_name
    _MapEntry.__qualname__ = entry_cls_name
    return _MapEntry


# ---------------------------------------------------------------------------
# @msg_schema 装饰器 — 为每个 map 字段生成 entry 子消息
# ---------------------------------------------------------------------------

def msg_schema(cls: Type[T]) -> Type[T]:
    """
    装饰器: 遍历类体中声明的 Field 描述符, 注册到 _field_descriptors。
    对 map 字段自动生成 entry 子消息类型, 并填到 message_cls 中。
    """
    if not issubclass(cls, Message):
        raise SchemaError(f"@msg_schema can only decorate Message subclasses, got {cls}")

    # 1) 先从基类拷贝已有字段
    inherited: Dict[int, FieldDescriptor] = {}
    for base in cls.__mro__[1:]:
        base_fields = getattr(base, "_field_descriptors", {})
        if base_fields is Message._field_descriptors:
            continue
        for n, desc in base_fields.items():
            if n in inherited:
                continue
            inherited[n] = desc

    # 2) 为当前类创建独立的字段表
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
            try:
                delattr(cls, name)
            except AttributeError:
                pass

    # 4) 按字段号排序注册
    fields_found.sort(key=lambda x: x[1].number)
    for name, spec in fields_found:
        if spec.is_map:
            # 先注册一个临时 desc, 再生成 entry class, 再回填 message_cls
            temp_desc = FieldDescriptor(
                spec.number, name, FieldType.MESSAGE,
                repeated=True, packed=False, message_cls=None,
                is_map=True, key_type=spec.key_type,
                value_type=spec.value_type,
                value_message_cls=spec.value_message_cls,
            )
            entry_cls = _make_map_entry_class(temp_desc)
            spec.message_cls = entry_cls
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
# 单元素编码/解码 — 支持 map entry 的 key/value
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
    """
    把一个 LEN 块 (packed) 解码成该字段的元素列表。
    严格校验: 每个子 varint 必须完整, 不能在子缓冲中间截断。
    """
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
    while stream.tell() < len(payload):
        raw = decode_varint(stream, f"{ctx}.element[{idx}]")
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
# Encoder — 支持 packed + map
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
        if desc.is_map:
            entry_cls = desc.message_cls
            for k, v in value.items():
                entry = entry_cls(key=k, value=v)
                buf.extend(_encode_field(desc, entry))
        elif desc.repeated:
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
    if desc.is_map:
        if isinstance(value, dict) and len(value) == 0:
            return True
        return False
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
# Decoder — map + 严格 packed 校验
# ---------------------------------------------------------------------------

def _skip_field(stream: BytesIO, wire_type: int) -> None:
    """
    根据 wireType 跳过未知字段, 保证向前兼容。
    skip 时允许 64 位 varint 高 3 位不全为 0 (allow_64bit_spill=True),
    因为只需要跳过字节, 不需要值的正确性。
    """
    if wire_type == WIRE_VARINT:
        decode_varint(stream, "skip.varint", allow_64bit_spill=True)
    elif wire_type == WIRE_LEN:
        length = decode_varint(stream, "skip.len.length", allow_64bit_spill=True)
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
    - repeated 数值字段同时接受 packed (LEN) 和普通 (default wire) 两种写法
    - map 字段被还原成 dict, 老版本 decoder 会按嵌套消息跳过
    """
    if stack is None:
        stack = [cls.__name__]
    stream = BytesIO(data)
    kwargs: Dict[str, Any] = {}
    for desc in cls._field_descriptors.values():
        if desc.is_map:
            kwargs[desc.name] = {}
        elif desc.repeated:
            kwargs[desc.name] = []

    expected_wire_defaults: Dict[int, int] = {
        n: _WIRE_MAP[d.field_type] for n, d in cls._field_descriptors.items()
    }

    while True:
        pos = stream.tell()
        try:
            tag_val = decode_varint(
                stream, f"{cls.__name__}.tag@offset={pos}",
                allow_64bit_spill=False,
            )
        except TruncatedDataError:
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

        # ---- map 字段 ----
        if desc.is_map:
            if wire_type != WIRE_LEN:
                _skip_field(stream, wire_type)
                continue
            entry = _decode_single_value(desc, stream, ctx)
            kwargs[desc.name][entry.key] = entry.value
            continue

        # ---- repeated + packable: 接受 packed (LEN) 和普通两种 ----
        if desc.repeated and desc.packable():
            if wire_type == WIRE_LEN:
                length = decode_varint(stream, f"{ctx}.packed.length")
                payload = _assert_read(stream, length, f"{ctx}.packed.payload")
                kwargs[desc.name].extend(_decode_packed_values(desc, payload, ctx))
                continue
            elif wire_type == default_wire:
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
# 诊断 dump — 给一段 bytes 按 schema 逐字段打印
# ---------------------------------------------------------------------------

def _format_value(ft: FieldType, value: Any) -> str:
    if ft == FieldType.BYTES and isinstance(value, bytes):
        return value.hex(" ")
    return repr(value)


def _peek_field_raw(stream: BytesIO, start: int, wire_type: int,
                    ctx: str) -> bytes:
    """从当前位置读一个 field (含 tag) 的原始字节, 读完后定位在字段末尾。"""
    pos = stream.tell()
    if wire_type == WIRE_VARINT:
        decode_varint(stream, f"{ctx}.value", allow_64bit_spill=True)
    elif wire_type == WIRE_LEN:
        length = decode_varint(stream, f"{ctx}.length", allow_64bit_spill=True)
        _assert_read(stream, length, f"{ctx}.payload")
    elif wire_type == WIRE_FIXED32:
        _assert_read(stream, 4, ctx)
    elif wire_type == WIRE_FIXED64:
        _assert_read(stream, 8, ctx)
    else:
        raise WireTypeMismatchError(f"Unknown wire type {wire_type}")
    end = stream.tell()
    # 从源数据切片拿到原始字节
    return stream.getvalue()[start:end]


def dump_data(cls: Type[Message], data: bytes,
              out: Optional[Callable[[str], None]] = None) -> None:
    """
    诊断工具: 给定 bytes 和消息类, 逐字段打印 tag / wire_type / 原始 hex / 值。
    遇到无法识别的字段也会打出来 (标记为 "unknown")。
    """
    if out is None:
        out = print

    out(f"=== tinybuf dump: {cls.__name__} ({len(data)} bytes) ===")
    out(f"  raw hex: {data.hex(' ')}")
    out("")

    stream = BytesIO(data)
    idx = 0
    total_len = len(data)
    while stream.tell() < total_len:
        start = stream.tell()
        try:
            tag_raw = decode_varint(stream, f"tag@{start}", allow_64bit_spill=True)
        except DecodeError as exc:
            out(f"  [{idx:03d}] offset={start:<6} ERROR: {exc}")
            break

        field_number, wire_type = parse_tag(tag_raw)
        tag_end = stream.tell()
        tag_bytes = data[start:tag_end]

        # 先取原始字节 (从 start 开始)
        try:
            raw_bytes = _peek_field_raw(stream, start, wire_type, f"dump@{start}")
        except DecodeError as exc:
            out(f"  [{idx:03d}] offset={start:<6} field={field_number:<4} "
                f"wire={wire_type}({_WIRE_NAME.get(wire_type, '?'):<7}) "
                f"ERROR: {exc}")
            break
        except Exception as exc:
            out(f"  [{idx:03d}] offset={start:<6} field={field_number:<4} "
                f"wire={wire_type}({_WIRE_NAME.get(wire_type, '?'):<7}) "
                f"ERROR: {exc}")
            break

        hex_str = raw_bytes.hex(" ")
        desc = cls._field_descriptors.get(field_number)

        # 再解析值
        try:
            v_stream = BytesIO(raw_bytes)
            # 跳过 tag
            decode_varint(v_stream, "dump.tag", allow_64bit_spill=True)
            if desc is None:
                value_repr = "<unknown>"
            else:
                if desc.is_map:
                    entry = _decode_single_value(desc, v_stream, f"dump.{desc.name}")
                    value_repr = f"entry(key={entry.key!r}, value={entry.value!r})"
                elif desc.repeated and desc.packable() and wire_type == WIRE_LEN:
                    length = decode_varint(v_stream, f"dump.{desc.name}.packed.length")
                    payload = _assert_read(v_stream, length, "dump.packed.payload")
                    values = _decode_packed_values(desc, payload, f"dump.{desc.name}")
                    value_repr = f"packed[{', '.join(repr(v) for v in values)}]"
                else:
                    value = _decode_single_value(desc, v_stream, f"dump.{desc.name}")
                    value_repr = _format_value(desc.field_type, value)
        except DecodeError as exc:
            value_repr = f"<decode error: {exc}>"
        except Exception as exc:
            value_repr = f"<error: {exc}>"

        if desc is None:
            name_str = "?"
        else:
            extra = ""
            if desc.is_map:
                extra = " map"
            elif desc.repeated and desc.packed:
                extra = " repeated packed"
            elif desc.repeated:
                extra = " repeated"
            name_str = f"{desc.name}({desc.field_type.name}{extra})"

        out(f"  [{idx:03d}] offset={start:<6} "
            f"field={field_number:<4} "
            f"wire={wire_type}({_WIRE_NAME.get(wire_type, '?'):<7}) "
            f"name={name_str}")
        out(f"           tag={tag_bytes.hex(' '):<20} "
            f"raw={hex_str}")
        out(f"           value={value_repr}")
        out("")
        idx += 1
    out(f"=== end: {idx} fields ===")


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

    # ---- 1. Schema introspection & IDL 导出 ----
    print("=" * 70)
    print("[1/8] Schema introspection & IDL 导出")
    print("=" * 70)

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
        attributes = Field(11, map=(FieldType.STRING, FieldType.INT32))
        str_map = Field(12, map=(FieldType.STRING, FieldType.STRING))

    print("  --- list_fields(): ---")
    for n, desc in Person.list_fields():
        extra = []
        if desc.is_map:
            extra.append("map")
        if desc.repeated:
            extra.append("repeated")
        if desc.packed:
            extra.append("packed")
        extra_str = f" [{', '.join(extra)}]" if extra else ""
        print(f"    #{n:<3} {desc.name:<12} {desc.field_type.name:<8}{extra_str}")

    print()
    print("  --- format_schema(): ---")
    idl = Person.format_schema()
    print("\n".join("    " + line for line in idl.split("\n")))
    check("schema 导出包含 message", "message Person" in idl)
    check("schema 导出包含 map", "map<STRING, INT32>" in idl)
    check("schema 导出包含 nested message", "message Address" in idl)
    check("schema 导出包含 repeated packed", "repeated packed INT32 scores" in idl)

    # ---- 2. Map 字段 round-trip ----
    print()
    print("=" * 70)
    print("[2/8] Map 字段编码/解码 round-trip")
    print("=" * 70)

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
        attributes={"age": 30, "score": 95, "level": -5},
        str_map={"k1": "v1", "k2": "v2"},
    )

    data = encode(person)
    print(f"  字节长度: {len(data)} bytes")
    restored = decode(Person, data)
    check("map<string,int32> 还原",
          restored.attributes == {"age": 30, "score": 95, "level": -5},
          f"{restored.attributes}")
    check("map<string,string> 还原",
          restored.str_map == {"k1": "v1", "k2": "v2"},
          f"{restored.str_map}")
    check("整体相等", restored == person)

    # ---- 3. 老版本 schema 读取含 map 的数据 (向前兼容) ----
    print()
    print("=" * 70)
    print("[3/8] 老版本 schema 读取含 map 的数据 (向前兼容)")
    print("=" * 70)

    @msg_schema
    class OldPerson(Message):
        """没有定义 attributes 和 str_map 字段的旧版本。"""
        id = Field(1, FieldType.UINT32)
        name = Field(2, FieldType.STRING)
        address = Field(4, FieldType.MESSAGE, message_cls=Address)

    old_view = decode(OldPerson, data)
    check("老版本能读到 id", old_view.id == 42)
    check("老版本能读到 name", old_view.name == "Alice")
    check("老版本能读到 address", old_view.address.city == "Beijing")
    print(f"  老版本解码: {old_view}")
    print("  (attributes / str_map 等新字段被安全跳过)")

    # ---- 4. Packed varint 半字节截断报错 ----
    print()
    print("=" * 70)
    print("[4/8] Packed 半字节 varint 明确报错")
    print("=" * 70)

    @msg_schema
    class PackedInt32(Message):
        values = Field(1, FieldType.INT32, repeated=True, packed=True)

    # 构造一个 packed block: 总长度 5 字节, 前 3 个字节是完整 varint(1),
    # 第 4 字节是续位 (0x80), 但没有第 5 字节 —— 半字节 varint
    packed_body = bytearray()
    packed_body.extend(encode_varint(1))  # 1 = 0x01
    packed_body.append(0x80)              # 半个 varint, 没有后续
    # 打标签: [tag(LEN)][length=4][01 80 ?? ??]
    bad_data = (encode_varint(make_tag(1, WIRE_LEN))
                + encode_varint(len(packed_body))
                + bytes(packed_body))

    try:
        decode(PackedInt32, bad_data)
        check("packed 半字节未报错", False, "应该抛 TruncatedDataError")
    except TruncatedDataError as exc:
        check("packed 半字节 → TruncatedDataError", True, str(exc))

    # ---- 5. 第 10 字节越界 varint 报错 ----
    print()
    print("=" * 70)
    print("[5/8] 第 10 字节 varint 位越界报错")
    print("=" * 70)

    # 合法 10 字节 varint: 所有 64 位都是 1 (uint64 max = 2^64 - 1)
    legal_all_ones = b"\xff" * 9 + b"\x01"
    val_all_ones = decode_varint(BytesIO(legal_all_ones), "test.all_ones")
    check(f"合法 10 字节 varint (all bits set) → {val_all_ones}",
          val_all_ones == 0xFFFFFFFFFFFFFFFF)

    # 合法: 仅 bit 63 为 1 → byte 9 的 bit 0 = 1, 其余全 0
    legal_bit63 = b"\x80" * 9 + b"\x01"  # 前 9 字节: 0x80 (续位+0值), 最后: 0x01 (bit63)
    val_bit63 = decode_varint(BytesIO(legal_bit63), "test.bit63")
    check(f"合法 10 字节 varint (bit63 only) → {val_bit63}", val_bit63 == 2**63)

    # 非法: 第 10 字节的 bit 1 也置 1 (0x03) → 对应整体 bit 64, 超出 64 位
    illegal_bit64 = b"\xff" * 9 + b"\x03"
    try:
        decode_varint(BytesIO(illegal_bit64), "test.illegal")
        check("10 字节越界未报错", False, "应该抛 VarintOverflowError")
    except VarintOverflowError as exc:
        check("10 字节越界 (bit64 set) → VarintOverflowError", True, str(exc))

    # 非法: 超过 10 字节全带续位
    too_long = b"\x80" * 11 + b"\x00"
    try:
        decode_varint(BytesIO(too_long), "test.toolong")
        check(">10 字节未报错", False, "应该抛 VarintOverflowError")
    except VarintOverflowError as exc:
        check(">10 字节 → VarintOverflowError", True, str(exc))

    # ---- 6. 整数边界值再验证 ----
    print()
    print("=" * 70)
    print("[6/8] 整数边界值 (含 map 值里的边界)")
    print("=" * 70)

    person2 = Person(
        attributes={
            "min_int32": -2**31,
            "max_int32": 2**31 - 1,
            "neg_one": -1,
        },
        scores=[-1, -2**31, 2**31 - 1],
    )
    data2 = encode(person2)
    r2 = decode(Person, data2)
    check("map value min_int32", r2.attributes["min_int32"] == -2**31)
    check("map value max_int32", r2.attributes["max_int32"] == 2**31 - 1)
    check("map value neg_one", r2.attributes["neg_one"] == -1)
    check("scores packed 边界", r2.scores == [-1, -2**31, 2**31 - 1])

    # ---- 7. 诊断 dump 工具 ----
    print()
    print("=" * 70)
    print("[7/8] dump_data 诊断工具")
    print("=" * 70)

    sample_bytes = encode(person)
    captured: List[str] = []

    class _Lines:
        def __init__(self):
            self.lines: List[str] = []

        def __call__(self, s: str) -> None:
            self.lines.append(s)

    lines = _Lines()
    dump_data(Person, sample_bytes, out=lines)
    output = "\n".join(lines.lines)
    print("\n".join("  " + line for line in lines.lines[:30]))
    if len(lines.lines) > 30:
        print(f"  ... (共 {len(lines.lines)} 行)")
    check("dump 包含 header", "tinybuf dump: Person" in output)
    check("dump 包含 id 字段", "field=1" in output and "id(" in output)
    check("dump 包含 attributes map", "attributes(" in output and "map" in output)
    check("dump 包含 scores packed", "scores(" in output and "packed[" in output)
    check("dump 包含 str_map", "str_map(" in output and "entry(key=" in output)

    # dump 损坏数据的能力
    print()
    print("  --- dump 含 packed 半字节损坏的数据: ---")
    captured2: List[str] = []
    dump_data(PackedInt32, bad_data, out=lambda s: captured2.append(s))
    dump_output = "\n".join(captured2)
    print("\n".join("  " + line for line in captured2))
    check("dump 错误数据也能显示", "ERROR" in dump_output or "decode error" in dump_output.lower())

    # ---- 8. 空消息 & 空 map & packed 空数组 ----
    print()
    print("=" * 70)
    print("[8/8] 空消息 & 空 map & packed 空数组省略")
    print("=" * 70)

    empty = Person()
    empty_bytes = encode(empty)
    check("空 Person → 0 字节", empty_bytes == b"", f"got {len(empty_bytes)} B")
    back_empty = decode(Person, b"")
    check("0 字节 → 默认 Person",
          back_empty.attributes == {}
          and back_empty.str_map == {}
          and back_empty.scores == []
          and back_empty.id == 0
          and back_empty.active is False)

    # 空数组 / 空 map 不编码
    only_empty = Person(attributes={}, str_map={}, scores=[], flags=[], tags=[])
    only_empty_bytes = encode(only_empty)
    check("只有空容器 → 0 字节", only_empty_bytes == b"",
          f"got {len(only_empty_bytes)} B")

    print()
    print("=" * 70)
    print(f"  总计: {passed} 通过, {failed} 失败")
    print("=" * 70)
    if failed:
        raise SystemExit(1)
    print("🎉 全部通过!")
