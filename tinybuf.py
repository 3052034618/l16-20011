"""
tinybuf - 精简版二进制序列化协议编解码器 (v4)
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

Oneof 字段
----------
oneof { ... } 声明一组互斥的字段, 编码时只允许其中一个有值。
解码时若数据流里出现多个 oneof 字段 (例如旧版本数据混发),
保留最后一次出现的值 ("最后写入者胜"), 与 protobuf 一致。

Required / Optional & Has Field
-------------------------------
required 字段: encode 之前如果未填充, 抛出 RequiredFieldError。
optional 字段: 解码后能区分 "没出现" 和 "出现了但等于默认值"。
通过 has_field(name) / _fields_present set 追踪。

Varint 编码原理
---------------
每个字节的最高位 (MSB, bit 7) 是 "续接标志":
  - 1 = 后面还有更多字节
  - 0 = 这是最后一个字节
低 7 位承载实际数据, 小端序排列 (little-endian group)。

ZigZag 编码原理
---------------
把有符号整数映射到无符号整数, 使绝对值小的数 (无论正负)
都映射到小的无符号值, 从而 varint 编码紧凑。

字段标签与向前兼容
-----------------
解码时先读 Tag (varint), 从中提取 wire_type 和 field_number。
若 field_number 在当前 schema 中未知, 只需根据 wire_type 跳过
固定长度即可 → 向前兼容。

嵌套消息的长度界定
-----------------
嵌套消息 wire_type = LEN (2), Value 部分先 varint 写出字节长度 L,
再跟 L 字节的子消息体。解码器在子缓冲上递归解码。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from io import BytesIO
from typing import (
    Any, Callable, Dict, ItemsView, Iterable, List, Optional, Sequence,
    Set, Tuple, Type, TypeVar, Union,
)

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


class EncodeError(Exception):
    """encode 时的业务校验错误基类。"""


class RequiredFieldError(EncodeError):
    """encode 时 required 字段缺失。"""


class OneofConflictError(EncodeError):
    """encode 时同一 oneof 组中同时设置了多个字段。"""


class CompatLevel(IntEnum):
    """两个 schema 版本之间的兼容性等级。"""
    FULLY_COMPATIBLE = 0  # 无任何破坏性改动
    SAFE_EXTENSION = 1    # 仅新增字段 (安全)
    WARNING = 2           # 有破坏性但不致命 (如 optional 互改)
    BREAKING = 3          # 破坏性改动 (号/类型改了, 删除了 required)


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

_PACKABLE = {
    FieldType.INT32, FieldType.INT64,
    FieldType.UINT32, FieldType.UINT64,
    FieldType.SINT32, FieldType.SINT64,
    FieldType.BOOL,
    FieldType.FIXED32, FieldType.FIXED64,
}

_MAP_KEY_TYPES = {
    FieldType.INT32, FieldType.INT64,
    FieldType.UINT32, FieldType.UINT64,
    FieldType.SINT32, FieldType.SINT64,
    FieldType.BOOL,
    FieldType.STRING,
    FieldType.FIXED32, FieldType.FIXED64,
}

_MAX_VARINT_BYTES = 10


# ---------------------------------------------------------------------------
# Varint
# ---------------------------------------------------------------------------

def encode_varint(value: int) -> bytes:
    if value < 0:
        value &= 0xFFFFFFFFFFFFFFFF
    buf = bytearray()
    while value > 0x7F:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.append(value & 0x7F)
    return bytes(buf)


def _assert_read(stream: BytesIO, n: int, field: str) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise TruncatedDataError(
            f"Truncated {field}: expected {n} bytes, got {len(data)}"
        )
    return data


def decode_varint(stream: BytesIO, ctx: str = "varint",
                  allow_64bit_spill: bool = False) -> int:
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
            if byte_index == 9 and not allow_64bit_spill and (b & 0xFE) != 0:
                raise VarintOverflowError(
                    f"{ctx}: 10th byte of varint has bits beyond 64-bit range "
                    f"(byte=0x{b:02x}, only bit 0 allowed for 64-bit varint)"
                )
            return result
        shift += 7
    raise VarintOverflowError(
        f"{ctx}: varint exceeds {_MAX_VARINT_BYTES} bytes (>64 bits)"
    )


# ---------------------------------------------------------------------------
# ZigZag
# ---------------------------------------------------------------------------

def zigzag_encode(value: int, bits: int = 64) -> int:
    sign = (value >> (bits - 1)) & 1
    return ((value << 1) ^ (-sign)) & ((1 << bits) - 1)


def zigzag_decode(value: int, bits: int = 64) -> int:
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
# 有符号/无符号按位宽转换
# ---------------------------------------------------------------------------

def _to_signed(value: int, bits: int) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")
    value &= (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    if value & sign_bit:
        return value - (1 << bits)
    return value


def _to_unsigned(value: int, bits: int) -> int:
    return value & ((1 << bits) - 1)


# ---------------------------------------------------------------------------
# _FieldSpec / Field / OneofSpec / Oneof
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
    required: bool = False
    oneof: Optional[str] = None


class Field:
    """
    在类体中声明字段的描述符。示例::

        @msg_schema
        class Person(Message):
            id = Field(1, FieldType.UINT32, required=True)
            scores = Field(9, FieldType.INT32, repeated=True, packed=True)
            attributes = Field(10, map=(FieldType.STRING, FieldType.INT32))
    """

    __slots__ = ("_spec",)

    def __init__(self, number: int, field_type: Optional[FieldType] = None,
                 repeated: bool = False,
                 packed: bool = False,
                 message_cls: Optional[Type["Message"]] = None,
                 map: Optional[Tuple[FieldType, FieldType]] = None,
                 value_message_cls: Optional[Type["Message"]] = None,
                 required: bool = False,
                 oneof: Optional[str] = None):
        if map is not None:
            if field_type is not None:
                raise SchemaError("Cannot specify both field_type= and map=")
            if not isinstance(map, tuple) or len(map) != 2:
                raise SchemaError("map= must be a tuple of (key_type, value_type)")
            key_type, value_type = map
            if key_type not in _MAP_KEY_TYPES:
                raise SchemaError(f"Map key type {key_type} not allowed")
            if value_type == FieldType.MESSAGE and value_message_cls is None:
                raise SchemaError(
                    "map value type is MESSAGE but no value_message_cls= given"
                )
            if value_type != FieldType.MESSAGE and value_message_cls is not None:
                raise SchemaError(
                    "value_message_cls= given but map value type is not MESSAGE"
                )
            if packed or repeated or required or oneof:
                raise SchemaError(
                    "map= field cannot use repeated/packed/required/oneof"
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
                    f"packed=True requires repeated=True and packable scalar type, "
                    f"got repeated={repeated}, type={field_type.name}"
                )
            if repeated and oneof:
                raise SchemaError(
                    f"Field #{number}: repeated fields cannot be inside oneof"
                )
            if field_type == FieldType.MESSAGE and oneof:
                # oneof message 是允许的 (protobuf 也支持)
                pass
            self._spec = _FieldSpec(
                number=number, field_type=field_type,
                repeated=repeated, packed=packed,
                message_cls=message_cls,
                required=required, oneof=oneof,
            )


@dataclass
class _OneofSpec:
    name: str
    fields: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FieldDescriptor
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
    required: bool = False
    oneof: Optional[str] = None

    def packable(self) -> bool:
        return self.repeated and not self.is_map and self.field_type in _PACKABLE


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

def items_view_sorted(d: Dict[int, FieldDescriptor]) -> ItemsView[int, FieldDescriptor]:
    return ItemsView(dict(sorted(d.items())))


def _make_map_entry_class(desc: FieldDescriptor) -> Type["Message"]:
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


class Message:
    _field_descriptors: Dict[int, FieldDescriptor] = {}
    _field_by_name: Dict[str, FieldDescriptor] = {}
    _oneofs: Dict[str, _OneofSpec] = {}

    def __init__(self, **kwargs: Any):
        self._fields_present: Set[str] = set()

        for desc in self.__class__._field_descriptors.values():
            if desc.is_map:
                value = dict(kwargs.get(desc.name, {}))
                setattr(self, desc.name, value)
                if value:
                    self._fields_present.add(desc.name)
            elif desc.repeated:
                value = list(kwargs.get(desc.name, []))
                setattr(self, desc.name, value)
                if value:
                    self._fields_present.add(desc.name)
            else:
                if desc.name in kwargs:
                    value = kwargs[desc.name]
                    self._fields_present.add(desc.name)
                    setattr(self, desc.name, value)
                else:
                    setattr(self, desc.name, self._default(desc))

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

    # ---------- field presence tracking ----------

    def has_field(self, name: str) -> bool:
        """
        判断字段 "是否出现在消息中"。
         - 标量字段: 返回 True 仅当构造时显式传入 OR 解码时遇到过该字段
         - repeated/map: 返回 True 当列表/字典非空
         - MESSAGE: 返回 True 当构造时显式传入 OR 解码时遇到 (即使子消息是 default)
        """
        if name not in self.__class__._field_by_name:
            raise KeyError(f"No such field: {name}")
        return name in self._fields_present

    def set_field_present(self, name: str, present: bool = True) -> None:
        if name not in self.__class__._field_by_name:
            raise KeyError(f"No such field: {name}")
        if present:
            self._fields_present.add(name)
        else:
            self._fields_present.discard(name)

    def clear_field(self, name: str) -> None:
        """把字段置为默认值并标记为未出现。"""
        if name not in self.__class__._field_by_name:
            raise KeyError(f"No such field: {name}")
        desc = self.__class__._field_by_name[name]
        if desc.is_map:
            setattr(self, name, {})
        elif desc.repeated:
            setattr(self, name, [])
        else:
            setattr(self, name, self._default(desc))
        self._fields_present.discard(name)

    def which_oneof(self, name: str) -> Optional[str]:
        """返回该 oneof 组里当前有值的字段名, 都没值返回 None。"""
        if name not in self.__class__._oneofs:
            raise KeyError(f"No such oneof: {name}")
        for field_num in self.__class__._oneofs[name].fields:
            fname = self.__class__._field_descriptors[field_num].name
            if self.has_field(fname):
                return fname
        return None

    # ---------- 验证 ----------

    def validate(self) -> None:
        """
        encode 之前的业务校验:
          1) required 字段必须存在或非默认
          2) 同一 oneof 组不得有多个字段同时被显式设置
        """
        cls = self.__class__

        # 1. required
        missing_required: List[str] = []
        for desc in cls._field_descriptors.values():
            if desc.required:
                if not self.has_field(desc.name):
                    missing_required.append(desc.name)
                else:
                    v = getattr(self, desc.name)
                    if desc.repeated and len(v) == 0:
                        missing_required.append(desc.name)
                    if desc.is_map and len(v) == 0:
                        missing_required.append(desc.name)
        if missing_required:
            raise RequiredFieldError(
                f"Required fields not set: {', '.join(missing_required)}"
            )

        # 2. oneof conflict (构造时同时显式 set 了多个)
        for oname, ospec in cls._oneofs.items():
            set_fields = [
                cls._field_descriptors[n].name
                for n in ospec.fields
                if self.has_field(cls._field_descriptors[n].name)
            ]
            if len(set_fields) > 1:
                raise OneofConflictError(
                    f"Oneof group '{oname}': multiple fields set "
                    f"({', '.join(set_fields)}); only one may be set"
                )

    # ---------- 注册 ----------

    @classmethod
    def _register_field(cls, spec: _FieldSpec, name: str) -> None:
        if not hasattr(cls, '_field_descriptors') or cls._field_descriptors is Message._field_descriptors:
            cls._field_descriptors = {}
            cls._field_by_name = {}
            cls._oneofs = {}
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
            spec.required, spec.oneof,
        )
        cls._field_descriptors[spec.number] = desc
        cls._field_by_name[name] = desc

        # 注册 oneof
        if spec.oneof:
            if spec.oneof not in cls._oneofs:
                cls._oneofs[spec.oneof] = _OneofSpec(spec.oneof)
            cls._oneofs[spec.oneof].fields.append(spec.number)

    @classmethod
    def define_field(cls, number: int, name: str, field_type: FieldType,
                     repeated: bool = False, packed: bool = False,
                     message_cls: Optional[Type["Message"]] = None,
                     map: Optional[Tuple[FieldType, FieldType]] = None,
                     value_message_cls: Optional[Type["Message"]] = None,
                     required: bool = False,
                     oneof: Optional[str] = None) -> None:
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
                required=required, oneof=oneof,
            )
        cls._register_field(spec, name)

    # ---------- introspection ----------

    @classmethod
    def list_fields(cls) -> ItemsView[int, FieldDescriptor]:
        return items_view_sorted(cls._field_descriptors)

    @classmethod
    def list_oneofs(cls) -> Dict[str, List[str]]:
        return {
            name: [cls._field_descriptors[n].name for n in ospec.fields]
            for name, ospec in cls._oneofs.items()
        }

    @classmethod
    def format_schema(cls, indent: int = 0) -> str:
        """导出 IDL 风格的 schema 文本, 递归展开嵌套 message。"""
        lines: List[str] = []
        prefix = "  " * indent
        lines.append(f"{prefix}message {cls.__name__} {{")

        ip = indent + 1
        iprefix = "  " * ip
        iiprefix = "  " * (indent + 2)

        oneof_emitted_groups: Set[str] = set()
        standalone_sorted = sorted(
            [d for d in cls._field_descriptors.values() if not d.oneof],
            key=lambda d: d.number,
        )

        all_items: List[Tuple[int, Any]] = []
        for d in standalone_sorted:
            all_items.append((d.number, ("field", d)))
        oname_groups: Dict[str, List[FieldDescriptor]] = {}
        for d in cls._field_descriptors.values():
            if d.oneof:
                oname_groups.setdefault(d.oneof, []).append(d)
        for oname, group in oname_groups.items():
            min_n = min(d.number for d in group)
            all_items.append((min_n, ("oneof", oname, group)))

        all_items.sort(key=lambda x: x[0])

        for _n, item in all_items:
            if item[0] == "field":
                lines.append(_format_field_line(item[1], ip))
            else:
                oname, group = item[1], item[2]
                lines.append(f"{iprefix}oneof {oname} {{")
                for gd in sorted(group, key=lambda d: d.number):
                    lines.append(_format_field_line(gd, indent + 2))
                lines.append(f"{iprefix}}}")

        lines.append(f"{prefix}}}")

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

    # ---------- repr / eq ----------

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


def _format_field_line(desc: FieldDescriptor, indent: int) -> str:
    p = "  " * indent
    if desc.is_map:
        k = desc.key_type.name
        v = (desc.value_message_cls.__name__
             if desc.value_type == FieldType.MESSAGE
             else desc.value_type.name)
        req = " required" if desc.required else ""
        return f"{p}map<{k}, {v}> {desc.name} = {desc.number}{req};"

    parts: List[str] = []
    if desc.required:
        parts.append("required")
    elif desc.repeated and desc.packed:
        parts.append("repeated packed")
    elif desc.repeated:
        parts.append("repeated")
    elif not desc.is_map:
        parts.append("optional")

    if desc.field_type == FieldType.MESSAGE:
        parts.append(desc.message_cls.__name__)
    else:
        parts.append(desc.field_type.name)

    parts.append(desc.name)
    parts.append(f"= {desc.number};")
    return f"{p}{' '.join(parts)}"


# ---------------------------------------------------------------------------
# @msg_schema 装饰器
# ---------------------------------------------------------------------------

def msg_schema(cls: Type[T]) -> Type[T]:
    if not issubclass(cls, Message):
        raise SchemaError(f"@msg_schema can only decorate Message subclasses")

    inherited_fields: Dict[int, FieldDescriptor] = {}
    inherited_oneofs: Dict[str, _OneofSpec] = {}
    for base in cls.__mro__[1:]:
        base_fields = getattr(base, "_field_descriptors", {})
        if base_fields is Message._field_descriptors:
            continue
        for n, desc in base_fields.items():
            if n in inherited_fields:
                continue
            inherited_fields[n] = desc
        base_oneofs = getattr(base, "_oneofs", {})
        if base_oneofs is not Message._oneofs:
            for oname, ospec in base_oneofs.items():
                if oname not in inherited_oneofs:
                    inherited_oneofs[oname] = _OneofSpec(
                        oname, list(ospec.fields)
                    )

    cls._field_descriptors = {}
    cls._field_by_name = {}
    cls._oneofs = {}
    for n, desc in inherited_fields.items():
        cls._field_descriptors[n] = desc
        cls._field_by_name[desc.name] = desc
    for oname, ospec in inherited_oneofs.items():
        cls._oneofs[oname] = _OneofSpec(oname, list(ospec.fields))

    fields_found: List[Tuple[str, _FieldSpec]] = []
    for name, attr in list(vars(cls).items()):
        if isinstance(attr, Field):
            fields_found.append((name, attr._spec))
            try:
                delattr(cls, name)
            except AttributeError:
                pass

    fields_found.sort(key=lambda x: x[1].number)
    for name, spec in fields_found:
        if spec.is_map:
            temp_desc = FieldDescriptor(
                spec.number, name, FieldType.MESSAGE,
                repeated=True, is_map=True,
                key_type=spec.key_type, value_type=spec.value_type,
                value_message_cls=spec.value_message_cls,
            )
            spec.message_cls = _make_map_entry_class(temp_desc)
        if spec.field_type == FieldType.MESSAGE and spec.message_cls is None:
            raise SchemaError(
                f"Field '{cls.__name__}.{name}' (number={spec.number}) is MESSAGE "
                f"but no message_cls= given"
            )
        cls._register_field(spec, name)

    return cls


# ---------------------------------------------------------------------------
# 单元素编码 / 解码
# ---------------------------------------------------------------------------

def _encode_single_value(field_type: FieldType,
                         message_cls: Optional[Type[Message]],
                         value: Any) -> bytes:
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
    ft = desc.field_type
    if ft == FieldType.INT32:
        raw = decode_varint(stream, ctx)
        if raw > 0xFFFFFFFF:
            raise VarintOverflowError(f"{ctx}: value {raw} exceeds uint32 range")
        return _to_signed(raw, 32)
    if ft == FieldType.INT64:
        return _to_signed(decode_varint(stream, ctx), 64)
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
        return zigzag_decode(decode_varint(stream, ctx), 64)
    if ft == FieldType.BOOL:
        return decode_varint(stream, ctx) != 0
    if ft == FieldType.FIXED32:
        return struct.unpack("<I", _assert_read(stream, 4, ctx))[0]
    if ft == FieldType.FIXED64:
        return struct.unpack("<Q", _assert_read(stream, 8, ctx))[0]
    if ft == FieldType.STRING:
        length = decode_varint(stream, f"{ctx}.length")
        raw = _assert_read(stream, length, f"{ctx}.payload")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecodeError(f"{ctx}: invalid UTF-8: {exc}") from exc
    if ft == FieldType.BYTES:
        length = decode_varint(stream, f"{ctx}.length")
        return _assert_read(stream, length, f"{ctx}.payload")
    if ft == FieldType.MESSAGE:
        length = decode_varint(stream, f"{ctx}.length")
        sub_buf = _assert_read(stream, length, f"{ctx}.payload")
        return decode(desc.message_cls, sub_buf,
                      stack=[desc.message_cls.__name__])
    raise ValueError(f"Unknown field type: {ft}")


def _decode_packed_values(desc: FieldDescriptor, payload: bytes,
                          ctx: str) -> List[Any]:
    stream = BytesIO(payload)
    items: List[Any] = []
    ft = desc.field_type
    if ft in (FieldType.FIXED32, FieldType.FIXED64):
        size = 4 if ft == FieldType.FIXED32 else 8
        if len(payload) % size != 0:
            raise TruncatedDataError(
                f"{ctx}: packed {ft.name} block length {len(payload)} "
                f"not divisible by {size}"
            )
        while True:
            raw = stream.read(size)
            if not raw:
                break
            fmt = "<I" if size == 4 else "<Q"
            items.append(struct.unpack(fmt, raw)[0])
        return items
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
# Encoder
# ---------------------------------------------------------------------------

def _encode_value(desc: FieldDescriptor, value: Any) -> bytes:
    return _encode_single_value(desc.field_type, desc.message_cls, value)


def _encode_field(desc: FieldDescriptor, value: Any) -> bytes:
    wire_type = _WIRE_MAP[desc.field_type]
    tag_bytes = encode_varint(make_tag(desc.number, wire_type))
    value_bytes = _encode_value(desc, value)
    return tag_bytes + value_bytes


def _encode_packed_field(desc: FieldDescriptor, values: Sequence[Any]) -> bytes:
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


def encode(msg: Message, validate: bool = True) -> bytes:
    """
    序列化消息。validate=True 时在编码前调用 msg.validate()
    检查 required 字段和 oneof 冲突。
    """
    if validate:
        msg.validate()

    buf = bytearray()
    cls = msg.__class__
    for desc in cls._field_descriptors.values():
        value = getattr(msg, desc.name)
        if desc.is_map:
            if not value:
                continue
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
            if not msg.has_field(desc.name):
                continue
            buf.extend(_encode_field(desc, value))
    return bytes(buf)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def _skip_field(stream: BytesIO, wire_type: int) -> None:
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
        raise WireTypeMismatchError(
            f"Unknown wire type {wire_type} while skipping field"
        )


def decode(cls: Type[T], data: bytes,
           stack: Optional[List[str]] = None) -> T:
    """
    反序列化。返回的消息中 _fields_present 会记录哪些字段实际出现过,
    可通过 has_field() 查询。oneof 组若出现多次, 保留最后写入的值。
    """
    if stack is None:
        stack = [cls.__name__]
    stream = BytesIO(data)
    kwargs: Dict[str, Any] = {}
    present: Set[str] = set()

    for desc in cls._field_descriptors.values():
        if desc.is_map:
            kwargs[desc.name] = {}
        elif desc.repeated:
            kwargs[desc.name] = []

    expected_wire_defaults: Dict[int, int] = {
        n: _WIRE_MAP[d.field_type]
        for n, d in cls._field_descriptors.items()
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

        # oneof: 如果该字段属于 oneof, 先清理同组其他字段的 presence
        if desc.oneof:
            ospec = cls._oneofs[desc.oneof]
            for other_num in ospec.fields:
                other_name = cls._field_descriptors[other_num].name
                if other_name != desc.name:
                    present.discard(other_name)

        if desc.is_map:
            if wire_type != WIRE_LEN:
                _skip_field(stream, wire_type)
                continue
            entry = _decode_single_value(desc, stream, ctx)
            kwargs[desc.name][entry.key] = entry.value
            present.add(desc.name)
            continue

        if desc.repeated and desc.packable():
            if wire_type == WIRE_LEN:
                length = decode_varint(stream, f"{ctx}.packed.length")
                payload = _assert_read(stream, length, f"{ctx}.packed.payload")
                kwargs[desc.name].extend(
                    _decode_packed_values(desc, payload, ctx)
                )
                present.add(desc.name)
                continue
            elif wire_type == default_wire:
                value = _decode_single_value(desc, stream, ctx)
                kwargs[desc.name].append(value)
                present.add(desc.name)
                continue
            else:
                raise WireTypeMismatchError(
                    f"{ctx}: wire type {wire_type} not accepted; "
                    f"expected {default_wire} or {WIRE_LEN}"
                )

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
            present.add(desc.name)
        else:
            kwargs[desc.name] = value
            present.add(desc.name)

    instance = cls.__new__(cls)
    # 先初始化 _fields_present
    instance._fields_present = present

    # 再填属性, 但默认值不走 has_field set
    for desc in cls._field_descriptors.values():
        if desc.name in kwargs:
            setattr(instance, desc.name, kwargs[desc.name])
        else:
            if desc.is_map:
                setattr(instance, desc.name, {})
            elif desc.repeated:
                setattr(instance, desc.name, [])
            else:
                setattr(instance, desc.name, Message._default(desc))

    return instance


# ---------------------------------------------------------------------------
# 兼容性检查 (Schema Compatibility Checker)
# ---------------------------------------------------------------------------

@dataclass
class FieldDiff:
    field_number: int
    change: str  # "added" | "removed" | "renamed" | "type_changed" |
                 # "label_changed" | "required_changed" | "number_moved"
    old_name: Optional[str] = None
    new_name: Optional[str] = None
    old_type: Optional[FieldType] = None
    new_type: Optional[FieldType] = None
    level: CompatLevel = CompatLevel.FULLY_COMPATIBLE
    detail: str = ""


@dataclass
class CompatReport:
    old_cls_name: str
    new_cls_name: str
    diffs: List[FieldDiff]
    overall: CompatLevel

    def worst(self) -> CompatLevel:
        return max((d.level for d in self.diffs),
                   default=CompatLevel.FULLY_COMPATIBLE)


def diff_schemas(old_cls: Type[Message], new_cls: Type[Message]) -> List[FieldDiff]:
    """
    对比两个 schema, 返回字段差异列表。
    对比逻辑: 以 field_number 为主键 (这是线上协议的主键),
              field_name / type / label 变化都算改动。
    """
    diffs: List[FieldDiff] = []

    old_by_num: Dict[int, FieldDescriptor] = dict(old_cls._field_descriptors)
    new_by_num: Dict[int, FieldDescriptor] = dict(new_cls._field_descriptors)

    # 建立 name→number 映射以辅助重命名检测
    old_names = {d.name: n for n, d in old_by_num.items()}
    new_names = {d.name: n for n, d in new_by_num.items()}

    removed_nums = set(old_by_num.keys()) - set(new_by_num.keys())
    added_nums = set(new_by_num.keys()) - set(old_by_num.keys())

    # 尝试配对 removed + added: 同一个 name 出现在不同 number → 改号
    renamed_pairs: List[Tuple[int, int]] = []
    processed_removed: Set[int] = set()
    processed_added: Set[int] = set()
    for oname, onum in old_names.items():
        if oname in new_names and onum != new_names[oname]:
            nnum = new_names[oname]
            if onum in removed_nums and nnum in added_nums:
                renamed_pairs.append((onum, nnum))
                processed_removed.add(onum)
                processed_added.add(nnum)
                od = old_by_num[onum]
                nd = new_by_num[nnum]
                level = CompatLevel.BREAKING
                detail = f"字段号从 {onum} 改成 {nnum} — 线上不兼容"
                if od.field_type != nd.field_type:
                    detail += f", 类型也从 {od.field_type.name} 改成 {nd.field_type.name}"
                if od.is_map or od.repeated or nd.is_map or nd.repeated:
                    detail += " (label 也不同)"
                diffs.append(FieldDiff(
                    field_number=nnum, change="number_moved",
                    old_name=oname, new_name=oname, level=level, detail=detail,
                ))

    for onum in removed_nums - processed_removed:
        od = old_by_num[onum]
        level = CompatLevel.BREAKING if od.required else CompatLevel.WARNING
        detail = f"字段 #{onum} '{od.name}' ({od.field_type.name}) 删除"
        if od.required:
            detail += " — ⚠️ 该字段是 required, 删除是破坏性改动"
        diffs.append(FieldDiff(
            field_number=onum, change="removed",
            old_name=od.name, old_type=od.field_type,
            level=level, detail=detail,
        ))

    for nnum in added_nums - processed_added:
        nd = new_by_num[nnum]
        level = (CompatLevel.SAFE_EXTENSION
                 if not nd.required else CompatLevel.WARNING)
        detail = (f"字段 #{nnum} '{nd.name}' ({nd.field_type.name}) 新增"
                  + (" — ⚠️ 新增 required 字段对旧数据不兼容" if nd.required else ""))
        diffs.append(FieldDiff(
            field_number=nnum, change="added",
            new_name=nd.name, new_type=nd.field_type,
            level=level, detail=detail,
        ))

    # 相同 number 的字段改动
    for num in set(old_by_num.keys()) & set(new_by_num.keys()):
        od = old_by_num[num]
        nd = new_by_num[num]

        # 改名 (仅 name 变, type/label 没改)
        if od.name != nd.name:
            diffs.append(FieldDiff(
                field_number=num, change="renamed",
                old_name=od.name, new_name=nd.name,
                level=CompatLevel.FULLY_COMPATIBLE,
                detail=f"字段 #{num} 名从 '{od.name}' 改成 '{nd.name}' — 线上兼容",
            ))

        # 类型变化
        if od.field_type != nd.field_type:
            diffs.append(FieldDiff(
                field_number=num, change="type_changed",
                old_name=od.name, new_name=nd.name,
                old_type=od.field_type, new_type=nd.field_type,
                level=CompatLevel.BREAKING,
                detail=f"字段 #{num} ('{od.name}'→'{nd.name}') 类型从 "
                       f"{od.field_type.name} 改成 {nd.field_type.name} — 破坏性!",
            ))

        # label 变化 (repeated/map/packed/oneof)
        old_label = (
            "map" if od.is_map else
            f"repeated packed" if od.packed else
            "repeated" if od.repeated else
            f"oneof({od.oneof})" if od.oneof else
            "scalar"
        )
        new_label = (
            "map" if nd.is_map else
            f"repeated packed" if nd.packed else
            "repeated" if nd.repeated else
            f"oneof({nd.oneof})" if nd.oneof else
            "scalar"
        )
        if ((od.is_map != nd.is_map)
                or (od.repeated != nd.repeated)
                or (od.packed != nd.packed)
                or (od.oneof != nd.oneof)):
            # repeated ↔ packed 通常兼容 (wire_type 通吃)
            level = CompatLevel.WARNING
            detail = (f"字段 #{num} label 从 '{old_label}' 改成 '{new_label}'"
                      + " — 建议仔细检查")
            # repeated ↔ scalar 或 map ↔ scalar 是破坏性的
            if (od.repeated != nd.repeated) or (od.is_map != nd.is_map):
                level = CompatLevel.BREAKING
                detail += " ⚠️ repeated/map ↔ scalar 是破坏性改动"
            diffs.append(FieldDiff(
                field_number=num, change="label_changed",
                old_name=od.name, new_name=nd.name,
                level=level, detail=detail,
            ))

        # required 变化
        if od.required != nd.required:
            if od.required and not nd.required:
                level = CompatLevel.SAFE_EXTENSION
                detail = f"字段 #{num} 从 required 改成 optional — 安全 (放宽约束)"
            else:
                level = CompatLevel.BREAKING
                detail = f"字段 #{num} 从 optional 改成 required — 旧数据没这个字段会失败"
            diffs.append(FieldDiff(
                field_number=num, change="required_changed",
                old_name=od.name, new_name=nd.name,
                level=level, detail=detail,
            ))

    return sorted(diffs, key=lambda d: (
        -d.level.value, d.field_number, d.change
    ))


def check_compatibility(old_cls: Type[Message],
                        new_cls: Type[Message]) -> CompatReport:
    diffs = diff_schemas(old_cls, new_cls)
    worst = max((d.level for d in diffs),
                default=CompatLevel.FULLY_COMPATIBLE)
    return CompatReport(
        old_cls_name=old_cls.__name__,
        new_cls_name=new_cls.__name__,
        diffs=diffs,
        overall=worst,
    )


_COMPAT_MARKS = {
    CompatLevel.FULLY_COMPATIBLE: "✅",
    CompatLevel.SAFE_EXTENSION:  "➕",
    CompatLevel.WARNING:         "⚠️",
    CompatLevel.BREAKING:        "💥",
}

_COMPAT_TITLES = {
    CompatLevel.FULLY_COMPATIBLE: "完全兼容 — 无任何破坏性改动",
    CompatLevel.SAFE_EXTENSION:  "安全扩展 — 仅新增字段",
    CompatLevel.WARNING:         "有警告 — 请仔细评估",
    CompatLevel.BREAKING:        "破坏性改动 — 不要直接上线!",
}


def format_compat_report(report: CompatReport) -> str:
    """生成人类可读的兼容性报告。"""
    lines: List[str] = []
    title = f"Schema 兼容性报告: {report.old_cls_name} → {report.new_cls_name}"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(
        f"总体评估: {_COMPAT_MARKS[report.overall]} "
        f"{_COMPAT_TITLES[report.overall]}"
    )
    lines.append("")
    if not report.diffs:
        lines.append("  (完全没有字段变化)")
        return "\n".join(lines)

    # 按等级分组
    for level in (CompatLevel.BREAKING, CompatLevel.WARNING,
                  CompatLevel.SAFE_EXTENSION, CompatLevel.FULLY_COMPATIBLE):
        items = [d for d in report.diffs if d.level == level]
        if not items:
            continue
        lines.append(f"  {_COMPAT_MARKS[level]} {level.name}:")
        for d in items:
            lines.append(f"    - #{d.field_number:<3} {d.change:<18} {d.detail}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# 诊断 dump_data — 升级: 损坏定位 + 概览 + only_fields 过滤
# ---------------------------------------------------------------------------

@dataclass
class DumpResult:
    fields: List[Dict[str, Any]] = field(default_factory=list)
    error_at_offset: Optional[int] = None
    error_message: Optional[str] = None
    total_bytes: int = 0


def _fmt_val(ft: FieldType, value: Any) -> str:
    if ft == FieldType.BYTES and isinstance(value, bytes):
        return value.hex(" ")
    return repr(value)


def dump_data(cls: Type[Message], data: bytes,
              out: Optional[Callable[[str], None]] = None,
              only_fields: Optional[Iterable[int]] = None,
              ) -> DumpResult:
    """
    诊断 dump: 逐字段打印 tag / wire_type / raw hex / 值。

    新增:
      - only_fields: 只看这些 field_number
      - 损坏定位: 标出首个错误 offset, 以及错误前后能解析的字段
      - 返回 DumpResult, 便于程序化使用
    """
    if out is None:
        def _default_out(s: str) -> None:
            print(s)
        out = _default_out

    only_set = (set(int(x) for x in only_fields)
                if only_fields is not None else None)

    result = DumpResult(total_bytes=len(data))

    header = f"=== tinybuf dump: {cls.__name__} ({len(data)} bytes)"
    if only_set:
        header += f", 仅显示字段 {sorted(only_set)}"
    header += " ==="
    out(header)
    out(f"  raw hex: {data.hex(' ')}")
    out("")

    stream = BytesIO(data)
    total = len(data)
    idx_visible = 0
    idx_total = 0
    while stream.tell() < total:
        start = stream.tell()
        error_here = False

        # 读 tag
        try:
            tag_raw = decode_varint(
                stream, f"tag@{start}", allow_64bit_spill=True,
            )
        except DecodeError as exc:
            out(f"  ❗ ERROR at offset={start:<6}: 无法读取 tag — {exc}")
            result.error_at_offset = start
            result.error_message = f"tag decode error: {exc}"
            error_here = True
            break
        except Exception as exc:
            out(f"  ❗ ERROR at offset={start:<6}: {exc}")
            result.error_at_offset = start
            result.error_message = str(exc)
            error_here = True
            break

        field_number, wire_type = parse_tag(tag_raw)
        tag_end = stream.tell()
        tag_bytes = data[start:tag_end]

        visible = (only_set is None) or (field_number in only_set)

        # 先 "探测" 整个 field 的长度, 尝试拿到 raw
        mark = stream.tell()
        try:
            raw_bytes = _peek_field_raw(stream, start, wire_type, f"dump@{start}")
        except DecodeError as exc:
            # 即使 peek 失败, 也把已经读到的字节截取出来展示
            raw_bytes = data[start:stream.tell()]
            if visible:
                out(f"  [{idx_visible:03d}] offset={start:<6} "
                    f"field={field_number:<4} "
                    f"wire={wire_type}({_WIRE_NAME.get(wire_type, '?'):<7}) "
                    f"❌ ERROR: {exc}")
                out(f"           tag={tag_bytes.hex(' '):<20} "
                    f"raw (部分)={raw_bytes.hex(' ')}")
                idx_visible += 1
            result.fields.append({
                "offset": start, "field_number": field_number,
                "wire_type": wire_type, "error": str(exc),
            })
            result.error_at_offset = start
            result.error_message = str(exc)
            error_here = True
            idx_total += 1
            break
        except Exception as exc:
            raw_bytes = data[start:stream.tell()]
            out(f"  ❗ ERROR at offset={start:<6}: {exc}")
            result.error_at_offset = start
            result.error_message = str(exc)
            error_here = True
            break

        desc = cls._field_descriptors.get(field_number)

        # 解析值
        value_repr = "<unknown>"
        try:
            v_stream = BytesIO(raw_bytes)
            decode_varint(v_stream, "dump.tag", allow_64bit_spill=True)
            if desc is None:
                value_repr = "<unknown (skipped)>"
            else:
                if desc.is_map:
                    entry = _decode_single_value(desc, v_stream, f"dump.{desc.name}")
                    value_repr = f"entry(key={entry.key!r}, value={entry.value!r})"
                elif desc.repeated and desc.packable() and wire_type == WIRE_LEN:
                    length = decode_varint(
                        v_stream, f"dump.{desc.name}.packed.length",
                    )
                    payload = _assert_read(
                        v_stream, length, "dump.packed.payload",
                    )
                    values = _decode_packed_values(desc, payload, f"dump.{desc.name}")
                    value_repr = f"packed[{', '.join(repr(v) for v in values)}]"
                else:
                    value = _decode_single_value(desc, v_stream, f"dump.{desc.name}")
                    value_repr = _fmt_val(desc.field_type, value)
        except DecodeError as exc:
            value_repr = f"<decode error: {exc}>"
        except Exception as exc:
            value_repr = f"<error: {exc}>"

        entry = {
            "offset": start,
            "field_number": field_number,
            "wire_type": wire_type,
            "tag_bytes": bytes(tag_bytes),
            "raw_bytes": bytes(raw_bytes),
            "desc": desc,
            "value_repr": value_repr,
            "value_error": "<decode error" in value_repr,
        }
        result.fields.append(entry)

        if visible:
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
                elif desc.oneof:
                    extra = f" oneof({desc.oneof})"
                if desc.required:
                    extra += " required"
                name_str = f"{desc.name}({desc.field_type.name}{extra})"

            out(
                f"  [{idx_visible:03d}] offset={start:<6} "
                f"field={field_number:<4} "
                f"wire={wire_type}({_WIRE_NAME.get(wire_type, '?'):<7}) "
                f"name={name_str}"
            )
            out(
                f"           tag={tag_bytes.hex(' '):<20} "
                f"raw={raw_bytes.hex(' ')}"
            )
            out(f"           value={value_repr}")
            out("")
            idx_visible += 1

        idx_total += 1

    # 结尾: 如果中途出错, 打印概览
    lines_addendum: List[str] = []
    if result.error_at_offset is not None:
        lines_addendum.append(
            f"*** 首个损坏位置: offset={result.error_at_offset}, "
            f"原因: {result.error_message}"
        )
        ok_numbers = [f["field_number"] for f in result.fields
                      if not f.get("error") and not f.get("value_error")]
        lines_addendum.append(
            f"  损坏之前成功解析的字段号: {ok_numbers or '(无)'}"
        )
        bad = stream.tell()
        if bad < total:
            residue = data[bad:]
            # 尝试从下一个合法 tag 恢复 (仅诊断)
            restore_offset: Optional[int] = None
            for probe in range(bad + 1, total):
                test = BytesIO(data[probe:])
                try:
                    t = decode_varint(test, f"probe@{probe}",
                                      allow_64bit_spill=True)
                    fn, wt = parse_tag(t)
                    if 1 <= fn <= 1_000_000 and wt in _WIRE_NAME:
                        # 尝试 peek 这个 field 看看能不能走通
                        test2 = BytesIO(data[probe:])
                        try:
                            _peek_field_raw(test2, probe, wt, f"restore@{probe}")
                            restore_offset = probe
                            break
                        except Exception:
                            pass
                except Exception:
                    continue
            lines_addendum.append(
                f"  损坏之后剩余字节: {len(residue)} bytes ({residue.hex(' ')})"
            )
            if restore_offset is not None:
                lines_addendum.append(
                    f"  可能能恢复解析的下一个 offset: {restore_offset}"
                )
    lines_addendum.append(
        f"=== end: 共扫描 {idx_total} 字段 (显示 {idx_visible}), "
        f"结果 = {'正常' if not error_here else '有损坏'} ==="
    )
    for line in lines_addendum:
        out(line)

    return result


def _peek_field_raw(stream: BytesIO, start: int, wire_type: int,
                    ctx: str) -> bytes:
    """Peek 整个 field (tag + value) 的原始字节。"""
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
    return stream.getvalue()[start:end]


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

    # ==================================================================
    # [1/8] Schema 兼容检查工具
    # ==================================================================
    print("=" * 70)
    print("[1/8] Schema 版本兼容性检查")
    print("=" * 70)

    # ---- 旧版本 Person V1 ----
    @msg_schema
    class AddressV1(Message):
        city = Field(1, FieldType.STRING)

    @msg_schema
    class PersonV1(Message):
        id = Field(1, FieldType.UINT32, required=True)
        name = Field(2, FieldType.STRING, required=True)
        old_score = Field(3, FieldType.SINT32)
        address = Field(4, FieldType.MESSAGE, message_cls=AddressV1)
        tags = Field(5, FieldType.STRING, repeated=True)

    # ---- 新版本 Person V2 (典型演进: 删 field、加 field、改号、改 required) ----
    @msg_schema
    class AddressV2(Message):
        city = Field(1, FieldType.STRING)
        zip_code = Field(2, FieldType.STRING)  # 新增

    @msg_schema
    class PersonV2(Message):
        id = Field(1, FieldType.UINT32, required=True)
        name = Field(2, FieldType.STRING)            # required → optional
        score = Field(9, FieldType.SINT32)           # old_score 从 #3 改成新增 #9
        address = Field(4, FieldType.MESSAGE, message_cls=AddressV2)
        tags = Field(6, FieldType.STRING, repeated=True)   # ⚠️ tags 同名字段从 #5 改成 #6 → 改号!
        big_id = Field(8, FieldType.SINT64)          # 新增
        attributes = Field(10, map=(FieldType.STRING, FieldType.INT32))  # 新增

    report = check_compatibility(PersonV1, PersonV2)
    print(format_compat_report(report))
    print()

    check("有 BREAKING 级别的改动",
          report.overall == CompatLevel.BREAKING)
    diff_changes = {d.change for d in report.diffs}
    check("检测到 number_moved", "number_moved" in diff_changes)
    check("检测到 required_changed", "required_changed" in diff_changes)
    check("检测到 added (big_id / attributes / zip_code)",
          sum(1 for d in report.diffs if d.change == "added") >= 2)

    # 再做一个 "仅新增字段" 的完全安全场景
    @msg_schema
    class FooV1(Message):
        x = Field(1, FieldType.INT32)

    @msg_schema
    class FooV2(Message):
        x = Field(1, FieldType.INT32)
        y = Field(2, FieldType.STRING)
        z = Field(3, FieldType.INT64, repeated=True, packed=True)

    safe_report = check_compatibility(FooV1, FooV2)
    check("仅新增字段 = SAFE_EXTENSION",
          safe_report.overall == CompatLevel.SAFE_EXTENSION,
          f"got {safe_report.overall.name}")

    # ==================================================================
    # [2/8] Oneof 互斥字段
    # ==================================================================
    print()
    print("=" * 70)
    print("[2/8] Oneof 互斥字段")
    print("=" * 70)

    @msg_schema
    class Result(Message):
        status = Field(1, FieldType.UINT32)
        error_msg = Field(10, FieldType.STRING, oneof="payload")
        ok_body = Field(11, FieldType.BYTES, oneof="payload")
        count = Field(12, FieldType.SINT64, oneof="payload")

    print("  Result schema:")
    print("\n".join("    " + l for l in Result.format_schema().split("\n")))

    # 2a) 正常: 只设置一个
    r_ok = Result(status=200, ok_body=b"hello")
    check("which_oneof('payload') → 'ok_body'",
          r_ok.which_oneof("payload") == "ok_body")
    check("has_field('ok_body') = True", r_ok.has_field("ok_body"))
    check("has_field('error_msg') = False", not r_ok.has_field("error_msg"))
    r_ok_bytes = encode(r_ok)
    r_ok_back = decode(Result, r_ok_bytes)
    check("encode/decode 单值 round-trip",
          r_ok_back.status == 200 and r_ok_back.ok_body == b"hello"
          and r_ok_back.which_oneof("payload") == "ok_body")

    # 2b) 同时设置多个 → encode 时抛 OneofConflictError
    r_bad = Result(status=500)
    r_bad.error_msg = "oops"
    r_bad.set_field_present("error_msg", True)
    r_bad.count = 999
    r_bad.set_field_present("count", True)
    try:
        encode(r_bad)
        check("oneof 冲突未抛异常", False, "应抛 OneofConflictError")
    except OneofConflictError as exc:
        check("oneof 冲突 → OneofConflictError", True, str(exc))

    # 2c) 构造时只给一个也能 encode (默认值不算 presence)
    r_empty = Result(status=200)  # oneof 组里都没值
    empty_bytes = encode(r_empty)
    check("oneof 全空也能编码 (没 presence)",
          encode(Result(status=200)) == encode_varint(make_tag(1, WIRE_VARINT)) + encode_varint(200))
    r_empty_back = decode(Result, empty_bytes)
    check("which_oneof 空组返回 None",
          r_empty_back.which_oneof("payload") is None)

    # 2d) 解码老数据里同时出现多个 oneof → 保留最后一个 ("最后写入者胜")
    mixed = bytearray()
    # 先 error_msg
    mixed.extend(encode(Result(status=1, error_msg="first")))
    # 手工再拼上 count=42
    mixed.extend(encode_varint(make_tag(12, WIRE_VARINT)))
    mixed.extend(encode_varint(zigzag_encode(42, 64)))
    r_mixed = decode(Result, bytes(mixed))
    check("oneof 同时出现保留最后值 (count=42)",
          r_mixed.count == 42
          and r_mixed.which_oneof("payload") == "count",
          f"count={r_mixed.count}, which={r_mixed.which_oneof('payload')}")
    check("oneof 最后赢, 前一个 error_msg 没 presence",
          not r_mixed.has_field("error_msg"))

    # ==================================================================
    # [3/8] Required + Has Field (区分没出现 vs 默认值)
    # ==================================================================
    print()
    print("=" * 70)
    print("[3/8] Required 校验 & has_field presence 追踪")
    print("=" * 70)

    @msg_schema
    class LoginReq(Message):
        username = Field(1, FieldType.STRING, required=True)
        password = Field(2, FieldType.STRING, required=True)
        remember = Field(3, FieldType.BOOL)           # optional
        session_ttl = Field(4, FieldType.UINT32)     # 可选, 默认 0

    # 3a) required 缺失 → encode 失败
    try:
        encode(LoginReq())
        check("required 缺失未抛异常", False, "应抛 RequiredFieldError")
    except RequiredFieldError as exc:
        check("required 缺失 → RequiredFieldError", True, str(exc))

    # 3b) 正常填充, 通过
    req = LoginReq(username="admin", password="secret", remember=False, session_ttl=0)
    req_bytes = encode(req)
    check("has_field username/pw = True",
          req.has_field("username") and req.has_field("password"))
    check("has_field remember (False) = True (显式 set 的 False 也要记录)",
          req.has_field("remember"))
    check("has_field session_ttl (显式 0) = True",
          req.has_field("session_ttl"))
    req_back = decode(LoginReq, req_bytes)
    check("解码后 presence 保留",
          req_back.has_field("username")
          and req_back.has_field("password")
          and req_back.has_field("remember")
          and req_back.has_field("session_ttl"))

    # 3c) 区分 "字段没出现" vs "字段出现但等于默认值"
    @msg_schema
    class Simple(Message):
        a = Field(1, FieldType.INT32)   # 可选, 默认 0
        b = Field(2, FieldType.STRING)  # 可选, 默认 ""

    only_a = Simple(a=0)       # 显式 a=0
    only_a_bytes = encode(only_a)
    back_only_a = decode(Simple, only_a_bytes)
    check("显式 0 → has_field(a)=True", back_only_a.has_field("a"))
    check("字段 b 没出现 → has_field(b)=False", not back_only_a.has_field("b"))
    check("b 的值仍是默认空串", back_only_a.b == "")

    # 3d) clear_field
    s = Simple(a=5, b="hello")
    s.clear_field("a")
    check("clear_field → 值为默认 0", s.a == 0)
    check("clear_field → presence 消失", not s.has_field("a"))

    # ==================================================================
    # [4/8] dump_data 升级: 损坏定位 + only_fields 过滤
    # ==================================================================
    print()
    print("=" * 70)
    print("[4/8] dump_data 升级: 损坏定位 & 过滤")
    print("=" * 70)

    @msg_schema
    class FullPerson(Message):
        id = Field(1, FieldType.UINT32)
        name = Field(2, FieldType.STRING)
        score = Field(3, FieldType.SINT32)
        scores = Field(9, FieldType.INT32, repeated=True, packed=True)
        attributes = Field(10, map=(FieldType.STRING, FieldType.INT32))
        str_map = Field(11, map=(FieldType.STRING, FieldType.STRING))

    fp = FullPerson(
        id=999, name="Charlie", score=5,
        scores=[1, 2, 3],
        attributes={"a": 1},
        str_map={"x": "y"},
    )
    good_bytes = encode(fp)

    # 4a) 只看字段 1 和 10
    print("  --- dump, only_fields={1, 10}: ---")
    captured: List[str] = []
    dump_data(FullPerson, good_bytes,
              out=lambda s: captured.append(s),
              only_fields={1, 10})
    filt_out = "\n".join(captured)
    check("only_fields 过滤: 出现 id", "field=1" in filt_out)
    check("only_fields 过滤: 出现 attributes", "field=10" in filt_out)
    check("only_fields 过滤: name(2) 不出现", "field=2" not in filt_out)

    # 4b) 损坏数据: 在末尾加 1 个 MSB=1 的 varint 字节 (半字节截断)
    damaged = good_bytes + b"\x80"  # 最后一个 varint 只有 MSB=1, 没有后续 — 明确截断

    print()
    print("  --- dump 损坏数据 (含定位): ---")
    captured2: List[str] = []
    result = dump_data(FullPerson, bytes(damaged), out=lambda s: captured2.append(s))
    dmg_out = "\n".join(captured2)
    print("\n".join("    " + l for l in captured2))
    check("dump 损坏数据: 检测到 error_at_offset",
          result.error_at_offset is not None)
    check("dump 损坏数据: 有错误信息",
          result.error_message is not None)
    check("损坏输出含 ERROR", "ERROR" in dmg_out)

    # ==================================================================
    # [5/8] 旧功能回归 — 整数边界 + packed/varint + map + 旧 schema
    # ==================================================================
    print()
    print("=" * 70)
    print("[5/8] 回归: 整数边界 & packed & map & 向前兼容")
    print("=" * 70)

    edge_cases = {
        "min_int32": -2**31,
        "max_int32": 2**31 - 1,
        "neg_one": -1,
    }
    p = FullPerson(
        attributes=edge_cases,
        scores=[-1, -2**31, 2**31 - 1, 0],
    )
    p_bytes = encode(p)
    p_back = decode(FullPerson, p_bytes)
    check("map 边界值 min_int32", p_back.attributes["min_int32"] == -2**31)
    check("map 边界值 max_int32", p_back.attributes["max_int32"] == 2**31 - 1)
    check("packed int32 边界值",
          p_back.scores == [-1, -2**31, 2**31 - 1, 0])

    # 老版本不认识 map 和 scores.packed → 跳过
    @msg_schema
    class OldPerson(Message):
        id = Field(1, FieldType.UINT32)
        name = Field(2, FieldType.STRING)

    old_decoded = decode(OldPerson, encode(fp))
    check("老版本读 id", old_decoded.id == 999)
    check("老版本读 name", old_decoded.name == "Charlie")

    # ==================================================================
    # [6/8] 回归: packed 半字节 & 10 字节越界
    # ==================================================================
    print()
    print("=" * 70)
    print("[6/8] 回归: packed 半字节 & 10 字节越界")
    print("=" * 70)

    @msg_schema
    class PackedI32(Message):
        values = Field(1, FieldType.INT32, repeated=True, packed=True)

    bad_packed_body = bytearray()
    bad_packed_body.extend(encode_varint(1))
    bad_packed_body.append(0x80)  # 半个 varint
    bad_data = (encode_varint(make_tag(1, WIRE_LEN))
                + encode_varint(len(bad_packed_body))
                + bytes(bad_packed_body))
    try:
        decode(PackedI32, bad_data)
        ok = False
    except TruncatedDataError:
        ok = True
    check("packed 半字节截断报错", ok)

    # 10 字节合法 / 非法
    v1 = decode_varint(BytesIO(b"\xff" * 9 + b"\x01"), "legal")
    check("10 字节合法 varint (all ones)", v1 == 0xFFFFFFFFFFFFFFFF)
    v2 = decode_varint(BytesIO(b"\x80" * 9 + b"\x01"), "legal63")
    check("10 字节合法 varint (bit63 only)", v2 == 2**63)
    try:
        decode_varint(BytesIO(b"\xff" * 9 + b"\x03"), "illegal")
        ok = False
    except VarintOverflowError:
        ok = True
    check("10 字节 varint 高 2 位非法报错", ok)
    try:
        decode_varint(BytesIO(b"\x80" * 11 + b"\x00"), "toolong")
        ok = False
    except VarintOverflowError:
        ok = True
    check(">10 字节 varint 报错", ok)

    # ==================================================================
    # [7/8] 回归: 基础字段 & format_schema 含 oneof
    # ==================================================================
    print()
    print("=" * 70)
    print("[7/8] 回归: format_schema 含 oneof / map / required")
    print("=" * 70)

    @msg_schema
    class FullFeatured(Message):
        uuid = Field(1, FieldType.UINT64, required=True)
        data = Field(2, FieldType.STRING)
        flag_a = Field(3, FieldType.BOOL, oneof="which")
        flag_b = Field(4, FieldType.INT32, oneof="which")
        attrs = Field(5, map=(FieldType.STRING, FieldType.BOOL))
        nums = Field(6, FieldType.FIXED32, repeated=True, packed=True)

    schema_text = FullFeatured.format_schema()
    print("\n".join("    " + l for l in schema_text.split("\n")))
    check("format_schema 含 'required'", "required" in schema_text)
    check("format_schema 含 'oneof which'", "oneof which" in schema_text)
    check("format_schema 含 'map<STRING, BOOL>'", "map<STRING, BOOL>" in schema_text)
    check("format_schema 含 'repeated packed FIXED32'",
          "repeated packed FIXED32" in schema_text)

    # ==================================================================
    # [8/8] 端到端: 带 required + oneof + map + packed 完整 round-trip
    # ==================================================================
    print()
    print("=" * 70)
    print("[8/8] 端到端: 全特性混合 round-trip")
    print("=" * 70)

    full = FullFeatured(
        uuid=0xDEADBEEF,
        data="ok",
        flag_b=42,  # oneof 里只选这个
        attrs={"foo": True, "bar": False},
        nums=[0xFFFFFFFF, 0, 1],
    )
    check("encode 前 validate 通过", True)  # 没抛异常就是通过
    full_bytes = encode(full)
    full_back = decode(FullFeatured, full_bytes)
    check("uuid 还原", full_back.uuid == 0xDEADBEEF)
    check("data 还原", full_back.data == "ok")
    check("which_oneof → flag_b", full_back.which_oneof("which") == "flag_b")
    check("oneof 值 flag_b", full_back.flag_b == 42)
    check("map attrs 还原", full_back.attrs == {"foo": True, "bar": False})
    check("packed fixed32 还原", full_back.nums == [0xFFFFFFFF, 0, 1])
    check("required + presence", full_back.has_field("uuid"))

    print()
    print("=" * 70)
    print(f"  总计: {passed} 通过, {failed} 失败")
    print("=" * 70)
    if failed:
        raise SystemExit(1)
    print("🎉 全部通过!")
