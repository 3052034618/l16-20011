"""
tinybuf - 精简版二进制序列化协议编解码器 (v5)
===============================================
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

class DecodeError(Exception): pass
class TruncatedDataError(DecodeError): pass
class VarintOverflowError(DecodeError): pass
class WireTypeMismatchError(DecodeError): pass
class SchemaError(Exception): pass
class EncodeError(Exception): pass
class RequiredFieldError(EncodeError): pass
class OneofConflictError(EncodeError): pass


class CompatLevel(IntEnum):
    FULLY_COMPATIBLE = 0
    SAFE_EXTENSION = 1
    WARNING = 2
    BREAKING = 3


# ---------------------------------------------------------------------------
# FieldType
# ---------------------------------------------------------------------------

class FieldType(IntEnum):
    INT32 = 0; INT64 = 1; UINT32 = 2; UINT64 = 3
    SINT32 = 4; SINT64 = 5; BOOL = 6; STRING = 7; BYTES = 8
    MESSAGE = 9; FIXED32 = 10; FIXED64 = 11


_WIRE_MAP: Dict[FieldType, int] = {
    FieldType.INT32: WIRE_VARINT, FieldType.INT64: WIRE_VARINT,
    FieldType.UINT32: WIRE_VARINT, FieldType.UINT64: WIRE_VARINT,
    FieldType.SINT32: WIRE_VARINT, FieldType.SINT64: WIRE_VARINT,
    FieldType.BOOL: WIRE_VARINT,
    FieldType.STRING: WIRE_LEN, FieldType.BYTES: WIRE_LEN,
    FieldType.MESSAGE: WIRE_LEN,
    FieldType.FIXED32: WIRE_FIXED32, FieldType.FIXED64: WIRE_FIXED64,
}

_PACKABLE = {FieldType.INT32, FieldType.INT64, FieldType.UINT32, FieldType.UINT64,
             FieldType.SINT32, FieldType.SINT64, FieldType.BOOL,
             FieldType.FIXED32, FieldType.FIXED64}

_MAP_KEY_TYPES = {FieldType.INT32, FieldType.INT64, FieldType.UINT32,
                  FieldType.UINT64, FieldType.SINT32, FieldType.SINT64,
                  FieldType.BOOL, FieldType.STRING,
                  FieldType.FIXED32, FieldType.FIXED64}

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


def make_tag(field_number: int, wire_type: int) -> int:
    if field_number < 1:
        raise SchemaError(f"Field number must be >= 1, got {field_number}")
    return (field_number << 3) | wire_type


def parse_tag(tag: int) -> Tuple[int, int]:
    return tag >> 3, tag & 0x07


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
# _FieldSpec / Field / _OneofSpec
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
    __slots__ = ("_spec",)

    def __init__(self, number: int, field_type: Optional[FieldType] = None,
                 repeated: bool = False, packed: bool = False,
                 message_cls: Optional[Type["Message"]] = None,
                 map: Optional[Tuple[FieldType, FieldType]] = None,
                 value_message_cls: Optional[Type["Message"]] = None,
                 required: bool = False, oneof: Optional[str] = None):
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

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        cls = self.__class__
        if name in cls._field_by_name and hasattr(self, "_fields_present"):
            desc = cls._field_by_name[name]
            if not desc.is_map and not desc.repeated:
                if desc.oneof:
                    self._clear_oneof_branches(desc.oneof, keep_name=name)
                self._fields_present.add(name)
        object.__setattr__(self, name, value)

    def __init__(self, **kwargs: Any):
        self._fields_present: Set[str] = set()

        for desc in self.__class__._field_descriptors.values():
            if desc.is_map:
                value = dict(kwargs.get(desc.name, {}))
                object.__setattr__(self, desc.name, value)
                if value:
                    self._fields_present.add(desc.name)
            elif desc.repeated:
                value = list(kwargs.get(desc.name, []))
                object.__setattr__(self, desc.name, value)
                if value:
                    self._fields_present.add(desc.name)
            else:
                if desc.name in kwargs:
                    value = kwargs[desc.name]
                    self._fields_present.add(desc.name)
                    object.__setattr__(self, desc.name, value)
                    if desc.oneof:
                        self._clear_oneof_branches(desc.oneof, keep_name=desc.name)
                else:
                    object.__setattr__(self, desc.name, self._default(desc))

    def _clear_oneof_branches(self, oneof_name: str, keep_name: Optional[str] = None) -> None:
        ospec = self.__class__._oneofs.get(oneof_name)
        if not ospec:
            return
        for other_num in ospec.fields:
            other_desc = self.__class__._field_descriptors[other_num]
            if other_desc.name == keep_name:
                continue
            object.__setattr__(self, other_desc.name, self._default(other_desc))
            self._fields_present.discard(other_desc.name)

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

    def has_field(self, name: str) -> bool:
        if name not in self.__class__._field_by_name:
            raise KeyError(f"No such field: {name}")
        return name in self._fields_present

    def set_field_present(self, name: str, present: bool = True) -> None:
        if name not in self.__class__._field_by_name:
            raise KeyError(f"No such field: {name}")
        desc = self.__class__._field_by_name[name]
        if present:
            self._fields_present.add(name)
            if desc.oneof:
                self._clear_oneof_branches(desc.oneof, keep_name=name)
        else:
            self._fields_present.discard(name)
            object.__setattr__(self, name, self._default(desc))

    def clear_field(self, name: str) -> None:
        if name not in self.__class__._field_by_name:
            raise KeyError(f"No such field: {name}")
        desc = self.__class__._field_by_name[name]
        if desc.is_map:
            object.__setattr__(self, name, {})
        elif desc.repeated:
            object.__setattr__(self, name, [])
        else:
            object.__setattr__(self, name, self._default(desc))
        self._fields_present.discard(name)

    def which_oneof(self, name: str) -> Optional[str]:
        if name not in self.__class__._oneofs:
            raise KeyError(f"No such oneof: {name}")
        for field_num in self.__class__._oneofs[name].fields:
            fname = self.__class__._field_descriptors[field_num].name
            if self.has_field(fname):
                return fname
        return None

    def validate(self) -> None:
        cls = self.__class__
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
                     required: bool = False, oneof: Optional[str] = None) -> None:
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
        lines: List[str] = []
        prefix = "  " * indent
        lines.append(f"{prefix}message {cls.__name__} {{")

        ip = indent + 1
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
                lines.append(f"{'  ' * ip}oneof {oname} {{")
                for gd in sorted(group, key=lambda d: d.number):
                    lines.append(_format_field_line(gd, indent + 2))
                lines.append(f"{'  ' * ip}}}")

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

        # oneof: 重置同组其他字段为默认值 + 清 presence
        if desc.oneof:
            ospec = cls._oneofs[desc.oneof]
            for other_num in ospec.fields:
                other_desc = cls._field_descriptors[other_num]
                if other_desc.name != desc.name:
                    present.discard(other_desc.name)
                    kwargs[other_desc.name] = Message._default(other_desc)

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
    instance._fields_present = present
    for desc in cls._field_descriptors.values():
        if desc.name in kwargs:
            object.__setattr__(instance, desc.name, kwargs[desc.name])
        else:
            if desc.is_map:
                object.__setattr__(instance, desc.name, {})
            elif desc.repeated:
                object.__setattr__(instance, desc.name, [])
            else:
                object.__setattr__(instance, desc.name, Message._default(desc))
    return instance


# ---------------------------------------------------------------------------
# 兼容性检查 — 升级为路径级递归
# ---------------------------------------------------------------------------

@dataclass
class FieldDiff:
    path: str
    field_number: Optional[int]
    change: str
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


def _compare_type(
    old_ft: FieldType, new_ft: FieldType,
    old_msg_cls: Optional[Type[Message]],
    new_msg_cls: Optional[Type[Message]],
    path: str, field_number: Optional[int],
    old_name: Optional[str], new_name: Optional[str],
    acc: List[FieldDiff],
) -> None:
    if old_ft != new_ft:
        acc.append(FieldDiff(
            path=path, field_number=field_number, change="type_changed",
            old_name=old_name, new_name=new_name,
            old_type=old_ft, new_type=new_ft,
            level=CompatLevel.BREAKING,
            detail=f"{path}: 类型从 {old_ft.name} 改成 {new_ft.name} — 破坏性!",
        ))
        return
    if old_ft == FieldType.MESSAGE and old_msg_cls and new_msg_cls:
        _recurse_diff(old_msg_cls, new_msg_cls, path, acc)


def _recurse_diff(old_cls: Type[Message], new_cls: Type[Message],
                  path_prefix: str, acc: List[FieldDiff]) -> None:
    old_by_num: Dict[int, FieldDescriptor] = dict(old_cls._field_descriptors)
    new_by_num: Dict[int, FieldDescriptor] = dict(new_cls._field_descriptors)

    old_names = {d.name: n for n, d in old_by_num.items()}
    new_names = {d.name: n for n, d in new_by_num.items()}

    removed_nums = set(old_by_num.keys()) - set(new_by_num.keys())
    added_nums = set(new_by_num.keys()) - set(old_by_num.keys())

    def _p(name: str) -> str:
        return f"{path_prefix}.{name}" if path_prefix else name

    # 同名字段改号
    for oname, onum in list(old_names.items()):
        if oname in new_names and onum != new_names[oname]:
            nnum = new_names[oname]
            if onum in removed_nums and nnum in added_nums:
                removed_nums.discard(onum); added_nums.discard(nnum)
                od, nd = old_by_num[onum], new_by_num[nnum]
                p = _p(oname)
                detail = f"{p}: 字段号从 {onum} 改成 {nnum} — 线上不兼容"
                if od.field_type != nd.field_type:
                    detail += f", 类型也从 {od.field_type.name} 改成 {nd.field_type.name}"
                acc.append(FieldDiff(
                    path=p, field_number=nnum, change="number_moved",
                    old_name=oname, new_name=oname,
                    level=CompatLevel.BREAKING, detail=detail,
                ))
                _compare_type(od.field_type, nd.field_type,
                              od.message_cls, nd.message_cls,
                              p, nnum, oname, oname, acc)
                if od.is_map and nd.is_map:
                    if od.key_type != nd.key_type:
                        kp = f"{p}<key>"
                        acc.append(FieldDiff(
                            path=kp, field_number=None, change="type_changed",
                            old_type=od.key_type, new_type=nd.key_type,
                            level=CompatLevel.BREAKING,
                            detail=f"{kp}: map key 从 {od.key_type.name} 改成 {nd.key_type.name}",
                        ))
                    if od.value_type != nd.value_type:
                        vp = f"{p}<value>"
                        acc.append(FieldDiff(
                            path=vp, field_number=None, change="type_changed",
                            old_type=od.value_type, new_type=nd.value_type,
                            level=CompatLevel.BREAKING,
                            detail=f"{vp}: map value 从 {od.value_type.name} 改成 {nd.value_type.name}",
                        ))
                    elif (od.value_type == FieldType.MESSAGE
                          and nd.value_type == FieldType.MESSAGE
                          and od.value_message_cls and nd.value_message_cls):
                        _recurse_diff(od.value_message_cls, nd.value_message_cls,
                                      vp, acc)

    for onum in removed_nums:
        od = old_by_num[onum]
        p = _p(od.name)
        level = CompatLevel.BREAKING if od.required else CompatLevel.WARNING
        detail = f"{p}: 字段 #{onum} '{od.name}' ({od.field_type.name}) 删除"
        if od.required:
            detail += " — ⚠️ 该字段是 required, 删除是破坏性改动"
        acc.append(FieldDiff(
            path=p, field_number=onum, change="removed",
            old_name=od.name, old_type=od.field_type,
            level=level, detail=detail,
        ))

    for nnum in added_nums:
        nd = new_by_num[nnum]
        p = _p(nd.name)
        level = (CompatLevel.SAFE_EXTENSION
                 if not nd.required else CompatLevel.WARNING)
        detail = f"{p}: 字段 #{nnum} '{nd.name}' ({nd.field_type.name}) 新增"
        if nd.required:
            detail += " — ⚠️ 新增 required 字段对旧数据不兼容"
        acc.append(FieldDiff(
            path=p, field_number=nnum, change="added",
            new_name=nd.name, new_type=nd.field_type,
            level=level, detail=detail,
        ))
        if nd.field_type == FieldType.MESSAGE and nd.message_cls:
            _recurse_diff(Message, nd.message_cls, p, acc)
        if (nd.is_map and nd.value_type == FieldType.MESSAGE
                and nd.value_message_cls):
            _recurse_diff(Message, nd.value_message_cls, f"{p}<value>", acc)

    for num in set(old_by_num.keys()) & set(new_by_num.keys()):
        od, nd = old_by_num[num], new_by_num[num]
        p = _p(od.name)

        if od.name != nd.name:
            acc.append(FieldDiff(
                path=p, field_number=num, change="renamed",
                old_name=od.name, new_name=nd.name,
                level=CompatLevel.FULLY_COMPATIBLE,
                detail=f"{p}: 字段 #{num} 名从 '{od.name}' 改成 '{nd.name}' — 线上兼容",
            ))

        _compare_type(od.field_type, nd.field_type,
                      od.message_cls, nd.message_cls,
                      p, num, od.name, nd.name, acc)

        if od.is_map and nd.is_map:
            if od.key_type != nd.key_type:
                kp = f"{p}<key>"
                acc.append(FieldDiff(
                    path=kp, field_number=None, change="type_changed",
                    old_type=od.key_type, new_type=nd.key_type,
                    level=CompatLevel.BREAKING,
                    detail=f"{kp}: map key 从 {od.key_type.name} 改成 {nd.key_type.name}",
                ))
            if od.value_type != nd.value_type:
                vp = f"{p}<value>"
                acc.append(FieldDiff(
                    path=vp, field_number=None, change="type_changed",
                    old_type=od.value_type, new_type=nd.value_type,
                    level=CompatLevel.BREAKING,
                    detail=f"{vp}: map value 从 {od.value_type.name} 改成 {nd.value_type.name}",
                ))
            elif (od.value_type == FieldType.MESSAGE
                  and nd.value_type == FieldType.MESSAGE
                  and od.value_message_cls and nd.value_message_cls):
                _recurse_diff(od.value_message_cls, nd.value_message_cls,
                              f"{p}<value>", acc)
        elif od.is_map != nd.is_map:
            acc.append(FieldDiff(
                path=p, field_number=num, change="label_changed",
                old_name=od.name, new_name=nd.name,
                level=CompatLevel.BREAKING,
                detail=f"{p}: label 从 {'map' if od.is_map else '非 map'} 改成 {'map' if nd.is_map else '非 map'} — 破坏性!",
            ))

        if not od.is_map and not nd.is_map:
            if ((od.repeated != nd.repeated)
                    or (od.packed != nd.packed)
                    or (od.oneof != nd.oneof)):
                old_label = (
                    "repeated packed" if od.packed else
                    "repeated" if od.repeated else
                    f"oneof({od.oneof})" if od.oneof else "scalar"
                )
                new_label = (
                    "repeated packed" if nd.packed else
                    "repeated" if nd.repeated else
                    f"oneof({nd.oneof})" if nd.oneof else "scalar"
                )
                level = CompatLevel.WARNING
                detail = f"{p}: label 从 '{old_label}' 改成 '{new_label}'"
                if od.repeated != nd.repeated:
                    level = CompatLevel.BREAKING
                    detail += " ⚠️ repeated ↔ scalar 是破坏性改动"
                acc.append(FieldDiff(
                    path=p, field_number=num, change="label_changed",
                    old_name=od.name, new_name=nd.name,
                    level=level, detail=detail,
                ))

        if od.required != nd.required:
            if od.required and not nd.required:
                level = CompatLevel.SAFE_EXTENSION
                detail = f"{p}: 从 required 改成 optional — 安全 (放宽约束)"
            else:
                level = CompatLevel.BREAKING
                detail = f"{p}: 从 optional 改成 required — 旧数据没这个字段会失败"
            acc.append(FieldDiff(
                path=p, field_number=num, change="required_changed",
                old_name=od.name, new_name=nd.name,
                level=level, detail=detail,
            ))


def diff_schemas(old_cls: Type[Message], new_cls: Type[Message]) -> List[FieldDiff]:
    diffs: List[FieldDiff] = []
    _recurse_diff(old_cls, new_cls, "", diffs)
    return sorted(diffs, key=lambda d: (-d.level.value, d.path, d.change))


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
    lines: List[str] = []
    title = f"Schema 兼容性报告: {report.old_cls_name} → {report.new_cls_name}"
    lines.append(title); lines.append("=" * len(title))
    lines.append(
        f"总体评估: {_COMPAT_MARKS[report.overall]} "
        f"{_COMPAT_TITLES[report.overall]}"
    )
    lines.append("")
    if not report.diffs:
        lines.append("  (完全没有字段变化)")
        return "\n".join(lines)
    for level in (CompatLevel.BREAKING, CompatLevel.WARNING,
                  CompatLevel.SAFE_EXTENSION, CompatLevel.FULLY_COMPATIBLE):
        items = [d for d in report.diffs if d.level == level]
        if not items:
            continue
        lines.append(f"  {_COMPAT_MARKS[level]} {level.name}:")
        for d in items:
            lines.append(f"    - {d.change:<18} {d.detail}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Schema 迁移辅助
# ---------------------------------------------------------------------------

@dataclass
class MigrationIssue:
    path: str
    kind: str
    detail: str = ""


@dataclass
class MigrationReport:
    issues: List[MigrationIssue]
    decoded: Optional[Message]

    def missing_required(self) -> List[str]:
        return [i.path for i in self.issues if i.kind == "missing_required"]

    def default_filled(self) -> List[str]:
        return [i.path for i in self.issues if i.kind == "default_filled"]


def _collect_migration_issues(msg: Message, path_prefix: str,
                              issues: List[MigrationIssue]) -> None:
    cls = msg.__class__
    for desc in cls._field_descriptors.values():
        p = f"{path_prefix}.{desc.name}" if path_prefix else desc.name
        if desc.is_map:
            if not msg.has_field(desc.name):
                if desc.required:
                    issues.append(MigrationIssue(
                        path=p, kind="missing_required",
                        detail=f"{p}: required map 字段缺失",
                    ))
                else:
                    issues.append(MigrationIssue(
                        path=p, kind="default_filled",
                        detail=f"{p}: map 字段未出现, 补空 dict",
                    ))
            elif (desc.value_type == FieldType.MESSAGE
                  and desc.value_message_cls):
                for k, v in getattr(msg, desc.name).items():
                    _collect_migration_issues(v, f"{p}[{k!r}]", issues)
            continue
        if desc.repeated:
            if not msg.has_field(desc.name):
                if desc.required:
                    issues.append(MigrationIssue(
                        path=p, kind="missing_required",
                        detail=f"{p}: required repeated 字段缺失",
                    ))
                else:
                    issues.append(MigrationIssue(
                        path=p, kind="default_filled",
                        detail=f"{p}: repeated 字段未出现, 补空列表",
                    ))
            elif desc.field_type == FieldType.MESSAGE and desc.message_cls:
                for i, v in enumerate(getattr(msg, desc.name)):
                    _collect_migration_issues(v, f"{p}[{i}]", issues)
            continue
        if not msg.has_field(desc.name):
            if desc.required:
                issues.append(MigrationIssue(
                    path=p, kind="missing_required",
                    detail=f"{p}: required 字段缺失",
                ))
            else:
                issues.append(MigrationIssue(
                    path=p, kind="default_filled",
                    detail=f"{p}: 字段未出现, 补默认值 {Message._default(desc)!r}",
                ))
            if desc.field_type == FieldType.MESSAGE and desc.message_cls:
                _collect_migration_issues(getattr(msg, desc.name), p, issues)
        elif desc.field_type == FieldType.MESSAGE and desc.message_cls:
            _collect_migration_issues(getattr(msg, desc.name), p, issues)


def try_migrate(old_bytes: bytes, new_cls: Type[T]) -> MigrationReport:
    try:
        decoded = decode(new_cls, old_bytes)
    except Exception as exc:
        issues = [MigrationIssue(
            path="<decode>", kind="missing_required",
            detail=f"解码失败: {exc}",
        )]
        return MigrationReport(issues=issues, decoded=None)
    issues: List[MigrationIssue] = []
    _collect_migration_issues(decoded, "", issues)
    return MigrationReport(issues=issues, decoded=decoded)


def format_migration_report(report: MigrationReport) -> str:
    lines: List[str] = []
    missing = [i for i in report.issues if i.kind == "missing_required"]
    defaulted = [i for i in report.issues if i.kind == "default_filled"]
    lines.append("Schema 迁移报告"); lines.append("================")
    lines.append(f"  ⚠️ 缺失 required 字段: {len(missing)}")
    for i in missing:
        lines.append(f"     - {i.detail}")
    lines.append(f"  📝 被补默认值的字段:    {len(defaulted)}")
    for i in defaulted:
        lines.append(f"     - {i.detail}")
    if report.decoded is None:
        lines.append("  ❌ 解码失败, 无法生成消息实例")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# dump_data 再升级: 名字/路径过滤、嵌套缩进、损坏恢复继续解析
# ---------------------------------------------------------------------------

@dataclass
class DumpResult:
    fields: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    total_bytes: int = 0


def _fmt_val(ft: FieldType, value: Any) -> str:
    if ft == FieldType.BYTES and isinstance(value, bytes):
        return value.hex(" ")
    return repr(value)


def _filter_accepts(field_number: int, field_name: str, path_prefix: str,
                    only_fields: Optional[Set[int]],
                    only_names: Optional[Set[str]],
                    only_paths: Optional[Set[str]]) -> bool:
    full_path = f"{path_prefix}.{field_name}" if path_prefix else field_name
    if only_fields is not None and field_number not in only_fields:
        return False
    if only_names is not None and field_name not in only_names:
        return False
    if only_paths is not None:
        match = any(full_path == op or full_path.startswith(op + ".")
                    for op in only_paths)
        if not match:
            return False
    return True


def _peek_field_raw(stream: BytesIO, start: int, wire_type: int,
                    ctx: str) -> bytes:
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


def _try_find_next_tag(data: bytes, start_probe: int, end: int,
                       known_field_numbers: Optional[Set[int]] = None) -> Optional[int]:
    """从 start_probe 开始逐字节寻找下一个能完整解码 field 的 offset。
    如果提供 known_field_numbers，优先返回那些字段号匹配的候选。"""
    first_valid: Optional[int] = None
    for probe in range(start_probe, end):
        test = BytesIO(data[probe:])
        try:
            t = decode_varint(test, f"probe@{probe}", allow_64bit_spill=True)
        except Exception:
            continue
        if t == 0:
            continue
        fn, wt = parse_tag(t)
        if fn < 1 or fn > 1_000_000 or wt not in _WIRE_NAME:
            continue
        test2 = BytesIO(data[probe:])
        try:
            _peek_field_raw(test2, probe, wt, f"restore@{probe}")
        except Exception:
            continue
        if known_field_numbers is not None and fn in known_field_numbers:
            return probe
        if first_valid is None:
            first_valid = probe
    return first_valid


def _dump_one_level(cls: Type[Message], data: bytes,
                    out: Callable[[str], None],
                    path_prefix: str, indent_level: int,
                    only_fields: Optional[Set[int]],
                    only_names: Optional[Set[str]],
                    only_paths: Optional[Set[str]],
                    result: DumpResult,
                    expand_nested: bool = True,
                    ) -> None:
    ind = "  " * indent_level
    stream = BytesIO(data)
    total = len(data)
    idx_visible = 0
    idx_total = 0

    if indent_level == 0:
        header = f"=== tinybuf dump: {cls.__name__} ({len(data)} bytes)"
        parts = []
        if only_fields is not None:
            parts.append(f"字段号={sorted(only_fields)}")
        if only_names is not None:
            parts.append(f"字段名={sorted(only_names)}")
        if only_paths is not None:
            parts.append(f"路径={sorted(only_paths)}")
        if parts:
            header += f" (过滤: {', '.join(parts)})"
        header += " ==="
        out(header)
        out(f"  raw hex: {data.hex(' ')}")
        out("")

    next_after_error_offset: Optional[int] = None
    resumed = False

    while stream.tell() < total:
        start = stream.tell()
        current_error: Optional[str] = None

        try:
            tag_raw = decode_varint(
                stream, f"tag@{start}", allow_64bit_spill=True,
            )
        except DecodeError as exc:
            current_error = f"无法读取 tag — {exc}"
        except Exception as exc:
            current_error = str(exc)

        if current_error:
            err_info = {
                "offset": start,
                "error": current_error,
                "resumed": False,
            }
            result.errors.append(err_info)
            out(f"{ind}  ❗ ERROR at offset={start:<6}: {current_error}")
            # 尝试恢复
            next_ok = _try_find_next_tag(
                data, start + 1, total,
                known_field_numbers=set(cls._field_descriptors.keys()),
            )
            if next_ok is not None:
                out(f"{ind}     ↻ 尝试恢复解析, 下一个合法 field 在 offset={next_ok}")
                err_info["resumed"] = True
                err_info["resume_offset"] = next_ok
                stream.seek(next_ok)
                resumed = True
                continue
            break

        field_number, wire_type = parse_tag(tag_raw)
        tag_end = stream.tell()
        tag_bytes = data[start:tag_end]

        # peek 整个 field 长度
        try:
            raw_bytes = _peek_field_raw(stream, start, wire_type,
                                         f"dump@{start}")
        except DecodeError as exc:
            raw_bytes = data[start:stream.tell()]
            err_info = {
                "offset": start,
                "field_number": field_number,
                "wire_type": wire_type,
                "error": str(exc),
            }
            result.errors.append(err_info)
            desc = cls._field_descriptors.get(field_number)
            fn = desc.name if desc else "?"
            if _filter_accepts(field_number, fn, path_prefix,
                               only_fields, only_names, only_paths):
                out(
                    f"{ind}  [{idx_visible:03d}] offset={start:<6} "
                    f"field={field_number:<4} "
                    f"wire={wire_type}({_WIRE_NAME.get(wire_type, '?'):<7}) "
                    f"❌ ERROR: {exc}"
                )
                out(f"{ind}           tag={tag_bytes.hex(' '):<20} "
                    f"raw (部分)={raw_bytes.hex(' ')}")
                idx_visible += 1
            probe_start = tag_end
            if probe_start <= start:
                probe_start = start + 1
            next_ok = _try_find_next_tag(
                data, probe_start, total,
                known_field_numbers=set(cls._field_descriptors.keys()),
            )
            if next_ok is not None:
                out(f"{ind}     ↻ 尝试恢复, 下一合法 field offset={next_ok}")
                stream.seek(next_ok)
                continue
            break

        desc = cls._field_descriptors.get(field_number)
        field_name = desc.name if desc else "?"

        visible = _filter_accepts(field_number, field_name, path_prefix,
                                  only_fields, only_names, only_paths)

        value_repr = "<unknown>"
        sub_payload_bytes: Optional[bytes] = None
        try:
            v_stream = BytesIO(raw_bytes)
            decode_varint(v_stream, "dump.tag", allow_64bit_spill=True)
            if desc is None:
                value_repr = "<unknown (skipped)>"
            elif desc.is_map:
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
                value_repr = f"packed[{', '.join(_fmt_val(desc.field_type, v) for v in values)}]"
            elif desc.repeated:
                value = _decode_single_value(desc, v_stream, f"dump.{desc.name}")
                value_repr = _fmt_val(desc.field_type, value)
            elif desc.field_type == FieldType.MESSAGE and not desc.is_map:
                length = decode_varint(v_stream, f"dump.{desc.name}.length")
                sub_payload_bytes = _assert_read(
                    v_stream, length, f"dump.{desc.name}.payload",
                )
                value_repr = f"<message {desc.message_cls.__name__}, {length} bytes>"
            else:
                value = _decode_single_value(desc, v_stream, f"dump.{desc.name}")
                value_repr = _fmt_val(desc.field_type, value)
        except DecodeError as exc:
            value_repr = f"❌ decode error: {exc}"

        idx_total += 1

        is_nested_message = (expand_nested and sub_payload_bytes is not None
                             and desc is not None and desc.message_cls is not None)
        sub_path = (f"{path_prefix}.{field_name}" if path_prefix
                    else field_name)

        if visible:
            field_info = {
                "index": idx_visible,
                "offset": start,
                "field_number": field_number,
                "field_name": field_name,
                "wire_type": wire_type,
                "wire_name": _WIRE_NAME.get(wire_type, "?"),
                "raw_hex": raw_bytes.hex(" "),
                "value_repr": value_repr,
                "path": sub_path,
            }
            result.fields.append(field_info)
            out(
                f"{ind}  [{idx_visible:03d}] offset={start:<6} "
                f"field={field_number:<4} ({field_name:<20}) "
                f"wire={wire_type}({_WIRE_NAME.get(wire_type, '?'):<7}) "
                f"= {value_repr}"
            )
            out(f"{ind}           tag={tag_bytes.hex(' '):<20} "
                f"raw={raw_bytes.hex(' ')}")
            idx_visible += 1

            if is_nested_message:
                out(f"{ind}     └─ nested {desc.message_cls.__name__} "
                    f"({len(sub_payload_bytes)} bytes):")
                _dump_one_level(
                    desc.message_cls, sub_payload_bytes, out,
                    sub_path, indent_level + 1,
                    only_fields, only_names, only_paths,
                    result, expand_nested=True,
                )
        elif is_nested_message and (only_names is not None or only_paths is not None):
            _dump_one_level(
                desc.message_cls, sub_payload_bytes, out,
                sub_path, indent_level + 1,
                only_fields, only_names, only_paths,
                result, expand_nested=True,
            )

    if indent_level == 0:
        out("")
        out(f"解析完成: 共 {idx_total} 个 field, "
            f"可见 {idx_visible} 个, "
            f"错误 {len(result.errors)} 个")


def dump_data(cls: Type[Message], data: bytes,
              out: Optional[Callable[[str], None]] = None,
              only_fields: Optional[Iterable[int]] = None,
              only_names: Optional[Iterable[str]] = None,
              only_paths: Optional[Iterable[str]] = None,
              expand_nested: bool = True,
              ) -> DumpResult:
    """按 schema 诊断 dump 一段二进制数据。

    Args:
        cls: 目标消息类
        data: 二进制数据
        out: 可选输出函数，不传则打印到 stdout
        only_fields: 只看这些字段号
        only_names: 只看这些字段名
        only_paths: 只看这些路径 (如 "address.zip_code", "attrs")
        expand_nested: 是否缩进展开嵌套消息
    """
    if out is None:
        import sys
        def _print(s):
            print(s)
        out = _print
    ofs = set(only_fields) if only_fields is not None else None
    ons = set(only_names) if only_names is not None else None
    ops = set(only_paths) if only_paths is not None else None
    result = DumpResult(total_bytes=len(data))
    _dump_one_level(cls, data, out, "", 0, ofs, ons, ops, result,
                    expand_nested=expand_nested)
    return result


# ---------------------------------------------------------------------------
# __main__ 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0

    def check(cond, msg):
        global passed, failed
        if cond:
            passed += 1
            print(f"  ✅ {msg}")
        else:
            failed += 1
            print(f"  ❌ {msg}")

    def section(title):
        print(f"\n=== {title} ===")

    # ============================================================
    # v1: 基础编码/解码
    # ============================================================
    section("v1: 基础 varint / zigzag / tag")

    check(encode_varint(0) == b"\x00", "varint(0)")
    check(encode_varint(1) == b"\x01", "varint(1)")
    check(encode_varint(150) == b"\x96\x01", "varint(150)")
    check(decode_varint(BytesIO(b"\x96\x01")) == 150, "decode varint(150)")

    check(zigzag_encode(0, 32) == 0, "zigzag(0)")
    check(zigzag_encode(-1, 32) == 1, "zigzag(-1)")
    check(zigzag_encode(1, 32) == 2, "zigzag(1)")
    check(zigzag_encode(-2, 32) == 3, "zigzag(-2)")
    check(zigzag_decode(1, 32) == -1, "de-zigzag(1) = -1")
    check(zigzag_decode(2, 32) == 1, "de-zigzag(2) = 1")

    check(make_tag(1, 0) == 8, "tag(1, VARINT) = 8")
    check(parse_tag(8) == (1, 0), "parse(8) = (1, VARINT)")

    section("v1: INT32 负数 round-trip + 边界值")
    for v in [-1, -2147483648, 2147483647, 0, 1, -2, 42]:
        enc = encode_varint(_to_unsigned(v, 32))
        dec = _to_signed(decode_varint(BytesIO(enc)), 32)
        check(dec == v, f"INT32 round-trip {v}")

    # ============================================================
    # v2/v3: Schema 定义、嵌套、packed、map
    # ============================================================
    section("v2: Schema 类定义 + 基础消息")

    @msg_schema
    class Phone(Message):
        number = Field(1, FieldType.STRING)
        type = Field(2, FieldType.INT32)

    @msg_schema
    class Person(Message):
        name = Field(1, FieldType.STRING)
        age = Field(2, FieldType.INT32)
        phones = Field(3, FieldType.MESSAGE, message_cls=Phone, repeated=True)
        scores = Field(4, FieldType.INT32, repeated=True, packed=True)

    p = Person(name="Alice", age=30,
               phones=[Phone(number="123", type=1), Phone(number="456", type=2)],
               scores=[10, 20, -5, 0])
    data = encode(p)
    p2 = decode(Person, data)
    check(p2.name == "Alice", "decode name")
    check(p2.age == 30, "decode age")
    check(len(p2.phones) == 2, "decode phones len")
    check(p2.phones[0].number == "123", "decode phone[0].number")
    check(p2.phones[1].number == "456", "decode phone[1].number")
    check(p2.scores == [10, 20, -5, 0], f"decode packed scores = {p2.scores}")
    check(p == p2, "Person encode/decode equal")

    section("v2: packed + 非 packed 两种写法都能读回")
    p3 = Person(scores=[1, 2, 3])
    data3 = encode(p3)
    p3d = decode(Person, data3)
    check(p3d.scores == [1, 2, 3], f"packed scores via encode: {p3d.scores}")

    section("v3: map 字段")

    @msg_schema
    class Config(Message):
        attrs = Field(1, map=(FieldType.STRING, FieldType.STRING))
        counts = Field(2, map=(FieldType.STRING, FieldType.INT32))

    c = Config(attrs={"env": "prod", "region": "cn"},
               counts={"a": 1, "b": 2})
    cdata = encode(c)
    c2 = decode(Config, cdata)
    check(c2.attrs == {"env": "prod", "region": "cn"}, f"map<string,string> = {c2.attrs}")
    check(c2.counts == {"a": 1, "b": 2}, f"map<string,int32> = {c2.counts}")

    section("v3: varint 溢出 / packed 截断 报错")
    try:
        bad = b"\x80\x80\x80\x80\x80\x80\x80\x80\x80\x02"
        decode_varint(BytesIO(bad), "test")
        check(False, "10-byte overflow varint should fail")
    except VarintOverflowError:
        check(True, "10-byte overflow varint raises VarintOverflowError")

    try:
        partial_packed = encode_varint(make_tag(4, WIRE_LEN)) + encode_varint(3) + b"\x80"
        decode(Person, partial_packed)
        check(False, "packed with truncated varint should fail")
    except TruncatedDataError:
        check(True, "packed truncated varint raises TruncatedDataError")

    section("v3: schema introspection / IDL")
    idl = Config.format_schema()
    check("message Config" in idl, "IDL contains 'message Config'")
    check("map<STRING, STRING> attrs = 1" in idl, "IDL contains attrs map")
    check("map<STRING, INT32> counts = 2" in idl, "IDL contains counts map")
    print("    IDL:\n" + "\n".join(f"      {l}" for l in idl.splitlines()))

    # ============================================================
    # v4: oneof / required / default / compat / dump basic
    # ============================================================
    section("v4: oneof 互斥组")

    @msg_schema
    class Event(Message):
        id = Field(1, FieldType.INT64, required=True)

        @msg_schema
        class Created(Message):
            user = Field(1, FieldType.STRING)

        @msg_schema
        class Updated(Message):
            field = Field(1, FieldType.STRING)
            value = Field(2, FieldType.STRING)

        created = Field(2, FieldType.MESSAGE, message_cls=Created, oneof="payload")
        updated = Field(3, FieldType.MESSAGE, message_cls=Updated, oneof="payload")

    ev = Event(id=1, created=Event.Created(user="alice"))
    check(ev.which_oneof("payload") == "created", "which_oneof -> created")
    ev_data = encode(ev)
    ev2 = decode(Event, ev_data)
    check(ev2.which_oneof("payload") == "created", "decode keeps which_oneof")

    try:
        bad = Event(id=1)
        bad._fields_present.add("created")
        bad._fields_present.add("updated")
        object.__setattr__(bad, "created", Event.Created(user="x"))
        object.__setattr__(bad, "updated", Event.Updated(field="y", value="z"))
        encode(bad)
        check(False, "oneof both set should fail")
    except OneofConflictError:
        check(True, "oneof both set raises OneofConflictError")

    section("v4: required 字段校验")
    try:
        encode(Event())  # no id
        check(False, "missing required should fail")
    except RequiredFieldError:
        check(True, "missing required raises RequiredFieldError")

    section("v4: presence 区分 默认值 vs 未出现")
    ev_empty = decode(Event, encode(Event(id=42)))
    check(not ev_empty.has_field("created"), "has_field(created) = False when absent")
    check(ev_empty.has_field("id"), "has_field(id) = True")
    check(ev_empty.created.user == "", "absent oneof branch reads default empty Message")

    section("v4: 兼容性检查 (基础)")

    @msg_schema
    class PersonV1(Message):
        name = Field(1, FieldType.STRING, required=True)
        age = Field(2, FieldType.INT32)
        email = Field(3, FieldType.STRING)
        phones = Field(4, FieldType.STRING, repeated=True)

    @msg_schema
    class PersonV2(Message):
        name = Field(1, FieldType.STRING)
        age = Field(2, FieldType.INT32)
        tags = Field(6, FieldType.STRING, repeated=True)
        address = Field(7, FieldType.STRING)
        phones = Field(5, FieldType.STRING, repeated=True)

    report = check_compatibility(PersonV1, PersonV2)
    print(f"    overall = {report.overall.name}")
    for d in report.diffs:
        print(f"    - {d.change} {d.detail}")
    check(report.overall == CompatLevel.BREAKING,
          "PersonV1->V2 overall = BREAKING (tags moved + email removed required)")
    changes = {d.change for d in report.diffs}
    check("number_moved" in changes, "detect tags number_moved (renumbered)")
    check("removed" in changes, "detect removed email")
    check("added" in changes, "detect added address")

    section("v4: dump_data 基础 + 字段号过滤 + 损坏数据")

    dump_result = dump_data(Person, data)
    check(len(dump_result.fields) > 0, "dump produces fields")
    check(len(dump_result.errors) == 0, "dump no errors on good data")

    dump_filtered = dump_data(Person, data, only_fields={1, 2})
    check(all(f["field_number"] in (1, 2) for f in dump_filtered.fields),
          f"only_fields filter works: {[f['field_number'] for f in dump_filtered.fields]}")

    # 构造损坏数据: 在末尾追加单独的 \x80 (截断 varint)
    bad_data = data + b"\x80"
    dump_bad = dump_data(Person, bad_data)
    check(len(dump_bad.errors) > 0, f"dump detects error: {dump_bad.errors}")
    print(f"    errors = {dump_bad.errors}")

    # ============================================================
    # v5 新功能测试
    # ============================================================

    # ---- 1. 路径级兼容性报告: 嵌套消息 + map ----
    section("v5: 兼容性报告 — 嵌套消息 + map 路径级")

    @msg_schema
    class AddressV1(Message):
        street = Field(1, FieldType.STRING)
        city = Field(2, FieldType.STRING)

    @msg_schema
    class UserV1(Message):
        name = Field(1, FieldType.STRING)
        address = Field(2, FieldType.MESSAGE, message_cls=AddressV1)
        attrs = Field(3, map=(FieldType.STRING, FieldType.INT32))

    @msg_schema
    class AddressV2(Message):
        street = Field(1, FieldType.STRING)
        city = Field(2, FieldType.STRING)
        zip_code = Field(3, FieldType.STRING)

    @msg_schema
    class UserV2(Message):
        name = Field(1, FieldType.STRING)
        address = Field(2, FieldType.MESSAGE, message_cls=AddressV2)
        attrs = Field(3, map=(FieldType.STRING, FieldType.STRING))

    r5 = check_compatibility(UserV1, UserV2)
    print(f"    overall = {r5.overall.name}")
    for d in r5.diffs:
        print(f"    - {d.level.name} {d.change}  {d.detail}")
    paths = {d.path for d in r5.diffs}
    check("address.zip_code" in paths, "检测到嵌套路径 address.zip_code 新增")
    check("attrs<value>" in paths, "检测到 map value 类型变化 attrs<value>")
    check(r5.overall == CompatLevel.BREAKING,
          "map value 类型变化为 BREAKING")
    print("  " + format_compat_report(r5).replace("\n", "\n  "))

    # ---- 2. oneof 解码重置分支为默认值 + 重新赋值 + 清空 ----
    section("v5: oneof 语义 — 解码重置默认值 / 重新赋值 / 清空编码")

    # 构造"老数据"：同时编码 created 和 updated (模拟老版本无序编码)
    ev_multi = Event(id=99)
    ev_multi.created = Event.Created(user="first")
    created_payload = encode(Event.Created(user="first"), validate=False)
    raw_created = (encode_varint(make_tag(2, WIRE_LEN))
                   + encode_varint(len(created_payload))
                   + created_payload)
    updated_payload = encode(Event.Updated(field="status", value="ok"), validate=False)
    raw_updated = (encode_varint(make_tag(3, WIRE_LEN))
                   + encode_varint(len(updated_payload))
                   + updated_payload)
    raw_id = encode_varint(make_tag(1, WIRE_VARINT)) + encode_varint(99)
    old_bytes = raw_id + raw_created + raw_updated  # updated 后出现，应获胜

    ev_decoded = decode(Event, old_bytes)
    check(ev_decoded.which_oneof("payload") == "updated",
          f"解码后 which_oneof = updated (最后写入者胜)")
    check(not ev_decoded.has_field("created"),
          "解码后 created has_field = False")
    # 核心: created 对象里的值应该被重置为默认空值
    check(ev_decoded.created.user == "",
          f"解码后 created.user 被重置为默认空串, 实际 = {ev_decoded.created.user!r}")
    check(ev_decoded.updated.field == "status",
          f"解码后 updated.field = status")

    # 重新赋值: 切到另一个分支
    ev_decoded.created = Event.Created(user="new_user")
    check(ev_decoded.which_oneof("payload") == "created",
          "重新赋值后 which_oneof -> created")
    check(not ev_decoded.has_field("updated"),
          "重新赋值后 updated has_field = False")
    check(ev_decoded.updated.field == "",
          f"重新赋值后 updated.field 重置为空, 实际 = {ev_decoded.updated.field!r}")
    reenc = encode(ev_decoded)
    redec = decode(Event, reenc)
    check(redec.which_oneof("payload") == "created",
          "重新编码后 which_oneof = created")
    check(redec.created.user == "new_user",
          f"重新编码后 created.user = new_user")

    # 清空分支: set_field_present False
    ev_decoded.set_field_present("created", False)
    check(ev_decoded.which_oneof("payload") is None,
          "清空 created 后 which_oneof = None")
    check(ev_decoded.created.user == "",
          f"清空后 created.user 重置为空")
    check(not ev_decoded.has_field("created"),
          "清空后 has_field(created) = False")
    empty_enc = encode(ev_decoded)
    empty_dec = decode(Event, empty_enc)
    check(empty_dec.which_oneof("payload") is None,
          "清空后编码再解码: which_oneof = None")
    check(not empty_dec.has_field("created"), "清空后编码: created absent")
    check(not empty_dec.has_field("updated"), "清空后编码: updated absent")

    # clear_field API
    ev3 = Event(id=1, created=Event.Created(user="x"))
    ev3.clear_field("created")
    check(not ev3.has_field("created"), "clear_field: has_field False")
    check(ev3.which_oneof("payload") is None, "clear_field: which_oneof None")

    # ---- 3. Schema 迁移辅助 ----
    section("v5: Schema 迁移辅助 — 缺失 required + 默认值填充")

    @msg_schema
    class OldUser(Message):
        name = Field(1, FieldType.STRING)

    @msg_schema
    class NewUser(Message):
        name = Field(1, FieldType.STRING)
        age = Field(2, FieldType.INT32, required=True)
        nickname = Field(3, FieldType.STRING)

        @msg_schema
        class Addr(Message):
            city = Field(1, FieldType.STRING, required=True)
            zip_code = Field(2, FieldType.STRING)

        address = Field(4, FieldType.MESSAGE, message_cls=Addr)
        tags = Field(5, FieldType.STRING, repeated=True)

    old_u = OldUser(name="Alice")
    old_u_bytes = encode(old_u)

    mig = try_migrate(old_u_bytes, NewUser)
    print("  " + format_migration_report(mig).replace("\n", "\n  "))
    check(mig.decoded is not None, "迁移解码成功")
    missing = set(mig.missing_required())
    check("age" in missing, f"检测到缺失 required age: missing={missing}")
    check("address.city" in missing,
          f"检测到嵌套缺失 required address.city: missing={missing}")
    defaults = set(mig.default_filled())
    check("nickname" in defaults, f"检测到默认值填充 nickname: defaults={defaults}")
    check("address" in defaults, "检测到默认值填充 address (嵌套消息)")
    check("tags" in defaults, "检测到默认值填充 tags (repeated)")

    # ---- 4. dump_data: 按字段名/路径过滤 + 嵌套缩进 + 损坏恢复后续可读 ----
    section("v5: dump_data — 字段名/路径过滤 + 嵌套缩进 + 损坏恢复")

    # 构造嵌套复杂数据
    @msg_schema
    class FullAddr(Message):
        street = Field(1, FieldType.STRING)
        city = Field(2, FieldType.STRING)
        zip = Field(3, FieldType.STRING)

    @msg_schema
    class FullPerson(Message):
        name = Field(1, FieldType.STRING)
        age = Field(2, FieldType.INT32)
        addr = Field(3, FieldType.MESSAGE, message_cls=FullAddr)
        score = Field(4, FieldType.INT32)

    fp = FullPerson(name="Bob", age=25,
                    addr=FullAddr(street="Main St", city="NYC", zip="10001"),
                    score=99)
    fp_data = encode(fp)

    print("    --- 按字段名过滤 (只看 name, city) ---")
    lines_buf = []
    dump_data(FullPerson, fp_data, out=lines_buf.append,
              only_names={"name", "city"})
    for l in lines_buf:
        print(f"    {l}")
    visible_names = {f["field_name"] for f in lines_buf
                     if isinstance(f, dict) and "field_name" in f}
    # 更简单: 检查 lines_buf 里输出了 name 和 city
    name_found = any("name" in l and "Bob" in l for l in lines_buf)
    city_found = any("city" in l and "NYC" in l for l in lines_buf)
    check(name_found and city_found,
          f"按字段名过滤: name={name_found}, city={city_found}")

    print("\n    --- 按路径过滤 (只看 addr.zip) ---")
    lines_buf2 = []
    dump_data(FullPerson, fp_data, out=lines_buf2.append,
              only_paths={"addr", "addr.zip"})
    for l in lines_buf2:
        print(f"    {l}")
    zip_found = any("zip" in l and "10001" in l for l in lines_buf2)
    check(zip_found, f"按路径过滤 addr.zip 可见: {zip_found}")

    # 嵌套缩进展示
    print("\n    --- 嵌套消息缩进展示 ---")
    lines_buf3 = []
    dump_data(FullPerson, fp_data, out=lines_buf3.append, expand_nested=True)
    indent_lines = [l for l in lines_buf3 if l.startswith("      ") or "      [" in l]
    has_nested = any("nested FullAddr" in l for l in lines_buf3)
    check(has_nested, f"检测到 nested FullAddr 标题行")

    # 损坏恢复: 中间插损坏字节，后续字段仍能解析
    # 构造: name(1) + [LEN tag + 超大长度 (不完整)] + age(2) + addr(3) + score(4)
    name_bytes = encode_varint(make_tag(1, WIRE_LEN)) + encode_varint(3) + b"Bob"
    age_bytes = encode_varint(make_tag(2, WIRE_VARINT)) + encode_varint(25)
    score_bytes = encode_varint(make_tag(4, WIRE_VARINT)) + encode_varint(99)
    # LEN field with tag(99, LEN) + length 99999 — way more than remaining buffer
    corrupt_bytes = encode_varint(make_tag(99, WIRE_LEN)) + encode_varint(999999)
    corrupt_middle = name_bytes + corrupt_bytes + age_bytes + score_bytes

    print("\n    --- 损坏数据恢复解析 ---")
    lines_buf4 = []
    res_corrupt = dump_data(FullPerson, corrupt_middle, out=lines_buf4.append)
    for l in lines_buf4:
        print(f"    {l}")
    check(len(res_corrupt.errors) >= 1, f"检测到错误: {len(res_corrupt.errors)}")
    # name 应该在损坏前被解析
    name_parsed = any("name" in l and "Bob" in l for l in lines_buf4)
    # age 和 score 应该在损坏恢复后被解析
    age_parsed = any("age" in l and "25" in l for l in lines_buf4)
    score_parsed = any("score" in l and "99" in l for l in lines_buf4)
    check(name_parsed, f"损坏前 name 被解析: {name_parsed}")
    check(age_parsed, f"恢复后 age 被解析: {age_parsed}")
    check(score_parsed, f"恢复后 score 被解析: {score_parsed}")

    # ============================================================
    # 汇总
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"总计: 通过 {passed}, 失败 {failed}")
    if failed > 0:
        sys.exit(1)
    sys.exit(0)
