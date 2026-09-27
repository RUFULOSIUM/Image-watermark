"""Payload schemas: the declarative description of a watermarking scheme.

A schema is an ordered list of fields. It fully determines how a payload
is serialised, so the encoder and the decoder never hard-code a layout.
Writing ``schemas/my_scheme.json`` is enough to define a new
watermarking scheme.

Example
-------
.. code-block:: json

    {
      "id": "my_scheme",
      "fields": [
        {"name": "artist", "type": "string"},
        {"name": "created", "type": "uint64", "default": "now"},
        {"name": "uuid", "type": "bytes", "length": 16,
         "display": "hex", "default": "uuid4"},
        {"name": "note", "type": "string", "compression": "zlib"}
      ]
    }
"""

from __future__ import annotations

import json
import os
import re
import struct

from dataclasses import dataclass, field as dc_field
from importlib.resources.abc import Traversable
from typing import Any, Callable, Iterator

from image_watermark.errors import SchemaError


# ============================================================
# FIELD TYPES
# ============================================================

#: Field types that carry a fixed byte width in the body.
FIXED_WIDTH_TYPES = {
    "uint8": (">B", 1),
    "uint16": (">H", 2),
    "uint32": (">I", 4),
    "uint64": (">Q", 8),
    "int32": (">i", 4),
    "float32": (">f", 4),
    "bool": (">B", 1),
}

#: Field types written as a length prefix followed by the raw bytes.
LENGTH_PREFIXED_TYPES = {"string", "bytes"}

FIELD_TYPES = set(FIXED_WIDTH_TYPES) | LENGTH_PREFIXED_TYPES

#: Per-field compression, applied before the body is compressed as a whole.
FIELD_COMPRESSION = {"none", "zlib", "lzma"}

#: Named generators usable as ``"default"``.
GENERATORS: dict[str, Callable[[], Any]] = {}

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _gen_uuid4() -> bytes:
    import uuid

    return uuid.uuid4().bytes


def _gen_now() -> int:
    import time

    return int(time.time())


def _gen_empty() -> bytes:
    return b""


def _gen_zero() -> int:
    return 0


GENERATORS.update(
    {
        "uuid4": _gen_uuid4,
        "now": _gen_now,
        "empty": _gen_empty,
        "zero": _gen_zero,
    }
)


# ============================================================
# FIELD
# ============================================================

@dataclass(frozen=True)
class FieldSpec:
    """One field of a payload schema.

    ``length`` is the fixed width in bytes for fixed-width types and the
    maximum accepted length for length-prefixed types. ``0`` means "no
    limit" for the latter and "use the natural width" for the former.
    """

    name: str
    type: str
    length: int = 0
    compression: str = "none"
    default: Any = None
    required: bool = True
    description: str = ""
    display: str = ""

    def __post_init__(self) -> None:

        if self.type not in FIELD_TYPES:
            raise SchemaError(
                f"Field {self.name!r}: unknown type {self.type!r}. "
                f"Known types: {', '.join(sorted(FIELD_TYPES))}"
            )

        if self.compression not in FIELD_COMPRESSION:
            raise SchemaError(
                f"Field {self.name!r}: unknown compression "
                f"{self.compression!r}. Known: "
                f"{', '.join(sorted(FIELD_COMPRESSION))}"
            )

        if self.length < 0:
            raise SchemaError(
                f"Field {self.name!r}: length must be >= 0"
            )

        if (
            self.default is not None
            and self.default not in GENERATORS
        ):
            raise SchemaError(
                f"Field {self.name!r}: default must be a value or one "
                f"of {', '.join(sorted(GENERATORS))}"
            )

    # --------------------------------------------------------
    # Derived properties
    # --------------------------------------------------------

    @property
    def prefix_bytes(self) -> int:
        """Width of the length prefix, 0 for fixed-width types."""

        if self.type in LENGTH_PREFIXED_TYPES:
            return 4

        return 0

    @property
    def min_size(self) -> int:
        """Smallest number of body bytes this field can occupy."""

        if self.type in FIXED_WIDTH_TYPES:
            return FIXED_WIDTH_TYPES[self.type][1]

        return self.prefix_bytes

    @property
    def max_size(self) -> int:
        """Largest number of body bytes this field can occupy."""

        if self.type in FIXED_WIDTH_TYPES:
            return FIXED_WIDTH_TYPES[self.type][1]

        limit = self.length or (1 << 32) - 1

        return self.prefix_bytes + limit

    def is_generated(self) -> bool:
        return (
            self.default is not None
            and self.default in GENERATORS
        )

    def empty(self) -> Any:
        """The zero value used when an optional field is absent.

        Every field occupies a fixed slot in the body, so an absent
        optional field is written as its zero value rather than
        skipped. Skipping it would shift all following fields and the
        decoder could no longer tell where it is.
        """

        if self.type == "string":
            return ""

        if self.type == "bytes":
            return b""

        if self.type == "bool":
            return False

        return 0

    # --------------------------------------------------------
    # Serialisation
    # --------------------------------------------------------

    def encode(self, value: Any) -> bytes:
        """Serialise one value into body bytes."""

        from image_watermark.payload import (
            _compress_field,
            _decompress_field,
        )

        if self.type in FIXED_WIDTH_TYPES:
            return self._encode_fixed(value)

        raw = self._encode_variable(value)

        compressed = _compress_field(raw, self.compression)

        limit = self.length

        if limit and len(compressed) > limit:
            raise SchemaError(
                f"Field {self.name!r}: encoded value is "
                f"{len(compressed)} bytes, limit is {limit}"
            )

        return (
            struct.pack(">I", len(compressed))
            + compressed
        )

    def decode(self, data: bytes, pos: int) -> tuple[Any, int]:
        """Deserialise one value starting at ``pos``.

        Returns the value and the new position.
        """

        from image_watermark.payload import _decompress_field

        if self.type in FIXED_WIDTH_TYPES:
            fmt, size = FIXED_WIDTH_TYPES[self.type]

            if pos + size > len(data):
                raise SchemaError(
                    f"Field {self.name!r}: truncated, need {size} "
                    f"bytes at offset {pos}, have "
                    f"{len(data) - pos}"
                )

            return (
                struct.unpack(fmt, data[pos:pos + size])[0],
                pos + size,
            )

        if pos + 4 > len(data):
            raise SchemaError(
                f"Field {self.name!r}: truncated length prefix"
            )

        length = struct.unpack(
            ">I", data[pos:pos + 4]
        )[0]

        pos += 4

        if pos + length > len(data):
            raise SchemaError(
                f"Field {self.name!r}: truncated value, need "
                f"{length} bytes, have {len(data) - pos}"
            )

        raw = data[pos:pos + length]
        pos += length

        return (
            self._decode_variable(
                _decompress_field(raw, self.compression)
            ),
            pos,
        )

    # --------------------------------------------------------
    # Internals
    # --------------------------------------------------------

    def _encode_fixed(self, value: Any) -> bytes:

        fmt, _size = FIXED_WIDTH_TYPES[self.type]

        try:
            if self.type == "bool":
                return struct.pack(fmt, 1 if value else 0)

            if self.type in ("float32",):
                return struct.pack(fmt, float(value))

            return struct.pack(
                fmt, int(value)
            )

        except (struct.error, TypeError, ValueError) as e:
            raise SchemaError(
                f"Field {self.name!r}: cannot encode "
                f"{value!r} as {self.type}: {e}"
            ) from e

    def _encode_variable(self, value: Any) -> bytes:

        if isinstance(value, str):
            return value.encode("utf-8")

        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)

        raise SchemaError(
            f"Field {self.name!r}: expected str or bytes for "
            f"{self.type}, got {type(value).__name__}"
        )

    def _decode_variable(self, raw: bytes) -> Any:
        """Turn stored bytes back into the declared Python type.

        ``string`` fields come back as ``str`` so a round trip is
        value-preserving; ``bytes`` fields stay raw. A string that is not
        valid utf-8 is a corrupt payload, not something to paper over
        with replacement characters.
        """

        if self.type != "string":
            return raw

        try:
            return raw.decode("utf-8")

        except UnicodeDecodeError as e:
            raise SchemaError(
                f"Field {self.name!r}: value is not valid utf-8 "
                f"({e})"
            ) from e


# ============================================================
# SCHEMA
# ============================================================

@dataclass(frozen=True)
class PayloadSchema:
    """An ordered, validated set of payload fields."""

    id: str
    fields: tuple[FieldSpec, ...] = ()
    description: str = ""
    version: int = 1
    by_name: dict[str, FieldSpec] = dc_field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:

        if not _ID_RE.match(self.id or ""):
            raise SchemaError(
                f"Invalid schema id {self.id!r}: must match "
                f"{_ID_RE.pattern} (max 64 chars, no spaces)"
            )

        if not self.fields:
            raise SchemaError(
                f"Schema {self.id!r} has no fields."
            )

        if self.version < 1:
            raise SchemaError(
                f"Schema {self.id!r}: version must be >= 1"
            )

        seen: set[str] = set()

        for spec in self.fields:

            if spec.name in seen:
                raise SchemaError(
                    f"Schema {self.id!r}: duplicate field "
                    f"{spec.name!r}"
                )

            seen.add(spec.name)

        object.__setattr__(
            self,
            "by_name",
            {f.name: f for f in self.fields},
        )

    # --------------------------------------------------------
    # Constructors
    # --------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PayloadSchema:
        """Build a schema from a parsed ``schema.json`` document."""

        if not isinstance(data, dict):
            raise SchemaError(
                f"Schema must be a JSON object, got "
                f"{type(data).__name__}"
            )

        unknown = set(data) - {
            "id",
            "name",
            "version",
            "description",
            "fields",
        }

        if unknown:
            raise SchemaError(
                f"Unknown schema key(s): "
                f"{', '.join(sorted(unknown))}. Allowed: id, name, "
                f"version, description, fields"
            )

        raw_fields = data.get("fields")

        if not isinstance(raw_fields, list) or not raw_fields:
            raise SchemaError(
                f"Schema {data.get('id')!r}: 'fields' must be a "
                f"non-empty list"
            )

        specs = []

        for index, entry in enumerate(raw_fields):

            if not isinstance(entry, dict):
                raise SchemaError(
                    f"Schema {data.get('id')!r}: field #{index} is "
                    f"not an object"
                )

            specs.append(cls._field_from_dict(entry, index))

        return cls(
            id=data.get("id") or data.get("name") or "",
            fields=tuple(specs),
            description=data.get("description") or "",
            version=int(data.get("version") or 1),
        )

    @staticmethod
    def _field_from_dict(
        entry: dict[str, Any], index: int
    ) -> FieldSpec:

        allowed = {
            "name",
            "type",
            "length",
            "compression",
            "default",
            "required",
            "description",
            "display",
        }

        unknown = set(entry) - allowed

        if unknown:
            raise SchemaError(
                f"Field #{index}: unknown key(s) "
                f"{', '.join(sorted(unknown))}. Allowed: "
                f"{', '.join(sorted(allowed))}"
            )

        name = entry.get("name")

        if not isinstance(name, str) or not name:
            raise SchemaError(
                f"Field #{index}: 'name' must be a non-empty string"
            )

        if "type" not in entry:
            raise SchemaError(
                f"Field {name!r}: missing 'type'"
            )

        try:
            return FieldSpec(
                name=name,
                type=entry["type"],
                length=int(entry.get("length") or 0),
                compression=entry.get("compression") or "none",
                default=entry.get("default"),
                required=bool(entry.get("required", True)),
                description=entry.get("description") or "",
                display=entry.get("display") or "",
            )

        except (TypeError, ValueError) as e:
            raise SchemaError(
                f"Field {name!r}: {e}"
            ) from e

    @classmethod
    def load(
        cls, source: str | os.PathLike[str] | Traversable
    ) -> PayloadSchema:
        """Read and validate a ``schema.json``.

        ``source`` may be a filesystem path or anything with a
        ``read_text`` method, which is what
        :mod:`importlib.resources` hands out. That matters because the
        bundled schema has to be readable from an installed wheel, not
        only from a source checkout.
        """

        label = getattr(source, "name", source)

        try:

            if isinstance(source, (str, os.PathLike)):
                with open(source, "r", encoding="utf-8") as handle:
                    text = handle.read()

            else:
                text = source.read_text(encoding="utf-8")

            data = json.loads(text)

        except FileNotFoundError as e:
            raise SchemaError(
                f"Schema file not found: {label}"
            ) from e

        except json.JSONDecodeError as e:
            raise SchemaError(
                f"Schema file {label} is not valid JSON: {e}"
            ) from e

        schema = cls.from_dict(data)

        return schema

    # --------------------------------------------------------
    # Rendering
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable dict, the inverse of :meth:`from_dict`.

        Field keys are only written when they differ from the default,
        so a hand-written schema stays as small as it was.
        """

        data: dict[str, Any] = {"id": self.id}

        if self.description:
            data["description"] = self.description

        if self.version != 1:
            data["version"] = self.version

        fields: list[dict[str, Any]] = []

        for spec in self.fields:
            entry: dict[str, Any] = {
                "name": spec.name,
                "type": spec.type,
            }

            if spec.type in LENGTH_PREFIXED_TYPES and spec.length:
                entry["length"] = spec.length

            if spec.compression != "none":
                entry["compression"] = spec.compression

            if spec.display != "auto":
                entry["display"] = spec.display

            if not spec.required:
                entry["required"] = False

            if spec.default is not None:
                entry["default"] = spec.default

            fields.append(entry)

        data["fields"] = fields

        return data

    def describe(self) -> str:
        """Human readable multi-line rendering of the schema."""

        lines = [f"schema {self.id}  (v{self.version})"]

        if self.description:
            lines.append(self.description)

        width = max((len(f.name) for f in self.fields), default=4)

        for spec in self.fields:

            bits = [spec.type]

            if spec.compression != "none":
                bits.append(spec.compression)

            if spec.type in LENGTH_PREFIXED_TYPES:
                bits.append(
                    f"max {spec.length}"
                    if spec.length
                    else "unbounded"
                )

            if spec.is_generated():
                bits.append(f"default {spec.default}")

            elif not spec.required:
                bits.append("optional")

            suffix = f"  ({', '.join(bits)})"

            lines.append(f"  {spec.name:<{width}}{suffix}")

        if any(
            f.type in LENGTH_PREFIXED_TYPES and not f.length
            for f in self.fields
        ):
            upper = "unbounded (container decides)"
        else:
            upper = str(self.max_body_size())

        lines.append(
            f"  body size       {self.min_body_size()} to "
            f"{upper} bytes"
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<PayloadSchema id={self.id!r} "
            f"fields={len(self.fields)}>"
        )

    # --------------------------------------------------------
    # Queries
    # --------------------------------------------------------

    def __iter__(self) -> Iterator[FieldSpec]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __contains__(self, name: object) -> bool:
        return name in self.by_name

    def __getitem__(self, name: str) -> FieldSpec:
        try:
            return self.by_name[name]

        except KeyError as e:
            raise SchemaError(
                f"Schema {self.id!r} has no field {name!r}. "
                f"Available: {', '.join(self.by_name)}"
            ) from e

    def min_body_size(self) -> int:
        """Smallest possible body size in bytes."""
        return sum(f.min_size for f in self.fields)

    def max_body_size(self) -> int:
        """Largest possible body size in bytes."""
        return sum(f.max_size for f in self.fields)

    def generate_defaults(self) -> dict[str, Any]:
        """Build a value dict from every generated field."""

        return {
            spec.name: GENERATORS[spec.default]()
            for spec in self.fields
            if spec.is_generated()
        }

    def display_value(
        self, name: str, value: Any
    ) -> str:
        """Render a decoded value for human consumption."""

        spec = self[name]

        if spec.display == "hex" and isinstance(
            value, (bytes, bytearray)
        ):
            return bytes(value).hex()

        if spec.display == "utf8" and isinstance(
            value, (bytes, bytearray)
        ):
            return bytes(value).decode(
                "utf-8", errors="replace"
            )

        if isinstance(value, (bytes, bytearray)):
            return bytes(value).hex()

        return str(value)
