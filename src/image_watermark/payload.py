"""The :class:`Payload` object and its schema-driven serialisation.

A payload is an ordered set of named values whose binary layout is
decided entirely by a :class:`~image_watermark.schema.PayloadSchema`.
The same schema class serves the encoder and the decoder, so there is no
second hard-coded copy of the layout anywhere in the project.
"""

from __future__ import annotations

import lzma
import zlib

from typing import Any, Iterator, Mapping

from image_watermark.errors import PayloadError, SchemaError
from image_watermark.schema import GENERATORS, PayloadSchema


# ============================================================
# FIELD LEVEL COMPRESSION
# ============================================================

def _compress_field(raw: bytes, algorithm: str) -> bytes:
    """Compress a single field value."""

    if algorithm == "none":
        return raw

    if algorithm == "zlib":
        return zlib.compress(raw)

    if algorithm == "lzma":
        return lzma.compress(
            raw, lzma.FORMAT_XZ, preset=6
        )

    raise SchemaError(
        f"Unknown field compression {algorithm!r}"
    )


def _decompress_field(
    raw: bytes, algorithm: str
) -> bytes:
    """Decompress a single field value."""

    if algorithm == "none":
        return raw

    try:
        if algorithm == "zlib":
            return zlib.decompress(raw)

        if algorithm == "lzma":
            return lzma.decompress(raw)

    except Exception as e:
        raise SchemaError(
            f"Field decompression failed "
            f"({algorithm}): {e}"
        ) from e

    raise SchemaError(
        f"Unknown field compression {algorithm!r}"
    )


# ============================================================
# PAYLOAD
# ============================================================

class Payload:
    """A set of watermark values bound to a schema.

    Values may be supplied at construction time, assigned afterwards, or
    omitted entirely if the schema marks them ``required: false`` or gives
    them a generator such as ``"uuid4"`` or ``"now"``.

    >>> payload = Payload(schema, prompt="hello", model="sd")
    >>> payload["image_id"]  # doctest: +SKIP
    b'\\x8f\\x2c...'
    """

    __slots__ = ("_schema", "_values", "signature", "body")

    def __init__(
        self,
        schema: PayloadSchema,
        values: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:

        if not isinstance(schema, PayloadSchema):
            raise PayloadError(
                f"schema must be a PayloadSchema, got "
                f"{type(schema).__name__}"
            )

        self._schema = schema
        self._values: dict[str, Any] = {}
        self.signature: bytes | None = None
        self.body: bytes | None = None

        merged: dict[str, Any] = dict(values or {})
        merged.update(kwargs)

        for name, value in merged.items():

            if name not in schema:
                raise PayloadError(
                    f"Schema {schema.id!r} has no field {name!r}. "
                    f"Available: {', '.join(schema.by_name)}"
                )

            self._values[name] = value

    # --------------------------------------------------------
    # Properties
    # --------------------------------------------------------

    @property
    def schema(self) -> PayloadSchema:
        return self._schema

    @property
    def values(self) -> dict[str, Any]:
        """A copy of the current values."""
        return dict(self._values)

    # --------------------------------------------------------
    # Mapping interface
    # --------------------------------------------------------

    def __getitem__(self, name: str) -> Any:
        try:
            return self._values[name]

        except KeyError as e:
            raise PayloadError(
                f"Payload has no value for {name!r}"
            ) from e

    def __setitem__(self, name: str, value: Any) -> None:

        if name not in self._schema:
            raise PayloadError(
                f"Schema {self._schema.id!r} has no field "
                f"{name!r}. Available: "
                f"{', '.join(self._schema.by_name)}"
            )

        self._values[name] = value

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(
        self, name: str, default: Any = None
    ) -> Any:
        return self._values.get(name, default)

    def items(self):
        return self._values.items()

    # --------------------------------------------------------
    # Serialisation
    # --------------------------------------------------------

    def complete(self) -> Payload:
        """Fill in generated defaults and return ``self``.

        Called automatically by :meth:`serialize`. A required field with
        no value and no generator is an error, because a payload that
        silently drops data is worse than one that refuses to build.
        """

        for spec in self._schema:

            if spec.name in self._values:
                continue

            if spec.is_generated():
                self._values[spec.name] = GENERATORS[
                    spec.default
                ]()

            elif not spec.required:
                continue

            else:
                raise PayloadError(
                    f"Missing required field {spec.name!r} for "
                    f"schema {self._schema.id!r}. Either pass a "
                    f"value or give the field a 'default' generator."
                )

        return self

    def serialize(self) -> bytes:
        """Serialise the complete payload into body bytes.

        Every field of the schema gets a slot, even when an optional
        field has no value; it is then written as its zero value. The
        layout is therefore fixed by the schema alone, and a decoder
        can find every field without guessing. Omitting a field would
        shift everything after it and silently corrupt the payload.
        """

        self.complete()

        chunks = []

        for spec in self._schema:

            if spec.name in self._values:
                value = self._values[spec.name]

            elif spec.required:
                raise PayloadError(
                    f"Missing required field {spec.name!r} for "
                    f"schema {self._schema.id!r}."
                )

            else:
                value = spec.empty()

            chunks.append(spec.encode(value))

        body = b"".join(chunks)

        self.body = body

        return body

    @classmethod
    def deserialize(
        cls,
        schema: PayloadSchema,
        body: bytes,
    ) -> Payload:
        """Rebuild a payload from body bytes using ``schema``."""

        payload = cls(schema)

        pos = 0

        for spec in schema:
            value, pos = spec.decode(body, pos)
            payload._values[spec.name] = value

        if pos != len(body):
            raise SchemaError(
                f"Schema {schema.id!r} left {len(body) - pos} "
                f"unread byte(s). The schema does not match the data."
            )

        payload.body = body

        return payload

    # --------------------------------------------------------
    # Presentation
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain dict copy, for JSON serialisation."""
        return dict(self._values)

    def format(self) -> str:
        """Human readable multi-line rendering of the payload."""

        width = max(
            (len(f.name) for f in self._schema),
            default=4,
        )

        lines = []

        for spec in self._schema:

            if spec.name not in self._values:
                continue

            value = self._schema.display_value(
                spec.name, self._values[spec.name]
            )

            if not value:
                value = "(empty)"

            lines.append(f"{spec.name:<{width}}  {value}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        filled = len(self._values)

        return (
            f"<Payload schema={self._schema.id!r} "
            f"fields={filled}/{len(self._schema)}>"
        )
