"""Image watermarking with Reed-Solomon error correction and DCT embedding.

The public API is object oriented:

* :class:`WatermarkSettings` holds every tunable value.
* :class:`PayloadSchema` describes the fields of a watermarking scheme and
  can be loaded from a ``schema.json`` you write yourself.
* :class:`Payload` binds values to a schema and serialises them.
* :class:`WatermarkEncoder` and :class:`WatermarkDecoder` do the work.

The ``encode`` and ``decode`` modules are kept as thin procedural
wrappers for backwards compatibility with v0.2.
"""

import importlib.resources
import os

from typing import Any

from image_watermark.container import Container, Header
from image_watermark.decoder import (
    DecodeResult,
    SchemaRegistry,
    WatermarkDecoder,
)
from image_watermark.dct import bits_to_bytes, bytes_to_bits
from image_watermark.encoder import EncodeResult, WatermarkEncoder
from image_watermark.errors import (
    CapacityError,
    ConfigError,
    ContainerError,
    DecodingError,
    ImageError,
    KeyFileError,
    PayloadError,
    SchemaError,
    WatermarkError,
)
from image_watermark.keys import (
    generate_key_pair,
    load_private_key,
    load_public_key,
)
from image_watermark.payload import Payload
from image_watermark.rs import RSGeometry
from image_watermark.schema import FieldSpec, PayloadSchema
from image_watermark.settings import WatermarkSettings

__version__ = "0.3.0"

#: Name of the directory holding the schemas shipped with the package.
SCHEMA_PACKAGE = "image_watermark.schemas"

#: The schema every install gets without passing a file.
DEFAULT_SCHEMA_FILE = "default.json"


def load_default_schema(
    path: str | os.PathLike[str] | None = None,
) -> PayloadSchema:
    """Load the ``default`` schema, or a schema from ``path``.

    With no argument the schema is read from the package data, so it
    works from a wheel and not just from a source checkout.
    """

    if path is not None:
        return PayloadSchema.load(path)

    resource = (
        importlib.resources.files(SCHEMA_PACKAGE)
        .joinpath(DEFAULT_SCHEMA_FILE)
    )

    return PayloadSchema.load(resource)


def default_settings(
    schema: PayloadSchema | None = None,
    **overrides: Any,
) -> WatermarkSettings:
    """Settings bound to the default schema, with optional overrides."""

    return WatermarkSettings(
        schema=schema or load_default_schema(),
        **overrides,
    )


def main() -> None:
    """Entry point for the ``image-watermark`` console script."""
    import sys

    from image_watermark.cli import main as cli_main

    sys.exit(cli_main())


__all__ = [
    "CapacityError",
    "ConfigError",
    "Container",
    "ContainerError",
    "DecodeResult",
    "DecodingError",
    "EncodeResult",
    "FieldSpec",
    "Header",
    "ImageError",
    "KeyFileError",
    "Payload",
    "PayloadError",
    "PayloadSchema",
    "RSGeometry",
    "SchemaError",
    "SchemaRegistry",
    "WatermarkDecoder",
    "WatermarkEncoder",
    "WatermarkError",
    "WatermarkSettings",
    "DEFAULT_SCHEMA_FILE",
    "SCHEMA_PACKAGE",
    "__version__",
    "bits_to_bytes",
    "bytes_to_bits",
    "default_settings",
    "generate_key_pair",
    "load_default_schema",
    "load_private_key",
    "load_public_key",
]
