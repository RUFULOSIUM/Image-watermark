"""Command line interface for v0.3.

Five subcommands:

``init-keys``   create an Ed25519 key pair
``encode``      watermark an image with a schema
``decode``      read a watermark back
``schema``      show or validate a schema
``info``        print the effective settings and capacity

Payload fields are not hard coded. ``encode`` takes repeatable
``-f NAME=VALUE`` pairs and validates the names against the schema, so
the same command works for the bundled default scheme and for any
custom ``schema.json``.

Run ``image-watermark COMMAND --help`` for the options of a command.
Every command exits 0 on success and 1 on a handled error, printing a
single ``error: ...`` line to stderr, so the tool composes with shell
scripts and CI.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

from image_watermark import (
    PayloadSchema,
    SchemaError,
    WatermarkDecoder,
    WatermarkEncoder,
    WatermarkError,
    default_settings,
    generate_key_pair,
    load_default_schema,
)
from image_watermark.errors import ConfigError
from image_watermark.schema import FIXED_WIDTH_TYPES

EXIT_OK = 0
EXIT_ERROR = 1


# ============================================================
# SHARED OPTIONS
# ============================================================

def _add_schema_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--schema",
        metavar="FILE",
        help=(
            "schema JSON to use; omit for the schema bundled "
            "with the package"
        ),
    )


def _add_key_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--private-key",
        default=None,
        metavar="FILE",
        help="private key path (default private_key.pem)",
    )
    parser.add_argument(
        "--public-key",
        default=None,
        metavar="FILE",
        help="public key path (default public_key.pem)",
    )


def _add_geometry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        metavar="PX",
        help="tile edge length in pixels (default 512)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        metavar="FLOAT",
        help="DCT coefficient strength (default 12.0)",
    )
    parser.add_argument(
        "--parity",
        type=int,
        default=None,
        metavar="N",
        help="Reed-Solomon parity bytes per 223 byte chunk (default 32)",
    )
    parser.add_argument(
        "--compression",
        choices=("lzma", "zlib", "none"),
        default=None,
        help="container compression (default lzma)",
    )
    parser.add_argument(
        "--no-sign",
        action="store_true",
        help="do not sign the payload",
    )
    parser.add_argument(
        "--no-redundant",
        action="store_true",
        help="embed into the first tile only instead of every tile",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        metavar="N",
        help="stop after N tiles even in redundant mode",
    )


def _settings_from_args(args: argparse.Namespace):
    """Build settings from the shared options."""

    overrides: dict = {}

    for name in (
        "tile_size",
        "strength",
        "parity",
        "compression",
        "max_tiles",
    ):
        value = getattr(args, name, None)

        if value is not None:
            overrides[name] = value

    if getattr(args, "no_sign", False):
        overrides["sign"] = False

    if getattr(args, "no_redundant", False):
        overrides["redundant"] = False

    if getattr(args, "private_key", None):
        overrides["private_key_path"] = args.private_key

    if getattr(args, "public_key", None):
        overrides["public_key_path"] = args.public_key

    schema = load_default_schema(args.schema)

    return default_settings(schema=schema, **overrides)


# ============================================================
# COMMANDS
# ============================================================

def cmd_init_keys(args: argparse.Namespace) -> int:
    """Create an Ed25519 key pair."""

    private = args.private_key or "private_key.pem"
    public = args.public_key or "public_key.pem"

    generate_key_pair(private, public)

    print(f"wrote {private}")
    print(f"wrote {public}")

    return EXIT_OK


def _coerce(spec, text: str):
    """Turn a command line string into a value of the field's type.

    The CLI only ever receives text, so integers are parsed and byte
    fields accept hex when they are displayed as hex. Anything else is
    passed through as a string and validated by the schema later.
    """

    if spec.type == "bool":
        lowered = text.strip().lower()

        if lowered in ("1", "true", "yes", "on"):
            return True

        if lowered in ("0", "false", "no", "off"):
            return False

        raise ConfigError(
            f"Field {spec.name!r}: {text!r} is not a boolean"
        )

    if spec.type in FIXED_WIDTH_TYPES:
        try:
            number = int(text, 0)

        except ValueError:
            raise ConfigError(
                f"Field {spec.name!r}: {text!r} is not an integer"
            ) from None

        fmt, size = FIXED_WIDTH_TYPES[spec.type]

        try:
            struct.pack(fmt, number)

        except struct.error:
            raise ConfigError(
                f"Field {spec.name!r}: {number} does not fit in "
                f"{size} byte(s)"
            ) from None

        return number

    if spec.type == "bytes":
        if spec.display == "hex":
            try:
                return bytes.fromhex(text)

            except ValueError:
                raise ConfigError(
                    f"Field {spec.name!r}: {text!r} is not valid hex"
                ) from None

        return text.encode("utf-8")

    return text


def _collect_values(
    schema: PayloadSchema,
    pairs: list[str],
) -> dict:
    """Turn ``-f name=value`` pairs into a payload value dict."""

    values: dict = {}

    for pair in pairs:

        if "=" not in pair:
            raise ConfigError(
                f"Expected NAME=VALUE, got {pair!r}"
            )

        name, _, text = pair.partition("=")

        name = name.strip()

        if name not in schema.by_name:
            known = ", ".join(sorted(schema.by_name))

            raise ConfigError(
                f"Unknown field {name!r} for schema "
                f"{schema.id!r}. Known fields: {known}"
            )

        if name in values:
            raise ConfigError(
                f"Field {name!r} given twice"
            )

        values[name] = _coerce(schema.by_name[name], text)

    return values


def cmd_encode(args: argparse.Namespace) -> int:
    """Watermark an image."""

    settings = _settings_from_args(args)

    encoder = WatermarkEncoder(settings=settings)

    if settings.sign:
        encoder.ensure_keys()

    values = _collect_values(
        settings.require_schema(), args.field
    )

    result = encoder.encode(
        args.input,
        args.output,
        values,
        progress=(
            None if args.quiet else lambda m: print(m)
        ),
    )

    if not args.quiet:
        print()
        print(result.describe())

    return EXIT_OK


def cmd_decode(args: argparse.Namespace) -> int:
    """Read a watermark back."""

    settings = _settings_from_args(args)

    decoder = WatermarkDecoder(settings=settings)

    if args.schema:

        decoder.register(load_default_schema(args.schema))

    if args.extra_schema:
        decoder.load_schema(args.extra_schema)

    result = decoder.decode(
        args.image,
        progress=(
            None if args.quiet else lambda m: print(m)
        ),
    )

    if args.json:
        print(
            json.dumps(
                {
                    "tile": list(result.tile),
                    "version": result.version,
                    "schema_id": result.schema_id,
                    "signature_valid": result.valid,
                    "compression": result.container.compression,
                    "values": {
                        key: _jsonable(value)
                        for key, value in result.payload.items()
                    },
                    "candidates": len(result.candidates),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    else:
        print(result.describe())

    return EXIT_OK


def cmd_schema(args: argparse.Namespace) -> int:
    """Show or validate a schema."""

    schema = load_default_schema(args.file)

    if args.validate:
        print(
            f"ok: {schema.id} "
            f"({len(schema.fields)} fields)"
        )

        return EXIT_OK

    if args.format == "json":
        print(
            json.dumps(
                schema.to_dict(), indent=2, ensure_ascii=False
            )
        )

    else:
        print(schema.describe())

    return EXIT_OK


def cmd_info(args: argparse.Namespace) -> int:
    """Print the effective settings and capacity."""

    settings = _settings_from_args(args)

    print(settings.summary())
    print()
    print("capacity")
    print(f"  tile             {settings.tile_size}x{settings.tile_size}")
    print(f"  blocks           {settings.block_count}")
    print(f"  codeword         {settings.capacity} bytes")
    print(f"  max container    {settings.max_payload} bytes")
    print(f"  max raw body     {settings.max_raw_body} bytes")

    return EXIT_OK


def _jsonable(value):
    """Make a payload value survive ``json.dumps``."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()

    return value


# ============================================================
# PARSER
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-watermark",
        description=(
            "Watermark images with Reed-Solomon error "
            "correction and DCT embedding."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the version and exit",
    )

    sub = parser.add_subparsers(dest="command")

    # ----------------------------------------------------
    # init-keys
    # ----------------------------------------------------
    keys = sub.add_parser(
        "init-keys", help="create an Ed25519 key pair"
    )
    _add_key_args(keys)
    keys.set_defaults(func=cmd_init_keys)

    # ----------------------------------------------------
    # encode
    # ----------------------------------------------------
    enc = sub.add_parser("encode", help="watermark an image")
    enc.add_argument("input", help="source image")
    enc.add_argument("output", help="destination image")
    _add_schema_args(enc)
    _add_key_args(enc)
    _add_geometry_args(enc)
    enc.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress progress output",
    )
    enc.add_argument(
        "-f",
        "--field",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "set a payload field; repeatable. Names come "
            "from the schema, and fields with a default are "
            "filled in automatically"
        ),
    )
    enc.set_defaults(func=cmd_encode)

    # ----------------------------------------------------
    # decode
    # ----------------------------------------------------
    dec = sub.add_parser("decode", help="read a watermark")
    dec.add_argument("image", help="image to inspect")
    _add_schema_args(dec)
    _add_key_args(dec)
    _add_geometry_args(dec)
    dec.add_argument(
        "--extra-schema",
        action="append",
        default=[],
        metavar="FILE",
        help=(
            "register an additional schema before decoding; "
            "repeatable"
        ),
    )
    dec.add_argument(
        "--json",
        action="store_true",
        help="print the result as JSON",
    )
    dec.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress progress output",
    )
    dec.set_defaults(func=cmd_decode)

    # ----------------------------------------------------
    # schema
    # ----------------------------------------------------
    sch = sub.add_parser(
        "schema", help="show or validate a schema"
    )
    sch.add_argument(
        "file",
        nargs="?",
        default=None,
        help="schema JSON; omit for the bundled default",
    )
    sch.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default text)",
    )
    sch.add_argument(
        "--validate",
        action="store_true",
        help="only report whether the schema is valid",
    )
    sch.set_defaults(func=cmd_schema)

    # ----------------------------------------------------
    # info
    # ----------------------------------------------------
    info = sub.add_parser(
        "info", help="print settings and capacity"
    )
    _add_schema_args(info)
    _add_key_args(info)
    _add_geometry_args(info)
    info.set_defaults(func=cmd_info)

    return parser


# ============================================================
# ENTRY
# ============================================================

def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""

    parser = build_parser()

    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from image_watermark import __version__

        print(__version__)

        return EXIT_OK

    if not getattr(args, "command", None):
        parser.print_help()

        return EXIT_ERROR

    try:
        return args.func(args)

    except (WatermarkError, SchemaError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)

        return EXIT_ERROR

    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)

        return EXIT_ERROR

    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)

        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
