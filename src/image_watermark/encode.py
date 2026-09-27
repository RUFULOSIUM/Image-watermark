"""Procedural wrapper around :class:`WatermarkEncoder`.

Kept for backwards compatibility with v0.1 and v0.2. Every function
keeps its old signature, its old module constants and its old error
messages, and it still terminates with ``SystemExit`` on failure.

The important detail: this module still writes the **v0.2 byte format**
(``WM01`` container, magic inside the body). That is what existing
callers, existing images and the existing test suite expect. New code
should use :class:`~image_watermark.encoder.WatermarkEncoder` with a
:class:`~image_watermark.schema.PayloadSchema`, which produces the
self-describing ``WM02`` container.
"""

from __future__ import annotations

import lzma
import os
import struct
import sys

from typing import NoReturn

import numpy as np
from reedsolo import RSCodec

from image_watermark import dct
from image_watermark.container import (
    LEGACY_HEADER_SIZE,
    MAGIC_LEGACY,
)
from image_watermark.container import (
    SIGNATURE_SIZE as CONTAINER_SIGNATURE_SIZE,
)
from image_watermark.errors import (
    CapacityError,
    KeyFileError,
    PayloadError,
    WatermarkError,
)
from image_watermark.keys import (
    DEFAULT_PRIVATE_KEY,
    DEFAULT_PUBLIC_KEY,
    generate_key_pair,
)
from image_watermark.keys import load_private_key as _read_private_key
from image_watermark.legacy import (
    LEGACY_MAX_PROMPT,
    LEGACY_MAX_SHORT,
    build_legacy_body,
)
from image_watermark.rs import (
    RS_PARITY,
    max_message_length,
    rs_len,
)
from image_watermark.settings import WatermarkSettings


# ============================================================
# CONSTANTS
#
# The v0.2 values, derived from the default settings so there is a
# single source of truth.
# ============================================================

_DEFAULT = WatermarkSettings()

TILE_SIZE = _DEFAULT.tile_size
BLOCK_SIZE = _DEFAULT.block_size

COEF_A = _DEFAULT.coef_a
COEF_B = _DEFAULT.coef_b
STRENGTH = _DEFAULT.strength

MAGIC = MAGIC_LEGACY

SIGNATURE_SIZE = CONTAINER_SIGNATURE_SIZE

#: Legacy ``WM01``: magic + 2 byte body length
HEADER_SIZE = LEGACY_HEADER_SIZE

#: A 512x512 tile holds 64x64 DCT blocks, one bit each.
TILE_BLOCKS = _DEFAULT.block_count
TILE_BYTES = _DEFAULT.capacity

#: Largest container that survives one Reed-Solomon round trip.
MAX_PAYLOAD_SIZE = max_message_length(TILE_BYTES)

#: Largest compressed body that still leaves room for the framing.
MAX_COMPRESSED_LENGTH = (
    MAX_PAYLOAD_SIZE - HEADER_SIZE - SIGNATURE_SIZE
)

PRIVATE_KEY = DEFAULT_PRIVATE_KEY
PUBLIC_KEY = DEFAULT_PUBLIC_KEY

MAX_MODEL_LENGTH = LEGACY_MAX_SHORT
MAX_VERSION_LENGTH = LEGACY_MAX_SHORT
MAX_PROMPT_LENGTH = LEGACY_MAX_PROMPT


# ============================================================
# CLEAN EXIT
# ============================================================

def error(message) -> NoReturn:
    sys.exit(f"ERROR: {message}")


def _guard(action, *args, **kwargs):
    """Run ``action``, turning library errors into SystemExit."""

    try:
        return action(*args, **kwargs)

    except CapacityError as e:
        error(str(e))

    except PayloadError as e:
        error(str(e))

    except KeyFileError as e:
        error(str(e))

    except WatermarkError as e:
        error(str(e))

    except ValueError as e:
        error(str(e))


# ============================================================
# KEYS
# ============================================================

def create_keys(
    private_path: str = PRIVATE_KEY,
    public_path: str = PUBLIC_KEY,
) -> None:
    """Create an Ed25519 key pair in the working directory."""

    try:
        generate_key_pair(private_path, public_path)

    except OSError as e:
        error(f"Could not write key files: {e}")

    print("Created private_key.pem")
    print("Created public_key.pem")


def load_private_key(path: str = PRIVATE_KEY):
    """Load the Ed25519 private key, exiting on failure."""

    return _guard(_read_private_key, path)


# ============================================================
# PAYLOAD
# ============================================================

def create_compressed_payload(
    prompt: str,
    model: str,
    version: str,
) -> bytes:
    """Build a signed v0.2 ``WM01`` container."""

    body = _guard(
        build_legacy_body,
        model=model,
        version=version,
        prompt=prompt,
    )

    compressed_body = lzma.compress(
        body, lzma.FORMAT_XZ, preset=9
    )

    if len(compressed_body) > MAX_COMPRESSED_LENGTH:
        error(
            f"Compressed payload is too large: "
            f"{len(compressed_body)} > {MAX_COMPRESSED_LENGTH} bytes. "
            f"Shorten the prompt or use a weaker compression preset."
        )

    # Signiert wird der unkomprimierte Body
    signature = _read_private_key(PRIVATE_KEY).sign(body)

    return (
        MAGIC
        + struct.pack(">H", len(compressed_body))
        + compressed_body
        + signature
    )


# ============================================================
# REED SOLOMON
# ============================================================

def reed_solomon_encode(data: bytes) -> bytes:
    """Reed-Solomon encode with the default parity."""

    codec = RSCodec(RS_PARITY)

    try:
        encoded = bytes(codec.encode(data))

    except Exception as e:
        error(f"Reed-Solomon encoding failed: {e}")

    if len(encoded) != rs_len(len(data)):
        error(
            f"Reed-Solomon produced {len(encoded)} bytes, expected "
            f"{rs_len(len(data))}. The reedsolo version does not "
            f"match this build."
        )

    return encoded


# ============================================================
# BIT HELPERS
# ============================================================

def bytes_to_bits(data: bytes) -> list[int]:
    return dct.bytes_to_bits(data)


def embed_bit(block: np.ndarray, bit: int) -> np.ndarray:
    return dct.embed_bit(block, bit, COEF_A, COEF_B, STRENGTH)


def embed_tile(tile: np.ndarray, bits: list[int]) -> np.ndarray:
    return dct.embed_tile(
        tile, bits, COEF_A, COEF_B, STRENGTH, BLOCK_SIZE
    )


# ============================================================
# ENCODE IMAGE
# ============================================================

def encode_image(
    input_path: str,
    output_path: str,
    prompt: str,
    model: str,
    version: str,
) -> None:
    """Watermark an image with the v0.2 format and print a report."""

    if not os.path.exists(input_path):
        error(f"Input image not found: {input_path}")

    if not os.path.exists(PRIVATE_KEY):
        print("No keys found.")
        print("Generating Ed25519 key pair...")
        create_keys()
        print()

    print("[1/5] Loading image...")
    image = _guard(dct.load_image, input_path)

    height, width = image.shape[:2]

    if width < TILE_SIZE or height < TILE_SIZE:
        error(
            f"Image must be at least {TILE_SIZE}x{TILE_SIZE} pixels."
        )

    print(f"      Image: {width}x{height}")
    print(f"      Tile capacity: {TILE_BYTES} bytes (codeword)")

    print("[2/5] Creating payload...")
    payload = create_compressed_payload(prompt, model, version)

    payload_length = len(payload)

    print(f"      Payload: {payload_length} bytes")

    if rs_len(payload_length) > TILE_BYTES:
        error(
            f"Encoded payload is too large: "
            f"{rs_len(payload_length)} > {TILE_BYTES} bytes. "
            f"Shorten the prompt."
        )

    print("[3/5] Reed-Solomon encoding...")
    encoded = reed_solomon_encode(payload)
    bits = bytes_to_bits(encoded)

    print(f"      Encoded: {len(encoded)} bytes")
    print(f"      Bits: {len(bits)}")

    print("[4/5] Embedding watermark...")
    output = image.copy()

    tiles_x = width // TILE_SIZE
    tiles_y = height // TILE_SIZE

    for ty in range(tiles_y):

        for tx in range(tiles_x):

            x = tx * TILE_SIZE
            y = ty * TILE_SIZE

            output[y:y + TILE_SIZE, x:x + TILE_SIZE] = embed_tile(
                output[y:y + TILE_SIZE, x:x + TILE_SIZE],
                bits,
            )

    print("[5/5] Saving...")
    _guard(dct.save_image, output_path, output)

    print("================================")
    print(" WATERMARK CREATED")
    print("================================")
    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Tiles:        {tiles_x * tiles_y}")
    print("================================")
