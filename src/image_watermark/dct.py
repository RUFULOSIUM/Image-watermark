"""Bit packing and the DCT domain watermarking primitives.

One bit is stored in an 8x8 luminance DCT block by enforcing an ordering
between two selected frequency coefficients: ``coef_a`` is made larger
than ``coef_b`` for a 1, smaller for a 0. Only the luminance channel is
touched, so the colour of the image is preserved exactly.

Everything in this module is a pure function of its arguments. Which
coefficients, how strong and how large a block is comes from
:class:`~image_watermark.settings.WatermarkSettings`.
"""

from __future__ import annotations

import cv2
import numpy as np

from image_watermark.errors import ConfigError, ImageError


# ============================================================
# BITS <-> BYTES
# ============================================================

def bytes_to_bits(data: bytes) -> list[int]:
    """Expand bytes into a list of 0/1 ints, most significant bit first."""
    bits: list[int] = []

    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)

    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    """Pack 0/1 ints back into bytes, dropping an incomplete tail."""
    out = bytearray()

    for i in range(0, len(bits) - 7, 8):

        value = 0

        for bit in bits[i:i + 8]:
            value = (value << 1) | bit

        out.append(value)

    return bytes(out)


# ============================================================
# SINGLE BLOCK
# ============================================================

def embed_bit(
    block: np.ndarray,
    bit: int,
    coef_a: tuple[int, int],
    coef_b: tuple[int, int],
    strength: float,
) -> np.ndarray:
    """Force one bit into an 8x8 luminance block."""

    dct = cv2.dct(block.astype(np.float32))

    a = dct[coef_a]
    b = dct[coef_b]

    if bit == 1:

        if a <= b:
            dct[coef_a] = b + strength
            dct[coef_b] = b

        else:
            dct[coef_a] = a
            dct[coef_b] = a - strength

    else:

        if b <= a:
            dct[coef_b] = a + strength
            dct[coef_a] = a

        else:
            dct[coef_b] = b
            dct[coef_a] = b - strength

    return np.clip(cv2.idct(dct), 0, 255)


def extract_bit(
    block: np.ndarray,
    coef_a: tuple[int, int],
    coef_b: tuple[int, int],
) -> int:
    """Read one bit back out of an 8x8 luminance block."""
    dct = cv2.dct(block.astype(np.float32))

    return 1 if dct[coef_a] > dct[coef_b] else 0


# ============================================================
# TILE
# ============================================================

def _iter_blocks(
    height: int, width: int, block_size: int
):
    """Yield the top-left corner of every block, row major."""

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            yield y, x


def block_positions(
    height: int, width: int, block_size: int
) -> list[tuple[int, int]]:
    return list(_iter_blocks(height, width, block_size))


def count_blocks(
    height: int, width: int, block_size: int
) -> int:
    """How many blocks a ``height`` x ``width`` region holds."""

    if height < block_size or width < block_size:
        return 0

    rows = height // block_size
    cols = width // block_size

    return rows * cols


def embed_tile(
    tile: np.ndarray,
    bits: list[int],
    coef_a: tuple[int, int],
    coef_b: tuple[int, int],
    strength: float,
    block_size: int,
) -> np.ndarray:
    """Write ``bits`` into ``tile``, one bit per block.

    Blocks that run out of bits are left untouched, so a short payload
    simply uses less of the tile.
    """

    ycrcb = cv2.cvtColor(tile, cv2.COLOR_RGB2YCrCb)
    luminance = ycrcb[:, :, 0].astype(np.float32)

    height, width = luminance.shape

    index = 0
    total = len(bits)

    for y, x in _iter_blocks(height, width, block_size):

        if index >= total:
            break

        luminance[
            y:y + block_size, x:x + block_size
        ] = embed_bit(
            luminance[y:y + block_size, x:x + block_size],
            bits[index],
            coef_a,
            coef_b,
            strength,
        )

        index += 1

    ycrcb[:, :, 0] = luminance

    return cv2.cvtColor(
        np.clip(ycrcb, 0, 255).astype(np.uint8),
        cv2.COLOR_YCrCb2RGB,
    )


def extract_tile(
    tile: np.ndarray,
    bit_count: int,
    coef_a: tuple[int, int],
    coef_b: tuple[int, int],
    block_size: int,
) -> list[int]:
    """Read ``bit_count`` bits back out of ``tile``."""

    ycrcb = cv2.cvtColor(tile, cv2.COLOR_RGB2YCrCb)
    luminance = ycrcb[:, :, 0]

    height, width = luminance.shape

    bits: list[int] = []

    for y, x in _iter_blocks(height, width, block_size):

        if len(bits) >= bit_count:
            break

        bits.append(
            extract_bit(
                luminance[y:y + block_size, x:x + block_size],
                coef_a,
                coef_b,
            )
        )

    return bits


# ============================================================
# IMAGE PLUMBING
# ============================================================

def load_image(path: str) -> np.ndarray:
    """Read an image as an RGB uint8 array."""

    from PIL import Image

    try:
        with Image.open(path) as handle:
            return np.array(handle.convert("RGB"))

    except FileNotFoundError as e:
        raise ImageError(
            f"Image not found: {path}"
        ) from e

    except Exception as e:
        raise ImageError(
            f"Could not read image {path}: {e}"
        ) from e


def save_image(
    path: str, image: np.ndarray
) -> None:
    """Write an RGB array as PNG.

    PNG is deliberate: a lossless codec keeps the embedded watermark
    readable, which JPEG would smear.
    """

    from PIL import Image

    try:
        Image.fromarray(image).save(path, format="PNG")

    except Exception as e:
        raise ImageError(
            f"Could not write image {path}: {e}"
        ) from e


def validate_coefficients(
    coef_a: tuple[int, int],
    coef_b: tuple[int, int],
    block_size: int,
) -> None:
    """Reject coefficient choices that cannot encode a bit."""

    limit = block_size

    for name, coef in (("coef_a", coef_a), ("coef_b", coef_b)):

        row, col = coef

        if not (0 <= row < limit and 0 <= col < limit):
            raise ConfigError(
                f"{name}={tuple(coef)} is outside a "
                f"{block_size}x{block_size} DCT block."
            )

    if tuple(coef_a) == tuple(coef_b):
        raise ConfigError(
            f"coef_a and coef_b must differ, both are "
            f"{tuple(coef_a)}."
        )
