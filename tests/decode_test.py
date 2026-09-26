import struct
import zlib
import lzma

import numpy as np
import pytest
from PIL import Image

import image_watermark.decode as decode
import image_watermark.encode as encode

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)


# ============================================================
# BITS -> BYTES
# ============================================================

def test_bits_to_bytes_known_values():
    assert decode.bits_to_bytes([1] * 8) == b"\xff"
    assert decode.bits_to_bytes([0] * 8) == b"\x00"
    assert decode.bits_to_bytes([1, 0, 1, 0, 1, 0, 1, 0]) == b"\xaa"


def test_bits_to_bytes_drops_partial_bytes():
    assert decode.bits_to_bytes([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]) == b"\x01"


def test_bits_to_bytes_roundtrip():
    data = bytes(range(256))
    assert decode.bits_to_bytes(encode.bytes_to_bits(data)) == data


# ============================================================
# EXTRACT BIT / BITS
# ============================================================

@pytest.mark.parametrize("bit", [0, 1])
def test_extract_bit_embedded(bit):
    rng = np.random.default_rng(3)
    block = rng.integers(0, 256, (8, 8), dtype=np.uint8)

    assert decode.extract_bit(encode.embed_bit(block, bit)) == bit


def test_extract_bits_roundtrip():
    rng = np.random.default_rng(11)
    tile = np.full(
        (decode.TILE_SIZE, decode.TILE_SIZE, 3),
        90,
        dtype=np.uint8,
    )
    bits = list(rng.integers(0, 2, 400).astype(int))

    embedded = encode.embed_tile(tile, bits)

    assert decode.extract_bits(embedded, len(bits)) == bits


def test_extract_bits_stops_at_bit_count():
    rng = np.random.default_rng(13)
    tile = np.full(
        (decode.TILE_SIZE, decode.TILE_SIZE, 3),
        150,
        dtype=np.uint8,
    )
    bits = list(rng.integers(0, 2, 2000).astype(int))

    embedded = encode.embed_tile(tile, bits)

    assert len(decode.extract_bits(embedded, 100)) == 100


# ============================================================
# DECODE TILE
# ============================================================

def test_decode_tile_roundtrip():
    rng = np.random.default_rng(5)
    tile = np.full(
        (decode.TILE_SIZE, decode.TILE_SIZE, 3),
        70,
        dtype=np.uint8,
    )
    data = bytes(range(64))

    embedded = encode.embed_tile(tile, encode.bytes_to_bits(data))

    assert decode.decode_tile(embedded, len(data)) == data


# ============================================================
# PAYLOAD LENGTH
# ============================================================

def test_get_payload_length_plain_tile_none():
    tile = np.full((decode.TILE_SIZE, decode.TILE_SIZE, 3), 42, dtype=np.uint8)

    assert decode.get_payload_length(tile) is None


def test_get_payload_length_embedded_tile(key_dir, make_payload):
    private_key = key_dir[1]

    payload = make_payload(private_key)
    encoded = encode.reed_solomon_encode(payload)

    assert encoded is not None

    rng = np.random.default_rng(0)
    tile = np.full(
        (decode.TILE_SIZE, decode.TILE_SIZE, 3),
        130,
        dtype=np.uint8,
    )

    embedded = encode.embed_tile(tile, encode.bytes_to_bits(encoded))

    assert decode.get_payload_length(embedded) == len(payload)


# ============================================================
# PARSE PAYLOAD
# ============================================================

def test_parse_payload_valid(key_dir, make_payload):
    image_id = bytes(range(16))

    payload = make_payload(
        key_dir[1],
        prompt="hello world",
        model="flux",
        version="v3.2",
        image_id=image_id,
        timestamp=1234567890,
    )

    parsed = decode.parse_payload(payload)

    assert parsed["prompt"] == "hello world"
    assert parsed["model"] == "flux"
    assert parsed["version"] == "v3.2"
    assert parsed["image_id"] == image_id.hex()
    assert parsed["timestamp"] == 1234567890
    assert parsed["signature"] == payload[-64:]

    # body ist der unkomprimierte Body, nicht der Container
    assert parsed["body"][:4] == b"WM01"
    assert b"flux" in parsed["body"]


def test_parse_payload_too_small():
    with pytest.raises(ValueError, match="Payload too small"):
        decode.parse_payload(b"WM")


def test_parse_payload_bad_magic():
    with pytest.raises(ValueError, match="Invalid watermark magic"):
        decode.parse_payload(b"XXXX" + b"\x00" * 100)


def test_parse_payload_invalid_compressed_length(make_body):
    body = make_body()
    compressed = lzma.compress(body, lzma.FORMAT_XZ, preset=9)

    # Behauptet 500 Bytes, liefert aber nur wenige
    data = (
        b"WM01"
        + struct.pack(">H", 500)
        + compressed
        + b"\x00" * 64
    )

    with pytest.raises(
        ValueError, match="Invalid compressed body length"
    ):
        decode.parse_payload(data)


def test_parse_payload_bad_compressed_data():
    """Beschaedigter XZ-Header -> harter LZMA-Fehler."""
    data = (
        b"WM01"
        + struct.pack(">H", 12)
        + b"\xfd7zXZ\x00"
        + b"\xaa" * 6
        + b"\x00" * 64
    )

    with pytest.raises(
        ValueError, match="Body decompression failed"
    ):
        decode.parse_payload(data)


def test_parse_payload_incomplete_lzma_stream_no_data():
    """Daten, die zwar als XZ akzeptiert werden, aber sofort
    enden, werden abgelehnt. lzma.decompress wuerde hier
    stillschweigend nichts liefern."""
    data = (
        b"WM01"
        + struct.pack(">H", 8)
        + b"\xff" * 8
        + b"\x00" * 64
    )

    with pytest.raises(ValueError, match="Incomplete LZMA stream"):
        decode.parse_payload(data)


def test_parse_payload_truncated_signature(make_body):
    body = make_body()
    compressed = lzma.compress(body, lzma.FORMAT_XZ, preset=9)

    data = (
        b"WM01"
        + struct.pack(">H", len(compressed))
        + compressed
    )

    with pytest.raises(
        ValueError, match="Invalid compressed body length"
    ):
        decode.parse_payload(data)


def test_parse_payload_rejects_trailing_data(key_dir, make_payload):
    """Zusaetzliche Bytes hinter der Signatur deuten auf eine
    falsche Laengenannahme hin und duerfen nicht stillschweigend
    verworfen werden."""
    payload = make_payload(key_dir[1])

    with pytest.raises(ValueError, match="trailing data"):
        decode.parse_payload(payload + b"\x00" * 8)


def test_parse_payload_rejects_extra_compressed_bytes(make_body):
    """Der LZMA-Stream muss exakt zur Laengenangabe passen."""
    body = make_body()
    compressed = lzma.compress(body, lzma.FORMAT_XZ, preset=9)

    data = (
        b"WM01"
        + struct.pack(">H", len(compressed) + 4)
        + compressed
        + b"\x00" * 4
        + b"\x00" * 64
    )

    with pytest.raises(ValueError, match="LZMA"):
        decode.parse_payload(data)


def test_parse_payload_incomplete_lzma_stream(make_body):
    """Ein abgeschnittener LZMA-Stream wird abgelehnt, statt ein
    unvollstaendiges Ergebnis zu liefern."""
    body = make_body()
    compressed = lzma.compress(body, lzma.FORMAT_XZ, preset=9)

    data = (
        b"WM01"
        + struct.pack(">H", len(compressed) - 8)
        + compressed[:-8]
        + b"\x00" * 64
    )

    with pytest.raises(ValueError, match="LZMA"):
        decode.parse_payload(data)


# ============================================================
# PARSE BODY
# ============================================================

def test_parse_body_valid(make_body):
    image_id = bytes(range(16))

    body = make_body(
        prompt="hello world",
        model="flux",
        version="v3.2",
        image_id=image_id,
        timestamp=1234567890,
    )

    parsed = decode.parse_body(body)

    assert parsed["prompt"] == "hello world"
    assert parsed["model"] == "flux"
    assert parsed["version"] == "v3.2"
    assert parsed["image_id"] == image_id.hex()
    assert parsed["timestamp"] == 1234567890


def test_parse_body_bad_magic():
    with pytest.raises(ValueError, match="Invalid watermark magic"):
        decode.parse_body(b"XXXX" + b"\x00" * 10)


def test_parse_body_missing_model_length():
    with pytest.raises(ValueError, match="Missing model length"):
        decode.parse_body(b"WM01")


def test_parse_body_invalid_model_length():
    with pytest.raises(ValueError, match="Invalid model length"):
        decode.parse_body(b"WM01" + bytes([100]) + b"x" * 5)


def test_parse_body_missing_image_id():
    data = b"WM01" + b"\x00\x00" + b"\x00" * 10

    with pytest.raises(ValueError, match="Missing image ID"):
        decode.parse_body(data)


def test_parse_body_missing_timestamp():
    data = b"WM01" + b"\x00\x00" + b"\x00" * 16

    with pytest.raises(ValueError, match="Missing timestamp"):
        decode.parse_body(data)


def test_parse_body_missing_prompt_length():
    data = b"WM01" + b"\x00\x00" + b"\x00" * 16 + b"\x00" * 8

    with pytest.raises(ValueError, match="Missing prompt length"):
        decode.parse_body(data)


def test_parse_body_invalid_prompt_length():
    data = (
        b"WM01"
        + b"\x00\x00"
        + b"\x00" * 16
        + b"\x00" * 8
        + struct.pack(">H", 100)
        + b"x" * 5
    )

    with pytest.raises(ValueError, match="Invalid prompt length"):
        decode.parse_body(data)


def test_parse_body_bad_prompt_data():
    data = (
        b"WM01"
        + b"\x00\x00"
        + b"\x00" * 16
        + b"\x00" * 8
        + struct.pack(">H", 4)
        + b"\xff\xff\xff\xff"
    )

    with pytest.raises(
        ValueError, match="Prompt decompression failed"
    ):
        decode.parse_body(data)


# ============================================================
# VERIFY SIGNATURE
# ============================================================

def test_verify_signature_valid(key_dir, make_payload):
    payload = make_payload(key_dir[1])

    assert decode.verify_signature(decode.parse_payload(payload)) is True


def test_verify_signature_rejects_tampered_signature(key_dir, make_payload):
    payload_data = bytearray(make_payload(key_dir[1]))
    payload_data[-1] ^= 0xFF

    payload = decode.parse_payload(bytes(payload_data))

    assert decode.verify_signature(payload) is False


def test_parse_payload_rejects_tampered_body(key_dir, make_payload):
    payload_data = bytearray(make_payload(key_dir[1]))
    payload_data[35] ^= 0xFF

    with pytest.raises(
        ValueError, match="Body decompression failed"
    ):
        decode.parse_payload(bytes(payload_data))


# ============================================================
# DECODE IMAGE
# ============================================================

def test_decode_image_missing_file(key_dir, tmp_path):
    with pytest.raises(SystemExit) as exc:
        decode.decode_image(str(tmp_path / "nope.png"))

    assert "Image not found" in str(exc.value)


def test_decode_image_too_small(key_dir, tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    path = tmp_path / "small.png"
    Image.fromarray(img).save(path)

    with pytest.raises(SystemExit, match="Image must be at least"):
        decode.decode_image(str(path))


def test_decode_image_no_watermark(key_dir, tmp_path):
    img = np.full((512, 512, 3), 42, dtype=np.uint8)
    path = tmp_path / "plain.png"
    Image.fromarray(img).save(path)

    with pytest.raises(SystemExit, match="No valid watermark found"):
        decode.decode_image(str(path))


def test_decode_image_prefers_valid_signature(
    key_dir, make_payload, tmp_path
):
    """Ein frueher gefundener Tile mit falscher Signatur darf einen
    spaeteren Tile mit korrekt signiertem Payload nicht verdecken.

    Vor v0.2 wurde der erste parsebare Kandidat zurueckgegeben --
    auch wenn seine Signatur ungueltig war.
    """
    private_key = key_dir[1]
    foreign_key = Ed25519PrivateKey.generate()

    rng = np.random.default_rng(5)
    image = rng.integers(
        60, 200, (512, 1024, 3), dtype=np.uint8
    )

    # Tile (0,0): Payload mit fremdem Schluessel -> Signatur ungueltig
    # Tile (1,0): Payload mit echtem Schluessel  -> Signatur gueltig
    tiles = (
        (
            make_payload(
                foreign_key,
                prompt="forged",
                model="forged",
                version="v0.2",
                image_id=b"\x00" * 16,
            ),
            0,
        ),
        (
            make_payload(
                private_key,
                prompt="genuine",
                model="genuine",
                version="v0.2",
                image_id=b"\x01" * 16,
            ),
            512,
        ),
    )

    for payload, x in tiles:
        encoded = encode.reed_solomon_encode(payload)
        image[:, x:x + 512] = encode.embed_tile(
            image[:, x:x + 512],
            encode.bytes_to_bits(encoded),
        )

    path = str(tmp_path / "mixed.png")
    Image.fromarray(image).save(path)

    result = decode.decode_image(path)

    assert result is not None

    # Der echte Payload muss gewinnen, nicht der zuerst gefundene.
    assert result["prompt"] == "genuine"
    assert result["model"] == "genuine"
    assert decode.verify_signature(result) is True


def test_decode_image_reports_only_invalid_signature(
    key_dir, make_payload, tmp_path
):
    """Ohne gueltigen Kandidaten wird der parsebare Payload
    gemeldet, aber klar als INVALID markiert."""
    foreign_key = Ed25519PrivateKey.generate()

    rng = np.random.default_rng(6)
    image = rng.integers(
        60, 200, (512, 512, 3), dtype=np.uint8
    )

    payload = make_payload(
        foreign_key,
        prompt="forged",
        model="forged",
        version="v0.2",
    )

    encoded = encode.reed_solomon_encode(payload)
    image[:, :] = encode.embed_tile(
        image[:, :], encode.bytes_to_bits(encoded)
    )

    path = str(tmp_path / "forged.png")
    Image.fromarray(image).save(path)

    result = decode.decode_image(path)

    assert result is not None
    assert result["prompt"] == "forged"
    assert decode.verify_signature(result) is False
