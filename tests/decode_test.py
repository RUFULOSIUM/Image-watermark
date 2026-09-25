import struct
import zlib

import numpy as np
import pytest
from PIL import Image

import image_watermark.decode as decode
import image_watermark.encode as encode


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
    assert parsed["body"] == payload[:-64]


def test_parse_payload_too_small():
    with pytest.raises(ValueError, match="Payload too small"):
        decode.parse_payload(b"WM")


def test_parse_payload_bad_magic():
    with pytest.raises(ValueError, match="Invalid watermark magic"):
        decode.parse_payload(b"XXXX" + b"\x00" * 100)


def test_parse_payload_missing_model_length():
    with pytest.raises(ValueError, match="Missing model length"):
        decode.parse_payload(b"WM01")


def test_parse_payload_invalid_model_length():
    data = b"WM01" + bytes([100]) + b"x" * 5

    with pytest.raises(ValueError, match="Invalid model length"):
        decode.parse_payload(data)


def test_parse_payload_missing_image_id():
    data = b"WM01" + b"\x00\x00" + b"\x00" * 10

    with pytest.raises(ValueError, match="Missing image ID"):
        decode.parse_payload(data)


def test_parse_payload_missing_timestamp():
    data = b"WM01" + b"\x00\x00" + b"\x00" * 16

    with pytest.raises(ValueError, match="Missing timestamp"):
        decode.parse_payload(data)


def test_parse_payload_missing_prompt_length():
    data = b"WM01" + b"\x00\x00" + b"\x00" * 16 + b"\x00" * 8

    with pytest.raises(ValueError, match="Missing prompt length"):
        decode.parse_payload(data)


def test_parse_payload_invalid_prompt_length():
    data = (
        b"WM01"
        + b"\x00\x00"
        + b"\x00" * 16
        + b"\x00" * 8
        + struct.pack(">H", 100)
        + b"x" * 5
    )

    with pytest.raises(ValueError, match="Invalid prompt length"):
        decode.parse_payload(data)


def test_parse_payload_missing_signature():
    compressed = zlib.compress(b"")
    body = (
        b"WM01"
        + b"\x00\x00"
        + b"\x00" * 16
        + b"\x00" * 8
        + struct.pack(">H", len(compressed))
        + compressed
    )

    with pytest.raises(ValueError, match="Missing signature"):
        decode.parse_payload(body)


def test_parse_payload_bad_prompt_data():
    data = (
        b"WM01"
        + b"\x00\x00"
        + b"\x00" * 16
        + b"\x00" * 8
        + struct.pack(">H", 4)
        + b"\xff\xff\xff\xff"
        + b"\x00" * 64
    )

    with pytest.raises(ValueError, match="Prompt decompression failed"):
        decode.parse_payload(data)


# ============================================================
# VERIFY SIGNATURE
# ============================================================

def test_verify_signature_valid(key_dir, make_payload):
    payload = make_payload(key_dir[1])

    assert decode.verify_signature(decode.parse_payload(payload)) is True


def test_verify_signature_rejects_tampered_body(key_dir, make_payload):
    payload_data = bytearray(make_payload(key_dir[1]))
    payload_data[35] ^= 0xFF

    payload = decode.parse_payload(bytes(payload_data))

    assert decode.verify_signature(payload) is False


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