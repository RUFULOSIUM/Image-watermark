import numpy as np
import pytest
from reedsolo import RSCodec

import image_watermark.decode as decode
import image_watermark.encode as encode


# ============================================================
# BYTES -> BITS
# ============================================================

def test_bytes_to_bits_known_values():
    assert encode.bytes_to_bits(b"\xff") == [1] * 8
    assert encode.bytes_to_bits(b"\x00") == [0] * 8
    assert encode.bytes_to_bits(b"\x80") == [1, 0, 0, 0, 0, 0, 0, 0]
    assert encode.bytes_to_bits(b"\x01") == [0, 0, 0, 0, 0, 0, 0, 1]


def test_bytes_to_bits_roundtrip():
    data = bytes(range(256))
    assert decode.bits_to_bytes(encode.bytes_to_bits(data)) == data


# ============================================================
# REED SOLOMON
# ============================================================

def test_reed_solomon_encode_roundtrip():
    data = b"hello watermark payload" * 4
    encoded = encode.reed_solomon_encode(data)

    assert encoded is not None
    assert len(encoded) == len(data) + encode.RS_PARITY

    decoded = RSCodec(encode.RS_PARITY).decode(encoded)[0]
    assert bytes(decoded) == data


# ============================================================
# EMBED BIT
# ============================================================

@pytest.mark.parametrize("bit", [0, 1])
def test_embed_bit_roundtrip(bit):
    rng = np.random.default_rng(42)
    block = rng.integers(0, 256, (8, 8), dtype=np.uint8)

    embedded = encode.embed_bit(block, bit)

    assert embedded.shape == (8, 8)
    assert decode.extract_bit(embedded) == bit


def test_embed_bit_clips_to_valid_range():
    rng = np.random.default_rng(1)
    block = rng.integers(0, 256, (8, 8), dtype=np.uint8)

    for bit in (0, 1):
        embedded = encode.embed_bit(block, bit)
        assert embedded.min() >= 0
        assert embedded.max() <= 255


# ============================================================
# EMBED TILE
# ============================================================

def test_embed_tile_roundtrip():
    rng = np.random.default_rng(7)
    tile = np.full(
        (encode.TILE_SIZE, encode.TILE_SIZE, 3),
        120,
        dtype=np.uint8,
    )
    bits = list(rng.integers(0, 2, 300).astype(int))

    embedded = encode.embed_tile(tile, bits)

    assert decode.extract_bits(embedded, len(bits)) == bits


# ============================================================
# PAYLOAD
# ============================================================

def test_create_payload_format(key_dir):
    payload = encode.create_payload("a prompt", "model-1", "v1.0")

    assert payload[:4] == b"WM01"
    assert len(payload) > 64

    parsed = decode.parse_payload(payload)
    assert parsed["prompt"] == "a prompt"
    assert parsed["model"] == "model-1"
    assert parsed["version"] == "v1.0"
    assert len(parsed["image_id"]) == 32
    assert isinstance(parsed["timestamp"], int)
    assert len(parsed["signature"]) == 64
    assert parsed["body"] == payload[:-64]


def test_create_payload_signature_is_valid(key_dir):
    payload = encode.create_payload("signed prompt", "m", "v")

    assert decode.verify_signature(decode.parse_payload(payload)) is True


def test_create_payload_model_too_long(key_dir):
    with pytest.raises(SystemExit) as exc:
        encode.create_payload("p", "m" * 256, "v")

    assert "Model name is too long" in str(exc.value)


def test_create_payload_version_too_long(key_dir):
    with pytest.raises(SystemExit) as exc:
        encode.create_payload("p", "m", "v" * 256)

    assert "Model version is too long" in str(exc.value)


def test_create_payload_prompt_too_large(key_dir):
    import random
    import string as string_module

    prompt = "".join(
        random.Random(1).choices(
            string_module.printable,
            k=90000,
        )
    )

    with pytest.raises(SystemExit) as exc:
        encode.create_payload(prompt, "m", "v")

    assert "Prompt is too large" in str(exc.value)


# ============================================================
# END-TO-END
# ============================================================

def test_encode_image_end_to_end(key_dir, rgb_image, tmp_path):
    output = str(tmp_path / "out.png")

    encode.encode_image(
        rgb_image,
        output,
        prompt="integration prompt",
        model="model",
        version="v1",
    )

    payload = decode.decode_image(output)

    assert payload is not None
    assert payload["prompt"] == "integration prompt"
    assert payload["model"] == "model"
    assert payload["version"] == "v1"
    assert decode.verify_signature(payload) is True


def test_encode_image_missing_input(key_dir, tmp_path):
    with pytest.raises(SystemExit) as exc:
        encode.encode_image(
            str(tmp_path / "nope.png"),
            str(tmp_path / "out.png"),
            prompt="p",
            model="m",
            version="v",
        )

    assert "Input image not found" in str(exc.value)


def test_encode_image_too_small(key_dir, tmp_path):
    import PIL.Image as PILImage

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    path = tmp_path / "small.png"
    PILImage.fromarray(img).save(path)

    with pytest.raises(SystemExit, match="Image must be at least") as exc:
        encode.encode_image(
            str(path),
            str(tmp_path / "out.png"),
            prompt="p",
            model="m",
            version="v",
        )