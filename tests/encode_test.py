import struct
import lzma
import random
import string as string_module

import numpy as np
import pytest
from reedsolo import RSCodec

import image_watermark.decode as decode
import image_watermark.encode as encode
from image_watermark.rs import chunk_count, max_message_length, rs_len


def make_incompressible_prompt(length, seed=7):
    """Random printable text, so LZMA cannot shrink it below the limit."""
    return "".join(
        random.Random(seed).choices(
            string_module.printable,
            k=length,
        )
    )


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
# REED SOLOMON CHUNKING
#
# reedsolo haengt 32 Parity-Byte an JEDEN 223-Byte-Block an.
# Der Decoder liest genau rs_len(n) Byte aus dem Tile. Stimmt
# diese Formel nicht, ist jedes Payload groesser als ein Chunk
# nicht dekodierbar. Das war der Bug in v0.1.
# ============================================================

@pytest.mark.parametrize(
    "length",
    [0, 1, 100, 222, 223, 224, 300, 380, 445, 446, 447, 480],
)
def test_rs_len_matches_reedsolo(length):
    assert len(RSCodec(encode.RS_PARITY).encode(bytes(length))) == (
        rs_len(length)
    )


def test_rs_len_chunk_boundaries():
    # 223 Byte passen in einen Chunk, 224 brauchen schon zwei.
    assert chunk_count(223) == 1
    assert chunk_count(224) == 2

    assert rs_len(223) == 255
    assert rs_len(224) == 288
    assert rs_len(380) == 444


def test_tile_capacity_is_not_tile_bytes_minus_parity():
    """Die Kapazitaet ist kleiner als 512 - 32, weil ab 447 Byte
    ein dritter Chunk dazukommt."""
    assert encode.TILE_BYTES == 512
    assert encode.MAX_PAYLOAD_SIZE == 446
    assert encode.MAX_PAYLOAD_SIZE < encode.TILE_BYTES - encode.RS_PARITY

    assert rs_len(encode.MAX_PAYLOAD_SIZE) <= encode.TILE_BYTES
    assert rs_len(encode.MAX_PAYLOAD_SIZE + 1) > encode.TILE_BYTES

    assert max_message_length(encode.TILE_BYTES) == 446


def test_max_compressed_length_leaves_room_for_header_and_signature():
    assert encode.MAX_COMPRESSED_LENGTH == (
        encode.MAX_PAYLOAD_SIZE
        - encode.HEADER_SIZE
        - encode.SIGNATURE_SIZE
    )
    assert encode.MAX_COMPRESSED_LENGTH == 376


@pytest.mark.parametrize("length", [223, 224, 300, 380, 446])
def test_reed_solomon_encode_length_matches_rs_len(length):
    encoded = encode.reed_solomon_encode(bytes(length))

    assert len(encoded) == rs_len(length)


def test_reed_solomon_encode_multi_chunk_roundtrip():
    """Ein Payload ueber 223 Byte muss mehrere Chunks benutzen
    und trotzdem verlustfrei dekodierbar sein."""
    data = bytes(range(256)) * 2
    assert len(data) == 512

    encoded = encode.reed_solomon_encode(data)

    assert len(encoded) == rs_len(len(data))
    assert chunk_count(len(data)) == 3

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
    payload = encode.create_compressed_payload("a prompt", "model-1", "v1.0")

    assert payload[:4] == b"WM01"
    assert len(payload) > 64

    # Laenge des komprimierten Bodies steht im Klartext
    compressed_length = struct.unpack(">H", payload[4:6])[0]

    assert compressed_length == len(payload) - 6 - 64

    parsed = decode.parse_payload(payload)
    assert parsed["prompt"] == "a prompt"
    assert parsed["model"] == "model-1"
    assert parsed["version"] == "v1.0"
    assert len(parsed["image_id"]) == 32
    assert isinstance(parsed["timestamp"], int)
    assert len(parsed["signature"]) == 64

    # Signiert wird der unkomprimierte Body
    assert parsed["body"][:4] == b"WM01"
    assert b"model-1" in parsed["body"]
    assert payload[-64:] == parsed["signature"]


def test_create_payload_body_is_compressed(key_dir):
    payload = encode.create_compressed_payload("a prompt", "model-1", "v1.0")

    compressed_length = struct.unpack(">H", payload[4:6])[0]
    compressed_body = payload[6:6 + compressed_length]

    # XZ-Container
    assert compressed_body.startswith(b"\xfd7zXZ\x00")

    # Dekomprimierbar, und der innere Body startet mit dem Magic
    assert lzma.decompress(compressed_body)[:4] == b"WM01"

    # Signatur liegt hinter dem komprimierten Body
    assert payload[6 + compressed_length:] == decode.parse_payload(
        payload
    )["signature"]


def test_create_payload_signature_is_valid(key_dir):
    payload = encode.create_compressed_payload("signed prompt", "m", "v")

    assert decode.verify_signature(decode.parse_payload(payload)) is True


def test_create_payload_model_too_long(key_dir):
    with pytest.raises(SystemExit) as exc:
        encode.create_compressed_payload("p", "m" * 256, "v")

    assert "Model name is too long" in str(exc.value)


def test_create_payload_version_too_long(key_dir):
    with pytest.raises(SystemExit) as exc:
        encode.create_compressed_payload("p", "m", "v" * 256)

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
        encode.create_compressed_payload(prompt, "m", "v")

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


# ============================================================
# END-TO-END, PAYLOAD GROESSER ALS EIN RS-CHUNK
#
# Regressionstest fuer den v0.1-Bug: der Decoder berechnete die
# Codeword-Laenge als n + 32. Ab 224 Byte Payload ist die
# tatsaechliche Laenge groesser, und jedes solche Bild wurde
# als "No valid watermark found" abgewiesen.
# ============================================================

@pytest.mark.parametrize("prompt_length", [150, 200, 250])
def test_encode_image_end_to_end_multi_chunk(
    key_dir, rgb_image, tmp_path, prompt_length
):
    prompt = make_incompressible_prompt(prompt_length)
    output = str(tmp_path / f"out{prompt_length}.png")

    payload_bytes = len(
        encode.create_compressed_payload(prompt, "m", "v0.2")
    )

    # Der Test ist nur aussagekraeftig, wenn der Payload
    # wirklich mehrere RS-Chunks belegt.
    assert payload_bytes > 223
    assert chunk_count(payload_bytes) >= 2

    encode.encode_image(
        rgb_image,
        output,
        prompt=prompt,
        model="m",
        version="v0.2",
    )

    payload = decode.decode_image(output)

    assert payload is not None
    assert payload["prompt"] == prompt
    assert payload["version"] == "v0.2"
    assert decode.verify_signature(payload) is True


def test_encode_image_rejects_payload_too_large(
    key_dir, rgb_image, tmp_path
):
    """Ueber der Kapazitaet muss der Encoder klar ablehnen,
    statt spaeter einen unlesbaren Payload zu erzeugen."""
    prompt = make_incompressible_prompt(600)

    with pytest.raises(SystemExit) as exc:
        encode.encode_image(
            rgb_image,
            str(tmp_path / "out.png"),
            prompt=prompt,
            model="m",
            version="v0.2",
        )

    assert "too large" in str(exc.value)


def test_create_payload_rejects_oversized_compressed_body(key_dir):
    with pytest.raises(SystemExit) as exc:
        encode.create_compressed_payload(
            make_incompressible_prompt(600),
            "m",
            "v0.2",
        )

    assert "too large" in str(exc.value)


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