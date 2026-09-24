import os
import sys
import struct
import zlib

import cv2
import numpy as np
from PIL import Image

from reedsolo import RSCodec
from cryptography.hazmat.primitives import serialization


# ============================================================
# CONFIG
# ============================================================

TILE_SIZE = 512
RS_PARITY = 32

COEF_A = (3, 4)
COEF_B = (4, 3)


# ============================================================
# CLEAN EXIT
# ============================================================

def error(message):
    sys.exit(f"ERROR: {message}")


# ============================================================
# PUBLIC KEY
# ============================================================

def load_public_key():

    if not os.path.exists("public_key.pem"):
        error("public_key.pem not found.")

    try:
        with open("public_key.pem", "rb") as f:
            return serialization.load_pem_public_key(
                f.read()
            )

    except Exception as e:
        error(f"Could not load public key: {e}")


# ============================================================
# EXTRACT BIT
# ============================================================

def extract_bit(block):

    block = block.astype(np.float32)

    dct = cv2.dct(block)

    a = dct[COEF_A]
    b = dct[COEF_B]

    return 1 if a > b else 0


# ============================================================
# EXTRACT BITS
# ============================================================

def extract_bits(tile, bit_count):

    ycrcb = cv2.cvtColor(
        tile,
        cv2.COLOR_RGB2YCrCb
    )

    luminance = ycrcb[:, :, 0]

    bits = []

    for y in range(0, TILE_SIZE, 8):

        for x in range(0, TILE_SIZE, 8):

            if len(bits) >= bit_count:
                return bits

            block = luminance[
                y:y + 8,
                x:x + 8
            ]

            bits.append(
                extract_bit(block)
            )

    return bits


# ============================================================
# BITS -> BYTES
# ============================================================

def bits_to_bytes(bits):

    output = bytearray()

    for i in range(0, len(bits) - 7, 8):

        value = 0

        for bit in bits[i:i + 8]:
            value = (value << 1) | bit

        output.append(value)

    return bytes(output)


# ============================================================
# READ PAYLOAD LENGTH
# ============================================================

def get_payload_length(tile):

    # --------------------------------------------------------
    # We need enough bytes to reach the prompt length.
    #
    # Minimum structure before prompt:
    #
    # WM01       4
    # model len  1
    # model      0+
    # version    1
    # version    0+
    # UUID       16
    # timestamp  8
    # prompt len 2
    #
    # With empty model/version:
    #
    # 4 + 1 + 1 + 16 + 8 + 2 = 32
    #
    # Read a bit more to safely reach it.
    # --------------------------------------------------------

    bootstrap_bytes = 64

    bits = extract_bits(
        tile,
        bootstrap_bytes * 8
    )

    data = bits_to_bytes(bits)

    if len(data) < 34:
        return None

    # --------------------------------------------------------
    # Magic
    # --------------------------------------------------------

    if data[:4] != b"WM01":
        return None

    pos = 4

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if pos >= len(data):
        return None

    model_len = data[pos]
    pos += 1

    if pos + model_len > len(data):
        return None

    pos += model_len

    # --------------------------------------------------------
    # Version
    # --------------------------------------------------------

    if pos >= len(data):
        return None

    version_len = data[pos]
    pos += 1

    if pos + version_len > len(data):
        return None

    pos += version_len

    # --------------------------------------------------------
    # UUID
    # --------------------------------------------------------

    if pos + 16 > len(data):
        return None

    pos += 16

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if pos + 8 > len(data):
        return None

    pos += 8

    # --------------------------------------------------------
    # Prompt length
    # --------------------------------------------------------

    if pos + 2 > len(data):
        return None

    prompt_len = struct.unpack(
        ">H",
        data[pos:pos + 2]
    )[0]

    pos += 2

    # --------------------------------------------------------
    # Total payload:
    #
    # body + 64 byte Ed25519 signature
    # --------------------------------------------------------

    body_length = pos + prompt_len

    total_payload_length = (
        body_length + 64
    )

    return total_payload_length


# ============================================================
# PARSE PAYLOAD
# ============================================================

def parse_payload(data):

    if len(data) < 4:
        raise ValueError("Payload too small.")

    if data[:4] != b"WM01":
        raise ValueError("Invalid watermark magic.")

    pos = 4

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if pos >= len(data):
        raise ValueError("Missing model length.")

    model_len = data[pos]
    pos += 1

    if pos + model_len > len(data):
        raise ValueError("Invalid model length.")

    model = data[
        pos:pos + model_len
    ].decode("utf-8")

    pos += model_len

    # --------------------------------------------------------
    # Version
    # --------------------------------------------------------

    if pos >= len(data):
        raise ValueError("Missing version length.")

    version_len = data[pos]
    pos += 1

    if pos + version_len > len(data):
        raise ValueError("Invalid version length.")

    version = data[
        pos:pos + version_len
    ].decode("utf-8")

    pos += version_len

    # --------------------------------------------------------
    # UUID
    # --------------------------------------------------------

    if pos + 16 > len(data):
        raise ValueError("Missing image ID.")

    image_id = data[
        pos:pos + 16
    ].hex()

    pos += 16

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if pos + 8 > len(data):
        raise ValueError("Missing timestamp.")

    timestamp = struct.unpack(
        ">Q",
        data[pos:pos + 8]
    )[0]

    pos += 8

    # --------------------------------------------------------
    # Prompt length
    # --------------------------------------------------------

    if pos + 2 > len(data):
        raise ValueError("Missing prompt length.")

    prompt_len = struct.unpack(
        ">H",
        data[pos:pos + 2]
    )[0]

    pos += 2

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    if pos + prompt_len > len(data):
        raise ValueError("Invalid prompt length.")

    compressed_prompt = data[
        pos:pos + prompt_len
    ]

    pos += prompt_len

    # --------------------------------------------------------
    # Signature
    # --------------------------------------------------------

    if pos + 64 > len(data):
        raise ValueError("Missing signature.")

    signature = data[
        pos:pos + 64
    ]

    body = data[:pos]

    # --------------------------------------------------------
    # Decompress prompt
    # --------------------------------------------------------

    try:
        prompt = zlib.decompress(
            compressed_prompt
        ).decode("utf-8")

    except Exception as e:
        raise ValueError(
            f"Prompt decompression failed: {e}"
        )

    return {
        "model": model,
        "version": version,
        "image_id": image_id,
        "timestamp": timestamp,
        "prompt": prompt,
        "signature": signature,
        "body": body
    }


# ============================================================
# VERIFY SIGNATURE
# ============================================================

def verify_signature(payload):

    public_key = load_public_key()

    try:

        public_key.verify(
            payload["signature"],
            payload["body"]
        )

        return True

    except Exception:

        return False


# ============================================================
# DECODE TILE
# ============================================================

def decode_tile(tile, byte_count):

    bits = extract_bits(
        tile,
        byte_count * 8
    )

    return bits_to_bytes(bits)


# ============================================================
# DECODE IMAGE
# ============================================================

def decode_image(path):

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(path):
        error(
            f"Image not found: {path}"
        )

    print("[1/5] Loading image...")

    try:

        image = np.array(
            Image.open(path).convert("RGB")
        )

    except Exception as e:

        error(
            f"Could not open image: {e}"
        )

    height, width, _ = image.shape

    if width < TILE_SIZE or height < TILE_SIZE:

        error(
            f"Image must be at least "
            f"{TILE_SIZE}x{TILE_SIZE} pixels."
        )

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    max_bits = (
        (TILE_SIZE // 8) *
        (TILE_SIZE // 8)
    )

    max_bytes = max_bits // 8

    print(
        f"      Tile size: {TILE_SIZE}x{TILE_SIZE}"
    )

    print(
        f"      Capacity: {max_bytes} bytes"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    print("[2/5] Searching watermark...")

    tiles_x = width // TILE_SIZE
    tiles_y = height // TILE_SIZE

    rs = RSCodec(
        RS_PARITY
    )

    candidates = []

    for ty in range(tiles_y):

        for tx in range(tiles_x):

            x = tx * TILE_SIZE
            y = ty * TILE_SIZE

            tile = image[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            # ------------------------------------------------
            # Find payload size
            # ------------------------------------------------

            payload_length = get_payload_length(
                tile
            )

            if payload_length is None:
                continue

            print(
                f"      Found header at tile "
                f"({tx},{ty})"
            )

            # ------------------------------------------------
            # Reed-Solomon adds parity
            # ------------------------------------------------

            encoded_length = (
                payload_length +
                RS_PARITY
            )

            if encoded_length > max_bytes:

                print(
                    f"      Payload too large: "
                    f"{encoded_length} > {max_bytes}"
                )

                continue

            # ------------------------------------------------
            # Extract complete encoded data
            # ------------------------------------------------

            try:

                raw = decode_tile(
                    tile,
                    encoded_length
                )

                decoded = rs.decode(
                    raw
                )[0]

                candidates.append(
                    bytes(decoded)
                )

                print(
                    f"      Tile ({tx},{ty}) "
                    f"decoded successfully"
                )

            except Exception as e:

                print(
                    f"      Tile ({tx},{ty}) "
                    f"failed: {e}"
                )

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    if not candidates:

        error(
            "No valid watermark found."
        )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    print("[3/5] Parsing payload...")

    for candidate in candidates:

        try:

            payload = parse_payload(
                candidate
            )

        except Exception as e:

            print(
                f"      Invalid payload: {e}"
            )

            continue

        # ----------------------------------------------------
        # Verify
        # ----------------------------------------------------

        print("[4/5] Verifying signature...")

        valid = verify_signature(
            payload
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        print("[5/5] Done.")
        print()

        print("================================")
        print(" WATERMARK FOUND")
        print("================================")

        print(
            f"Model:       "
            f"{payload['model'] or '(empty)'}"
        )

        print(
            f"Version:     "
            f"{payload['version'] or '(empty)'}"
        )

        print(
            f"Image ID:    "
            f"{payload['image_id']}"
        )

        print(
            f"Timestamp:   "
            f"{payload['timestamp']}"
        )

        print(
            f"Signature:   "
            f"{'VALID' if valid else 'INVALID'}"
        )

        print()

        print("Prompt:")
        print(
            payload["prompt"] or "(empty)"
        )

        print("================================")

        return payload

    error(
        "Watermark data was found, "
        "but could not be decoded."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    decode_image(
        "watermarked.png"
    )