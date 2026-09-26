import os
import sys
import time
import struct
import zlib
import lzma
import uuid

from typing import NoReturn

import cv2
import numpy as np
from PIL import Image

from reedsolo import RSCodec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from image_watermark.rs import RS_PARITY, max_message_length, rs_len


# ============================================================
# CONFIG
# ============================================================

TILE_SIZE = 512

MAGIC = b"WM01"

SIGNATURE_SIZE = 64

# Magic (4) + Laenge des komprimierten Bodies (2)
HEADER_SIZE = 6

COEF_A = (3, 4)
COEF_B = (4, 3)

STRENGTH = 12.0

# ---------------------------------------------------------
# Kapazitaet
#
# Ein 512x512 Tile hat 64x64 = 4096 Bloecke, also 4096 Bit
# = 512 Byte fuer das RS-Codeword.
# ---------------------------------------------------------

TILE_BLOCKS = (TILE_SIZE // 8) ** 2
TILE_BYTES = TILE_BLOCKS // 8

# reedsolo haengt 32 Parity-Byte an JEDEN 223-Byte-Block an.
# Ein Payload passt also nur in einen Tile, wenn sein Codeword
# die 512 Byte nicht ueberschreitet. Die Grenze wird berechnet,
# nicht geraten.
MAX_PAYLOAD_SIZE = max_message_length(TILE_BYTES)

MAX_COMPRESSED_LENGTH = (
    MAX_PAYLOAD_SIZE - HEADER_SIZE - SIGNATURE_SIZE
)


# ============================================================
# CLEAN EXIT
# ============================================================

def error(message) -> NoReturn:
    sys.exit(f"ERROR: {message}")


# ============================================================
# KEYS
# ============================================================

def create_keys():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    with open("private_key.pem", "wb") as f:
        f.write(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            )
        )

    with open("public_key.pem", "wb") as f:
        f.write(
            public_key.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )

    print("Created private_key.pem")
    print("Created public_key.pem")


def load_private_key() -> Ed25519PrivateKey:
    if not os.path.exists("private_key.pem"):
        error(
            "private_key.pem not found. "
            "Run the encoder again to create one."
        )

    try:
        with open("private_key.pem", "rb") as f:
            key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )
    except Exception:
        error("Could not load private_key.pem.")

    if not isinstance(key, Ed25519PrivateKey):
        error("private_key.pem is not an Ed25519 key.")

    return key


# ============================================================
# PAYLOAD
# ============================================================

def create_compressed_payload(prompt, model, version):

    model_bytes = model.encode("utf-8")
    version_bytes = version.encode("utf-8")
    prompt_bytes = prompt.encode("utf-8")

    if len(model_bytes) > 255:
        error("Model name is too long.")

    if len(version_bytes) > 255:
        error("Model version is too long.")

    compressed_prompt = zlib.compress(prompt_bytes)

    if len(compressed_prompt) > 65535:
        error("Prompt is too large.")

    image_id = uuid.uuid4().bytes
    timestamp = int(time.time())

    # --------------------------------------------------------
    # Payload body
    #
    # Unkomprimiert. Wird mit LZMA komprimiert und
    # anschliessend signiert.
    # --------------------------------------------------------
    body = (

        MAGIC +

        # Model
        bytes([len(model_bytes)]) +
        model_bytes +

        # Version
        bytes([len(version_bytes)]) +
        version_bytes +

        # Image ID
        image_id +

        # Timestamp
        struct.pack(">Q", timestamp) +

        # Prompt length
        struct.pack(">H", len(compressed_prompt)) +

        # Prompt
        compressed_prompt
    )

    compressed_body = lzma.compress(
        body,
        lzma.FORMAT_XZ,
        preset=9
    )

    if len(compressed_body) > MAX_COMPRESSED_LENGTH:
        error(
            f"Compressed payload is too large "
            f"({len(compressed_body)} bytes, "
            f"max {MAX_COMPRESSED_LENGTH})."
        )

    # --------------------------------------------------------
    # Sign body
    #
    # Signiert wird der unkomprimierte Body, nicht der
    # LZMA-Stream. So ist die Signatur unabhaengig von der
    # Kompressions-Einstellung.
    # --------------------------------------------------------

    private_key = load_private_key()

    signature = private_key.sign(body)

    # --------------------------------------------------------
    # Container
    #
    # Die Laenge des komprimierten Bodies steht im Klartext,
    # weil der Decoder sie kennen muss, BEVOR er den Body
    # dekomprimieren kann.
    # --------------------------------------------------------

    return (
        MAGIC +
        struct.pack(
            ">H",
            len(compressed_body)
        ) +
        compressed_body +
        signature
    )


# ============================================================
# REED SOLOMON
# ============================================================

def reed_solomon_encode(data):

    rs = RSCodec(RS_PARITY)

    try:
        encoded = bytes(rs.encode(data))
    except Exception as e:
        error(f"Reed-Solomon encoding failed: {e}")

    # Der Decoder berechnet die Laenge des Codewords ueber
    # rs_len(). Wenn reedsolo anders chunkt, ist die Laenge
    # falsch und jedes Payload groesser als ein Chunk ist
    # nicht mehr dekodierbar. Deshalb hier pruefen.
    expected = rs_len(len(data))

    if len(encoded) != expected:
        error(
            f"Unexpected Reed-Solomon length: "
            f"got {len(encoded)} bytes, expected {expected}."
        )

    return encoded


# ============================================================
# BYTES -> BITS
# ============================================================

def bytes_to_bits(data):

    bits = []

    for byte in data:

        for i in range(7, -1, -1):

            bits.append(
                (byte >> i) & 1
            )

    return bits


# ============================================================
# EMBED BIT
# ============================================================

def embed_bit(block, bit):

    block = block.astype(np.float32)

    dct = cv2.dct(block)

    a = dct[COEF_A]
    b = dct[COEF_B]

    if bit == 1:

        if a <= b:
            dct[COEF_A] = b + STRENGTH
            dct[COEF_B] = b
        else:
            dct[COEF_A] = a
            dct[COEF_B] = a - STRENGTH

    else:

        if b <= a:
            dct[COEF_B] = a + STRENGTH
            dct[COEF_A] = a
        else:
            dct[COEF_B] = b
            dct[COEF_A] = b - STRENGTH

    result = cv2.idct(dct)

    return np.clip(result, 0, 255)


# ============================================================
# EMBED TILE
# ============================================================

def embed_tile(tile, bits):

    ycrcb = cv2.cvtColor(
        tile,
        cv2.COLOR_RGB2YCrCb
    )

    luminance = ycrcb[:, :, 0].astype(np.float32)

    bit_index = 0

    for y in range(0, TILE_SIZE, 8):

        for x in range(0, TILE_SIZE, 8):

            # Keine Bits mehr?
            # Einfach die restlichen Blöcke unverändert lassen.
            if bit_index >= len(bits):
                break

            block = luminance[
                y:y + 8,
                x:x + 8
            ]

            luminance[
                y:y + 8,
                x:x + 8
            ] = embed_bit(
                block,
                bits[bit_index]
            )

            bit_index += 1

        # Wenn alle Bits geschrieben wurden,
        # äußere Schleife ebenfalls verlassen.
        if bit_index >= len(bits):
            break

    # WICHTIG:
    # Veränderte Luminanz zurückschreiben.
    ycrcb[:, :, 0] = luminance

    result = cv2.cvtColor(
        np.clip(
            ycrcb,
            0,
            255
        ).astype(np.uint8),
        cv2.COLOR_YCrCb2RGB
    )

    return result

# ============================================================
# ENCODE IMAGE
# ============================================================

def encode_image(
    input_path,
    output_path,
    prompt,
    model,
    version
):

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not os.path.exists(input_path):
        error(
            f"Input image not found: {input_path}"
        )

    print("[1/5] Loading image...")

    try:
        image = np.array(
            Image.open(input_path).convert("RGB")
        )
    except Exception as e:
        error(f"Could not open image: {e}")

    height, width, _ = image.shape

    if width < TILE_SIZE or height < TILE_SIZE:
        error(
            f"Image must be at least "
            f"{TILE_SIZE}x{TILE_SIZE} pixels."
        )

    # --------------------------------------------------------
    # Payload
    # --------------------------------------------------------

    print("[2/5] Creating payload...")

    payload = create_compressed_payload(
        prompt,
        model,
        version
    )

    print(
        f"      Payload: {len(payload)} bytes"
    )

    # --------------------------------------------------------
    # Reed Solomon
    # --------------------------------------------------------

    print("[3/5] Reed-Solomon encoding...")

    encoded = reed_solomon_encode(
        payload
    )

    print(
        f"      Encoded: {len(encoded)} bytes"
    )

    # --------------------------------------------------------
    # Kapazitaet
    #
    # Ein Tile hat 64x64 = 4096 DCT-Bloecke und damit
    # 4096 Bit = 512 Byte Platz. Das RS-Codeword muss
    # komplett hineinpassen.
    # --------------------------------------------------------

    if len(payload) > MAX_PAYLOAD_SIZE:
        error(
            f"Payload too large for one tile "
            f"({len(payload)} bytes, max {MAX_PAYLOAD_SIZE})."
        )

    bits = bytes_to_bits(
        encoded
    )

    if len(bits) > TILE_BLOCKS:
        error(
            f"Encoded payload too large for one tile "
            f"({len(bits)} bits, max {TILE_BLOCKS})."
        )

    print(
        f"      Watermark: {len(bits)} bits"
    )

    # --------------------------------------------------------
    # Embed
    # --------------------------------------------------------

    print("[4/5] Embedding watermark...")

    output = image.copy()

    tiles_x = width // TILE_SIZE
    tiles_y = height // TILE_SIZE

    tile_count = 0

    for ty in range(tiles_y):

        for tx in range(tiles_x):

            x = tx * TILE_SIZE
            y = ty * TILE_SIZE

            tile = output[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            output[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] = embed_tile(
                tile,
                bits
            )

            tile_count += 1

    print(
        f"      Embedded into {tile_count} tiles"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("[5/5] Saving...")

    try:
        Image.fromarray(
            output
        ).save(
            output_path,
            format="PNG"
        )
    except Exception as e:
        error(
            f"Could not save output: {e}"
        )

    print()
    print("================================")
    print(" WATERMARK CREATED")
    print("================================")
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Model:       {model or '(empty)'}")
    print(f"Version:     {version or '(empty)'}")
    print(f"Prompt:      {prompt or '(empty)'}")
    print(f"Payload:     {len(payload)} bytes")
    print(f"Encoded:     {len(encoded)} bytes")
    print(f"Tiles:       {tile_count}")
    print("================================")


# ============================================================
# MAIN
# ============================================================
if not os.path.exists("private_key.pem"):
    print("No keys found.")
    print("Generating Ed25519 key pair...")
    create_keys()
    print()

if __name__ == "__main__":



    encode_image(
        input_path="input.png",
        output_path="watermarked.png",

        prompt="",

        model="",

        version=""
    )