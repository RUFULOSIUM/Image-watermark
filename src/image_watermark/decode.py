import os
import sys
import struct
import zlib
import lzma

from typing import NoReturn

import cv2
import numpy as np
from PIL import Image

from reedsolo import RSCodec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)

from image_watermark.rs import RS_PARITY, rs_len


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

# Ein Tile hat 64x64 DCT-Bloecke, also 512 Byte Bit-Kapazitaet.
TILE_BLOCKS = (TILE_SIZE // 8) ** 2
TILE_BYTES = TILE_BLOCKS // 8


# ============================================================
# CLEAN EXIT
# ============================================================

def error(message) -> NoReturn:
    sys.exit(f"ERROR: {message}")


# ============================================================
# PUBLIC KEY
# ============================================================

def load_public_key() -> Ed25519PublicKey:

    if not os.path.exists("public_key.pem"):
        error("public_key.pem not found.")

    try:
        with open("public_key.pem", "rb") as f:
            key = serialization.load_pem_public_key(
                f.read()
            )

    except Exception as e:
        error(f"Could not load public key: {e}")

    if not isinstance(key, Ed25519PublicKey):
        error("public_key.pem is not an Ed25519 key.")

    return key


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
    # Der komplette Body ist LZMA-komprimiert. Seine innere
    # Struktur (Model, Version, UUID, Timestamp, Prompt) ist
    # deshalb nicht im Klartext lesbar.
    #
    # Nur der Container-Header steht unkomprimiert:
    #
    # magic              4
    # compressed length  2
    #
    # Die Laenge wird gebraucht, um zu wissen wie viele Bytes
    # aus dem Bild gelesen werden muessen. Ohne sie ist das
    # Reed-Solomon-Decoding nicht moeglich, weil die Laenge des
    # Codeworts exakt bekannt sein muss.
    # --------------------------------------------------------

    bits = extract_bits(
        tile,
        HEADER_SIZE * 8
    )

    data = bits_to_bytes(bits)

    # --------------------------------------------------------
    # Magic
    # --------------------------------------------------------

    if data[:4] != MAGIC:
        return None

    # --------------------------------------------------------
    # Laenge des komprimierten Bodies
    # --------------------------------------------------------

    compressed_length = struct.unpack(
        ">H",
        data[4:HEADER_SIZE]
    )[0]

    # 0 kann kein gueltiger LZMA-Stream sein
    if compressed_length == 0:
        return None

    return (
        HEADER_SIZE +
        compressed_length +
        SIGNATURE_SIZE
    )


# ============================================================
# PARSE PAYLOAD
# ============================================================

def parse_body(raw):

    if raw[:4] != MAGIC:
        raise ValueError("Invalid watermark magic.")

    pos = 4

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if pos >= len(raw):
        raise ValueError("Missing model length.")

    model_len = raw[pos]
    pos += 1

    if pos + model_len > len(raw):
        raise ValueError("Invalid model length.")

    model = raw[
        pos:pos + model_len
    ].decode("utf-8")

    pos += model_len

    # --------------------------------------------------------
    # Version
    # --------------------------------------------------------

    if pos >= len(raw):
        raise ValueError("Missing version length.")

    version_len = raw[pos]
    pos += 1

    if pos + version_len > len(raw):
        raise ValueError("Invalid version length.")

    version = raw[
        pos:pos + version_len
    ].decode("utf-8")

    pos += version_len

    # --------------------------------------------------------
    # UUID
    # --------------------------------------------------------

    if pos + 16 > len(raw):
        raise ValueError("Missing image ID.")

    image_id = raw[
        pos:pos + 16
    ].hex()

    pos += 16

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if pos + 8 > len(raw):
        raise ValueError("Missing timestamp.")

    timestamp = struct.unpack(
        ">Q",
        raw[pos:pos + 8]
    )[0]

    pos += 8

    # --------------------------------------------------------
    # Prompt length
    # --------------------------------------------------------

    if pos + 2 > len(raw):
        raise ValueError("Missing prompt length.")

    prompt_len = struct.unpack(
        ">H",
        raw[pos:pos + 2]
    )[0]

    pos += 2

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    if pos + prompt_len > len(raw):
        raise ValueError("Invalid prompt length.")

    compressed_prompt = raw[
        pos:pos + prompt_len
    ]

    pos += prompt_len

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
        "prompt": prompt
    }


# ============================================================
# PARSE PAYLOAD
# ============================================================

def parse_payload(data):

    # --------------------------------------------------------
    # Container
    #
    # magic              4
    # compressed length  2
    # compressed body    N
    # signature          64
    # --------------------------------------------------------

    if len(data) < HEADER_SIZE + SIGNATURE_SIZE:
        raise ValueError("Payload too small.")

    if data[:4] != MAGIC:
        raise ValueError("Invalid watermark magic.")

    compressed_length = struct.unpack(
        ">H",
        data[4:HEADER_SIZE]
    )[0]

    pos = HEADER_SIZE

    expected = pos + compressed_length + SIGNATURE_SIZE

    if expected > len(data):
        raise ValueError(
            "Invalid compressed body length."
        )

    # Der Container muss exakt passen. Zusaetzliche Bytes
    # wuerden auf einen falschen Kandidaten hindeuten
    # (falsche RS-Laenge) und duerfen nicht stillschweigend
    # verworfen werden.
    if expected != len(data):
        raise ValueError(
            "Unexpected trailing data after signature "
            f"({len(data) - expected} bytes)."
        )

    compressed_body = data[
        pos:pos + compressed_length
    ]

    signature = data[
        pos + compressed_length:
    ][:SIGNATURE_SIZE]

    # --------------------------------------------------------
    # Decompress body
    #
    # LZMADecompressor statt lzma.decompress: Letzteres
    # ignoriert angehaengte Bytes stillschweigend. Ein
    # unvollstaendig oder zu lang gelesener Stream soll
    # hier auffallen und nicht als gueltig durchgehen.
    # --------------------------------------------------------

    try:
        decompressor = lzma.LZMADecompressor(
            format=lzma.FORMAT_XZ
        )

        body = decompressor.decompress(
            compressed_body
        )

        if not decompressor.eof:
            raise ValueError(
                "Incomplete LZMA stream."
            )

        if decompressor.unused_data:
            raise ValueError(
                "Trailing data after LZMA stream."
            )

    except lzma.LZMAError as e:
        raise ValueError(
            f"Body decompression failed: {e}"
        )

    # --------------------------------------------------------
    # Innere Struktur
    # --------------------------------------------------------

    payload = parse_body(body)

    # Signiert wurde der unkomprimierte Body
    payload["signature"] = signature
    payload["body"] = body

    return payload


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

    max_bytes = TILE_BYTES

    print(
        f"      Tile size: {TILE_SIZE}x{TILE_SIZE}"
    )

    print(
        f"      Capacity: {max_bytes} bytes "
        f"(codeword)"
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
            # Reed-Solomon-Laenge
            #
            # WICHTIG: reedsolo haengt 32 Parity-Byte an
            # JEDEN 223-Byte-Block an. Ein Payload > 223
            # Byte braucht also mehr als 32 Parity-Byte.
            # Mit einem festen + RS_PARITY wurde ab der
            # zweiten Chunk-Grenze die falsche Anzahl Byte
            # aus dem Tile gelesen und jedes solche Payload
            # war nicht dekodierbar.
            # ------------------------------------------------

            encoded_length = rs_len(
                payload_length
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
                    (tx, ty, bytes(decoded))
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

    # ----------------------------------------------------
    # Parse
    #
    # Alle parsebaren Kandidaten sammeln, statt den ersten
    # zurueckzugeben. Ein Tile kann per Zufall RS-dekodierbar
    # sein, und ein frueher gefundener Kandidat kann eine
    # falsche Signatur haben, waehrend ein spaeterer Tile
    # den echten, korrekt signierten Payload enthaelt.
    # ----------------------------------------------------

    parsed = []

    for tx, ty, candidate in candidates:

        try:

            payload = parse_payload(
                candidate
            )

        except Exception as e:

            print(
                f"      Tile ({tx},{ty}) "
                f"invalid payload: {e}"
            )

            continue

        parsed.append(
            (tx, ty, payload)
        )

    if not parsed:

        error(
            "Watermark data was found, "
            "but could not be decoded."
        )

    # ----------------------------------------------------
    # Verify
    # ----------------------------------------------------

    print("[4/5] Verifying signature...")

    verified = []

    for tx, ty, payload in parsed:

        valid = verify_signature(
            payload
        )

        if not valid:
            print(
                f"      Tile ({tx},{ty}) "
                f"signature INVALID"
            )

        verified.append(
            (tx, ty, payload, valid)
        )

    # ----------------------------------------------------
    # Select
    #
    # Ein gueltig signierter Payload hat Vorrang. Erst wenn
    # kein einziger Kandidat die Signatur besteht, wird ein
    # parsebarer Kandidat gemeldet -- dann aber klar als
    # INVALID markiert.
    # ----------------------------------------------------

    selected = next(
        (entry for entry in verified if entry[3]),
        None,
    )

    if selected is None:
        selected = verified[0]

    tx, ty, payload, valid = selected

    # ----------------------------------------------------
    # Output
    # ----------------------------------------------------

    print("[5/5] Done.")
    print()

    print("================================")
    print(" WATERMARK FOUND")
    print("================================")

    print(
        f"Tile:        ({tx},{ty})"
    )

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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    decode_image(
        "watermarked.png"
    )