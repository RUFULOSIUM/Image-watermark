"""The v0.1 / v0.2 payload layout, kept for backwards compatibility.

Before v0.3 the payload body was a fixed sequence with the magic baked
into the body itself:

.. code-block:: text

    WM01
    model      1 byte length + utf-8
    version    1 byte length + utf-8
    image id   16 raw bytes
    timestamp  uint64 big endian
    prompt     2 byte length + zlib compressed utf-8

Since v0.3 the layout comes from a :class:`PayloadSchema` instead. This
module exists so that images written by older versions still decode with
identical semantics, and so that the legacy ``encode``/``decode``
wrappers keep producing the exact byte format their callers expect. It
must never be used for new payloads.
"""

from __future__ import annotations

import struct
import zlib

from typing import Any

from image_watermark.errors import PayloadError


LEGACY_MAGIC = b"WM01"

#: The schema id reported for legacy payloads.
LEGACY_SCHEMA_ID = "legacy-wm01"

#: Field names of the legacy layout, in wire order.
LEGACY_FIELDS = (
    "model",
    "version",
    "image_id",
    "timestamp",
    "prompt",
)

#: Single byte length prefix on model and version.
LEGACY_MAX_SHORT = 255

#: Two byte length prefix on the compressed prompt.
LEGACY_MAX_PROMPT = 0xFFFF


# ============================================================
# BUILD
# ============================================================

def build_legacy_body(
    model: str = "",
    version: str = "",
    image_id: bytes | None = None,
    timestamp: int | None = None,
    prompt: str = "",
) -> bytes:
    """Build a v0.2 body.

    Used by the legacy ``encode`` wrapper. New code should build a
    :class:`~image_watermark.payload.Payload` from a
    :class:`~image_watermark.schema.PayloadSchema` instead.
    """

    import time

    if image_id is None:
        import uuid

        image_id = uuid.uuid4().bytes

    if len(image_id) != 16:
        raise PayloadError(
            f"Image id must be 16 bytes, got {len(image_id)}"
        )

    if timestamp is None:
        timestamp = int(time.time())

    model_bytes = model.encode("utf-8")
    version_bytes = version.encode("utf-8")

    if len(model_bytes) > LEGACY_MAX_SHORT:
        raise PayloadError(
            f"Model name is too long "
            f"({len(model_bytes)} > {LEGACY_MAX_SHORT} bytes)"
        )

    if len(version_bytes) > LEGACY_MAX_SHORT:
        raise PayloadError(
            f"Model version is too long "
            f"({len(version_bytes)} > {LEGACY_MAX_SHORT} bytes)"
        )

    compressed_prompt = zlib.compress(prompt.encode("utf-8"))

    if len(compressed_prompt) > LEGACY_MAX_PROMPT:
        raise PayloadError(
            f"Prompt is too large "
            f"({len(compressed_prompt)} > {LEGACY_MAX_PROMPT} bytes "
            f"after compression)"
        )

    return (
        LEGACY_MAGIC
        + bytes([len(model_bytes)])
        + model_bytes
        + bytes([len(version_bytes)])
        + version_bytes
        + bytes(image_id)
        + struct.pack(">Q", int(timestamp))
        + struct.pack(">H", len(compressed_prompt))
        + compressed_prompt
    )


# ============================================================
# PARSE
# ============================================================

def parse_legacy_body(raw: bytes) -> dict[str, Any]:
    """Decode a v0.2 body into the field dict of a legacy payload.

    The v0.2 decoder returned ``image_id`` as hex and ``timestamp`` as an
    int; this reproduces that exactly so nothing observable changes for
    existing callers.
    """

    if raw[:4] != LEGACY_MAGIC:
        raise PayloadError("Invalid watermark magic.")

    pos = 4

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    if pos >= len(raw):
        raise PayloadError("Missing model length.")

    model_len = raw[pos]
    pos += 1

    if pos + model_len > len(raw):
        raise PayloadError("Invalid model length.")

    try:
        model = raw[pos:pos + model_len].decode("utf-8")

    except UnicodeDecodeError as e:
        raise PayloadError(
            f"Model is not valid utf-8: {e}"
        ) from e

    pos += model_len

    # --------------------------------------------------------
    # version
    # --------------------------------------------------------

    if pos >= len(raw):
        raise PayloadError("Missing version length.")

    version_len = raw[pos]
    pos += 1

    if pos + version_len > len(raw):
        raise PayloadError("Invalid version length.")

    try:
        version = raw[pos:pos + version_len].decode("utf-8")

    except UnicodeDecodeError as e:
        raise PayloadError(
            f"Version is not valid utf-8: {e}"
        ) from e

    pos += version_len

    # --------------------------------------------------------
    # image id
    # --------------------------------------------------------

    if pos + 16 > len(raw):
        raise PayloadError("Missing image ID.")

    image_id = raw[pos:pos + 16].hex()
    pos += 16

    # --------------------------------------------------------
    # timestamp
    # --------------------------------------------------------

    if pos + 8 > len(raw):
        raise PayloadError("Missing timestamp.")

    timestamp = struct.unpack(">Q", raw[pos:pos + 8])[0]
    pos += 8

    # --------------------------------------------------------
    # prompt
    # --------------------------------------------------------

    if pos + 2 > len(raw):
        raise PayloadError("Missing prompt length.")

    prompt_len = struct.unpack(">H", raw[pos:pos + 2])[0]
    pos += 2

    if pos + prompt_len > len(raw):
        raise PayloadError("Invalid prompt length.")

    compressed_prompt = raw[pos:pos + prompt_len]
    pos += prompt_len

    if pos != len(raw):
        raise PayloadError(
            f"Legacy body has {len(raw) - pos} trailing byte(s)."
        )

    try:
        prompt = zlib.decompress(compressed_prompt).decode(
            "utf-8"
        )

    except Exception as e:
        raise PayloadError(
            f"Prompt decompression failed: {e}"
        ) from e

    return {
        "model": model,
        "version": version,
        "image_id": image_id,
        "timestamp": timestamp,
        "prompt": prompt,
    }
