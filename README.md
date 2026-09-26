# Image Watermark
Invisible, cryptographically signed watermarks for AI-generated images.

This tool embeds metadata (model, version, prompt, timestamp) directly into the pixels of an image — without visible change. Every watermark is signed with an Ed25519 key pair, so the origin and integrity of the data can be verified. Thanks to Reed–Solomon encoding, the watermark survives compression, scaling, and other processing steps.

## How it works
The protection uses DCT-based steganography in multiple steps:

- **Create payload** – The metadata is packed into a binary payload (see below), signed with an Ed25519 private key, and error-corrected via Reed–Solomon (32 bytes parity per chunk).

- **Embed bits** – The image is split into 512×512-pixel tiles. In each tile the luminance channel (YCrCb) is split into 8×8 blocks and one of two DCT coefficients (`(3,4)` / `(4,3)`) is adjusted per bit. A bit is 1 when coefficient A is larger than B, otherwise 0.

- **Redundancy** – Every tile carries the same watermark. Damaged tiles or tiles without a watermark are automatically discarded during decoding.

### Payload format

The container is Reed–Solomon encoded as a whole. Only the outer container is
readable without decompression:

| Field | Size |
| --- | --- |
| Magic `WM01` | 4 bytes |
| Compressed body length | 2 bytes |
| Body (LZMA/XZ compressed) | variable |
| Ed25519 signature | 64 bytes |

The body itself is compressed with LZMA and signed in its **uncompressed** form,
so the signature does not depend on the compression settings:

| Field | Size |
| --- | --- |
| Magic `WM01` | 4 bytes |
| Model name | 1 byte length + content |
| Version | 1 byte length + content |
| Image ID (UUID) | 16 bytes |
| Timestamp (Unix, big-endian) | 8 bytes |
| Prompt (zlib-compressed) | 2 bytes length + content |

The compressed body length has to stay in the clear: the decoder must know how
many bytes to pull out of the image before it can decompress anything, and
Reed–Solomon needs the exact codeword length up front.

### Reed–Solomon chunking

`reedsolo` does not build one big codeword. It splits the message into chunks
of `nsize - nsym` = 223 bytes and appends 32 parity bytes to **every** chunk. The
encoded length is therefore

```text
rs_len(n) = n + ceil(n / 223) * 32
```

This matters because the decoder has to read exactly `rs_len(n)` bytes out of a
tile — no more, no less. A 300 byte payload is two chunks and occupies
300 + 64 = 364 bytes, not 332.

`src/image_watermark/rs.py` is the single source of truth for this arithmetic;
the encoder asserts that `len(RSCodec(32).encode(data)) == rs_len(len(data))` so
a change in `reedsolo` cannot silently break the format.

## Installation
Requirements: Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

```
## Usage
The encoder automatically generates an Ed25519 key pair (`private_key.pem`, `public_key.pem`) on first run.

### Watermark an image
```python
from image_watermark import encode, decode

encode.encode_image(
    input_path="images/test.png",
    output_path="encodet/test.png",
    prompt="Reflections and refractions from glass objects",
    model="myself",
    version="v0.2",
)

decode.decode_image("encodet/test.png")

```
Alternatively, run the example script:

```bash
uv run main.py

```
### Capacity and limitations
- Images must be at least **512×512 pixels**.

- One tile holds **4096 bits (512 bytes)** — that is the size of the whole
  Reed–Solomon codeword, not of the payload. Because 32 parity bytes are added
  per 223-byte chunk, a payload of `n` bytes needs `rs_len(n)` bytes of codeword.
  The largest payload that still fits a tile is therefore **446 bytes**
  (`446 + 2 × 32 = 510`), not `512 − 32 = 480`.

  | Payload | Chunks | Codeword | Fits (≤ 512) |
  | --- | --- | --- | --- |
  | 223 | 1 | 255 | yes |
  | 380 | 2 | 444 | yes |
  | 446 | 2 | 510 | yes |
  | 447 | 3 | 543 | no |

- The container overhead is 6 bytes of header plus the 64-byte signature, which
  leaves **376 bytes** for the LZMA-compressed body.

- LZMA carries a fixed container overhead of roughly 60 bytes. For short prompts it
  can therefore be *larger* than compressing the prompt alone — measure before
  assuming a win.

- Fewer tiles (smaller images) means less redundancy; larger images carry the watermark redundantly in every tile.

- Output is saved as PNG to avoid losses from lossy compression.

## Changelog

### v0.2
- **Changed** the payload container. The body is now LZMA/XZ compressed and the
  compressed length is stored in the clear, so the decoder knows the exact
  Reed–Solomon codeword length before it touches the pixel data. The signature
  still covers the **uncompressed** body and is therefore independent of the
  compression settings. Longer prompts now fit that were previously impossible.
- **Fixed** the Reed–Solomon codeword length. The decoder assumed a single
  223-byte chunk (`n + 32`), so **every payload larger than 223 bytes was
  undecodable** — it aborted with `No valid watermark found`. The length is now
  computed with `rs_len(n) = n + ceil(n / 223) * 32` for both encoder and decoder.
- **Fixed** the reported tile capacity, which claimed 480 usable bytes. The real
  limit is 446 payload bytes (376 bytes of LZMA-compressed body).
- **Fixed** `benchmark.py`, which called the removed `encode.create_payload` and
  would raise `AttributeError` on the first run.
- **Changed** the decoder now prefers a cryptographically valid payload. A tile
  that decodes by coincidence, or a forged payload, is no longer reported as a
  found watermark while a correctly signed tile exists further along.
- **Hardened** `parse_payload`: rejects trailing data after the signature and
  verifies the LZMA stream is fully consumed (`eof`), instead of silently
  ignoring it.
- **Fixed** the key loaders: they now reject a `private_key.pem` /
  `public_key.pem` that is not an Ed25519 key with a clear message instead of
  failing later with an `AttributeError`.
- Added regression tests that pin `rs_len` against real `reedsolo` output across
  the chunk boundaries, cover the 446/447 byte capacity limit, multi-chunk
  end-to-end roundtrips, and the signature-preference behaviour.
- Removed leftover merge-conflict markers from `.gitignore`.

## Security notes
- The `private_key.pem` must **never** be published or committed (already in `.gitignore`).

- The public key is required to verify signatures — and to decode images from other sources.

- The signature proves the integrity of the embedded payload, not that the pixels themselves were unmodified.

## License
This project is licensed under the [Image Watermark License](LICENSE).

## Project structure
```tree

├── main.py                  # Example: encode + decode an image
├── images/                  # Input images
├── encodet/                 # Output images (watermarked)
└── src/image_watermark/
    ├── encode.py            # Embed watermark
    ├── decode.py            # Extract and verify watermark
    └── rs.py                # Reed-Solomon codeword geometry

```