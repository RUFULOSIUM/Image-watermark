# Image Watermark
Invisible, cryptographically signed watermarks for AI-generated images.

This tool embeds metadata (model, version, prompt, timestamp) directly into the pixels of an image — without visible change. Every watermark is signed with an Ed25519 key pair, so the origin and integrity of the data can be verified. Thanks to Reed–Solomon encoding, the watermark survives compression, scaling, and other processing steps.

## How it works
The protection uses DCT-based steganography in multiple steps:

- **Create payload** – The metadata is packed into a binary payload (see below), signed with an Ed25519 private key, and error-corrected via Reed–Solomon (32 bytes parity).

- **Embed bits** – The image is split into 512×512-pixel tiles. In each tile the luminance channel (YCrCb) is split into 8×8 blocks and one of two DCT coefficients (`(3,4)` / `(4,3)`) is adjusted per bit. A bit is 1 when coefficient A is larger than B, otherwise 0.

- **Redundancy** – Every tile carries the same watermark. Damaged tiles or tiles without a watermark are automatically discarded during decoding.

### Payload format
| Field | Size |
| --- | --- |
| Magic `WM01` | 4 bytes |
| Model name | 1 byte length + content |
| Version | 1 byte length + content |
| Image ID (UUID) | 16 bytes |
| Timestamp (Unix, big-endian) | 8 bytes |
| Prompt (zlib-compressed) | 2 bytes length + content |
| Ed25519 signature | 64 bytes |

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
    version="v0.1",
)

decode.decode_image("encodet/test.png")

```
Alternatively, run the example script:

```bash
uv run main.py

```
### Capacity and limitations
- Images must be at least **512×512 pixels**.

- One tile holds **4096 bits (512 bytes)** after Reed–Solomon encoding, i.e. roughly ~450 bytes of usable payload per tile.

- Fewer tiles (smaller images) means less redundancy; larger images carry the watermark redundantly in every tile.

- Output is saved as PNG to avoid losses from lossy compression.

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
    └── decode.py            # Extract and verify watermark

```