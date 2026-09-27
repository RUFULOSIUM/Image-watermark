# Image Watermark
Invisible, cryptographically signed watermarks for AI-generated images.

This tool embeds metadata (model, version, image id, timestamp, prompt) directly
into the pixels of an image — without visible change. Every watermark is signed
with an Ed25519 key pair, so the origin and integrity of the data can be
verified. Thanks to Reed–Solomon encoding, the watermark survives compression,
scaling, and other processing steps.

The payload layout is **schema-driven**: what actually gets embedded is defined
by a JSON schema, not by hard-coded fields. The schema bundled with the package
is drop-in compatible with the v0.2 fields.

## How it works

- **Create payload** – The field values are serialised into a binary body
  according to the active schema, signed with an Ed25519 private key, and
  error-corrected via Reed–Solomon (32 parity bytes per chunk).

- **Embed bits** – The image is split into 512×512-pixel tiles. In each tile the
  luminance channel (YCrCb) is split into 8×8 blocks and one of two DCT
  coefficients (`(3,4)` / `(4,3)`) is adjusted per bit. A bit is 1 when
  coefficient A is larger than B, otherwise 0.

- **Redundancy** – Every tile carries the same watermark. Damaged tiles or tiles
  without a watermark are automatically discarded during decoding.

### WM02 container (v0.3)

| Field | Size |
| --- | --- |
| Magic `WM02` | 4 bytes |
| Version | 2 bytes |
| Compression | 2 bytes |
| Schema id length | 2 bytes |
| Flags | 2 bytes |
| Body length (compressed) | 4 bytes |
| Raw body length | 4 bytes |
| Schema id | variable (≤ 64 bytes) |
| Body | variable |
| Ed25519 signature | 64 bytes, **only if the signed flag is set** |

The signature always covers the **uncompressed** raw body, so it does not depend
on the compression settings. The decoder reads the fixed 20-byte prefix plus the
schema id, then knows the exact container length without decompression. That is
also what makes the header sniffable from the first 84 bytes of a tile.

### Reed–Solomon chunking

`reedsolo` does not build one big codeword. It splits the message into chunks of
`nsize - nsym` = 223 bytes and appends 32 parity bytes to **every** chunk. The
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

## CLI

```bash
# one-time key pair
uv run image-watermark init-keys

# embed, using the bundled default schema
uv run image-watermark encode in.png out.png \
    -f model=flux -f version=1.4 -f prompt="a lighthouse"

# read it back
uv run image-watermark decode out.png

# inspect the schema that would be used
uv run image-watermark schema
uv run image-watermark schema --format json

# show resolved settings and capacity
uv run image-watermark info
```

`-f/--field NAME=VALUE` is repeatable. Field names come from the active schema,
and every field with a `default` is filled in automatically — so only the
optional fields you care about have to be passed. Unknown field names are
rejected.

Pass `--schema FILE` to any of `encode`, `decode` and `info` to use your own
schema instead of the bundled one:

```bash
uv run image-watermark encode in.png out.png --schema schemas/provenance.json \
    -f artist="Ada Lovelace" -f camera="Hasselblad 500C/M" -f focal_mm=80
```

## Python API

```python
from image_watermark import (
    WatermarkDecoder,
    WatermarkEncoder,
    default_settings,
)

settings = default_settings()

WatermarkEncoder(settings=settings).encode(
    "in.png",
    "out.png",
    {"model": "flux", "version": "1.4", "prompt": "a lighthouse"},
    progress=None,
)

result = WatermarkDecoder(settings=settings).decode("out.png", progress=None)

print(result.schema_id, result.valid)
print(result.payload.get("prompt"))
```

`default_settings()` accepts the same knobs the CLI exposes, e.g.
`default_settings(sign=False)` for an unsigned payload or
`default_settings(schema=my_schema)` for a custom schema.

## Custom schemas

A schema is a JSON document describing an ordered list of fields:

```json
{
  "id": "provenance",
  "version": 1,
  "description": "Where the image came from.",
  "fields": [
    {"name": "artist", "type": "string", "length": 128, "required": true},
    {"name": "created", "type": "uint64", "required": false, "default": "now"},
    {"name": "camera", "type": "string", "length": 64, "required": false},
    {"name": "focal_mm", "type": "uint16", "required": false},
    {"name": "gps", "type": "bytes", "length": 16, "display": "hex",
     "required": false},
    {"name": "note", "type": "string", "compression": "zlib", "required": false}
  ]
}
```

The allowed field keys are `name`, `type`, `length`, `compression`, `default`,
`required`, `description` and `display`; the allowed schema keys are `id`/`name`,
`version`, `description` and `fields`. Anything else is rejected with an
explicit error instead of being ignored.

`type` is one of `string`, `bytes`, `bool`, `uint8`, `uint16`, `uint32` or
`uint64`. Variable-length fields (`string`, `bytes`) get a 2-byte length prefix
by default; `compression: "zlib"` compresses just that field, which pays off for
long and repetitive values. A field with a `default` is filled in automatically,
so it does not have to be passed. Fields that are not `required` are written as
their empty value when omitted, which keeps the layout stable.

Validate a schema before using it:

```bash
uv run image-watermark schema --validate schemas/provenance.json
```

## Capacity and limitations
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

- With the default schema and LZMA that leaves **355 bytes** for the
  uncompressed body. `image-watermark info` prints the exact numbers for the
  active settings.

- LZMA carries a fixed container overhead of roughly 60 bytes. For short prompts it
  can therefore be *larger* than compressing the prompt alone — measure before
  assuming a win.

- Fewer tiles (smaller images) means less redundancy; larger images carry the
  watermark redundantly in every tile. Use `--no-redundant` to write only the
  first tile, or `--max-tiles N` to cap it.

- Output is saved as PNG to avoid losses from lossy compression.

## Changelog

### v0.3
- **Added** a schema-driven payload. What is embedded is now defined by a JSON
  schema instead of hard-coded fields, so new metadata can be added without
  touching the container code. The bundled `default` schema keeps the v0.2
  fields (`model`, `version`, `image_id`, `timestamp`, `prompt`).
- **Added** a `WM02` container that carries a version, a compression tag, a
  schema id and a signed flag, with 4-byte length fields.
- **Added** a `image-watermark` CLI with `init-keys`, `encode`, `decode`,
  `schema` and `info`. Payload fields are passed generically via
  `-f/--field NAME=VALUE`.
- **Added** `Payload`, `WatermarkEncoder`, `WatermarkDecoder`,
  `WatermarkSettings` and a `SchemaRegistry`, and moved the default schema into
  the package so it is loaded via `importlib.resources`.
- **Fixed** optional payload fields being skipped instead of written, which
  corrupted the body layout. Every field now occupies its slot, so a schema
  change can never shift the following fields.
- **Fixed** the tile rotation offset, which swapped its axes and wrote tiles in
  the wrong order when an `embed_offset` was set.
- **Fixed** the reported capacity, which still subtracted the legacy header size
  from the WM02 budget.
- **Kept** v0.2 compatibility: images written by the old `encode.encode_image`
  wrapper still decode, and `WM01` is still recognised.

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

- An unsigned payload (`--no-sign`) can be written and read back, but
  `result.valid` is `False`: there is no way to tell who produced it.

## License
This project is licensed under the [Image Watermark License](LICENSE).

## Project structure
```tree
├── main.py                  # Example: encode + decode an image (v0.2 API)
├── images/                  # Input images
├── encodet/                 # Output images (watermarked)
├── schemas/                 # Custom schemas, e.g. provenance.json
└── src/image_watermark/
    ├── cli.py               # image-watermark command line
    ├── encoder.py           # OOP encoder (WM02)
    ├── decoder.py           # OOP decoder (WM02 + WM01) and registry
    ├── schema.py            # FieldSpec, PayloadSchema, schema loading
    ├── payload.py           # Field serialisation / deserialisation
    ├── container.py         # WM01/WM02 framing
    ├── legacy.py            # WM01 body builder and parsers
    ├── settings.py          # WatermarkSettings
    ├── keys.py              # Ed25519 key handling
    ├── dct.py               # DCT steganography
    ├── encode.py            # v0.2 procedural API (WM01)
    ├── decode.py            # v0.2 procedural API
    └── rs.py                # Reed-Solomon codeword geometry
```
