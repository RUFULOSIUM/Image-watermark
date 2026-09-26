import argparse
import contextlib
import csv
import io
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
from reedsolo import RSCodec

from image_watermark import decode, encode

PICSUM_URL = "https://picsum.photos/{width}/{height}"

USER_AGENT = {"User-Agent": "image-watermark-benchmark"}

SIZES = [
    (512, 512),
    (800, 600),
    (1024, 768),
    (1024, 1024),
    (1920, 1080),
    (2560, 1440),
]

TEST_PAYLOADS = [
    {
        "prompt": "A calm lake at sunrise with a lone fisherman.",
        "model": "test-model-a",
        "version": "v0.2.0",
    },
    {
        "prompt": (
            "Bustling city street, neon signs glowing in the rain, "
            "people with umbrellas."
        ),
        "model": "neon-model",
        "version": "v2.3",
    },
    {
        "prompt": "",
        "model": "minimal",
        "version": "",
    },
]

CSV_FIELDS = [
    "case",
    "width",
    "height",
    "tiles",
    "payload",
    "prompt_chars",
    "payload_bytes",
    "encoded_bytes",
    "watermark_bits",
    "encode_s",
    "decode_s",
    "decode_ok",
    "signature_valid",
    "mse",
    "psnr",
    "mae",
    "max_diff",
    "changed_pixels",
    "original_bytes",
    "signed_bytes",
    "size_change_pct",
]


def download_image(path, width, height):
    url = PICSUM_URL.format(width=width, height=height)
    request = urllib.request.Request(url, headers=USER_AGENT)

    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()

    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.save(path, format="PNG")


def compute_metrics(original, signed):
    a = original.astype(np.float64)
    b = signed.astype(np.float64)

    diff = a - b
    mse = float(np.mean(diff ** 2))
    psnr = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0.0 else float("inf")

    return {
        "mse": mse,
        "psnr": psnr,
        "mae": float(np.mean(np.abs(diff))),
        "max_diff": float(np.max(np.abs(diff))),
        "changed_pixels": float(np.mean(np.any(a != b, axis=2))),
    }


def decode_and_verify(path):
    image = np.array(Image.open(path).convert("RGB"))
    height, width, _ = image.shape

    fallback = None

    try:
        rs = RSCodec(decode.RS_PARITY)

        for ty in range(height // decode.TILE_SIZE):
            for tx in range(width // decode.TILE_SIZE):
                x = tx * decode.TILE_SIZE
                y = ty * decode.TILE_SIZE
                tile = image[y:y + decode.TILE_SIZE, x:x + decode.TILE_SIZE]

                payload_length = decode.get_payload_length(tile)

                if payload_length is None:
                    continue

                encoded_length = decode.rs_len(payload_length)
                raw = decode.decode_tile(tile, encoded_length)
                decoded = bytes(rs.decode(raw)[0])
                payload = decode.parse_payload(decoded)

                valid = decode.verify_signature(payload)

                # Ein gueltig signierter Payload hat Vorrang.
                if valid:
                    return True, True

                if fallback is None:
                    fallback = (True, False)
    except Exception:
        pass

    return fallback if fallback else (False, False)


def run_case(case_index, output_root, width, height, payload):
    images_dir = os.path.join(output_root, "images")
    encoded_dir = os.path.join(output_root, "encodet")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(encoded_dir, exist_ok=True)

    original_path = os.path.join(images_dir, f"{case_index}.png")
    output_path = os.path.join(encoded_dir, f"{case_index}.png")

    download_image(original_path, width, height)

    original = np.array(Image.open(original_path).convert("RGB"))
    original_bytes = os.path.getsize(original_path)

    computed_payload = encode.create_compressed_payload(
        payload["prompt"],
        payload["model"],
        payload["version"],
    )
    encoded = encode.reed_solomon_encode(computed_payload)
    bits = encode.bytes_to_bits(encoded)

    assert computed_payload is not None
    assert encoded is not None

    encode_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        encode.encode_image(
            input_path=original_path,
            output_path=output_path,
            prompt=payload["prompt"],
            model=payload["model"],
            version=payload["version"],
        )
    encode_s = time.perf_counter() - encode_start

    signed = np.array(Image.open(output_path).convert("RGB"))
    signed_bytes = os.path.getsize(output_path)

    decode_start = time.perf_counter()
    decode_ok, signature_valid = decode_and_verify(output_path)
    decode_s = time.perf_counter() - decode_start

    metrics = compute_metrics(original, signed)

    height_px, width_px, _ = original.shape
    tiles = (width_px // encode.TILE_SIZE) * (
        height_px // encode.TILE_SIZE
    )

    return {
        "case": case_index,
        "width": width_px,
        "height": height_px,
        "tiles": tiles,
        "payload": payload["model"],
        "prompt_chars": len(payload["prompt"]),
        "payload_bytes": len(computed_payload),
        "encoded_bytes": len(encoded),
        "watermark_bits": len(bits),
        "encode_s": encode_s,
        "decode_s": decode_s,
        "decode_ok": decode_ok,
        "signature_valid": signature_valid,
        "mse": metrics["mse"],
        "psnr": metrics["psnr"],
        "mae": metrics["mae"],
        "max_diff": metrics["max_diff"],
        "changed_pixels": metrics["changed_pixels"],
        "original_bytes": original_bytes,
        "signed_bytes": signed_bytes,
        "size_change_pct": (
            (signed_bytes - original_bytes) / original_bytes * 100.0
        ),
    }


def mean(values):
    return statistics.mean(values) if values else 0.0


def print_summary(rows, total_runs):
    print()
    print("=" * 60)
    print(" BENCHMARK SUMMARY")
    print("=" * 60)
    print(f" Runs:             {len(rows)}")
    print(f" Skipped:          {total_runs - len(rows)}")

    decode_ok = sum(1 for r in rows if r["decode_ok"])
    valid = sum(1 for r in rows if r["signature_valid"])
    print(f" Decode success:   {decode_ok}/{len(rows)}")
    print(f" Signature valid:  {valid}/{len(rows)}")
    print(f" Total encode:     {sum(r['encode_s'] for r in rows):.2f}s")
    print(f" Total decode:     {sum(r['decode_s'] for r in rows):.2f}s")
    print(f" Mean encode:      {mean([r['encode_s'] for r in rows]):.3f}s")
    print(f" Mean decode:      {mean([r['decode_s'] for r in rows]):.3f}s")
    print()
    print(f" {'metric':<18}{'min':>12}{'mean':>12}{'median':>12}{'max':>12}")
    print("-" * 66)

    for metric in ("psnr", "mse", "mae", "max_diff", "changed_pixels",
                   "size_change_pct"):
        values = [r[metric] for r in rows if r[metric] is not None]
        if not values:
            continue
        print(
            f" {metric:<18}"
            f"{min(values):>12.4f}"
            f"{mean(values):>12.4f}"
            f"{statistics.median(values):>12.4f}"
            f"{max(values):>12.4f}"
        )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the image watermark on random internet images."
    )
    parser.add_argument(
        "--images",
        type=int,
        default=10,
        help="number of random images to download and test (default: 10)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="",
        help="comma-separated image sizes WxH to choose from for each image "
             "(default: all builtin sizes)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="benchmark_results.csv",
        help="output CSV dataset path (default: benchmark_results.csv)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="benchmark",
        help="directory for downloaded and encoded images (default: benchmark)",
    )
    args = parser.parse_args()

    if args.sizes:
        sizes = []
        for item in args.sizes.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                width, height = item.lower().split("x")
                sizes.append((int(width), int(height)))
            except ValueError:
                parser.error(f"invalid size: {item!r} (expected WxH)")
        if not sizes:
            sizes = SIZES
    else:
        sizes = SIZES

    os.makedirs(args.workdir, exist_ok=True)

    rows = []
    total = args.images * len(TEST_PAYLOADS)
    case_index = 0

    for image_index in range(args.images):
        width, height = random.choice(sizes)

        for payload in TEST_PAYLOADS:
            total_run = image_index * len(TEST_PAYLOADS) + \
                TEST_PAYLOADS.index(payload) + 1
            case_index += 1

            print(
                f"[{total_run}/{total}] "
                f"image {image_index + 1}/{args.images} "
                f"{width}x{height} payload='{payload['model']}' ...",
                end="",
            )

            try:
                row = run_case(
                    case_index,
                    args.workdir,
                    width,
                    height,
                    payload,
                )
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f" download failed: {e}")
                continue
            except Exception as e:
                print(f" failed: {e}")
                continue

            rows.append(row)
            print(
                f" ok psnr={row['psnr']:.1f}dB "
                f"encode={row['encode_s']:.2f}s "
                f"decode={row['decode_s']:.2f}s"
            )

    if not rows:
        print("No benchmark results produced.", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows, total)
    print(f"Dataset written to: {args.out}")


if __name__ == "__main__":
    main()