import io
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

from image_watermark import decode, encode

PICSUM_URL = "https://picsum.photos/{width}/{height}"

USER_AGENT = {"User-Agent": "image-watermark-test"}

IMAGE_SIZES = [
    (512, 512),
    (1024, 768),
    (800, 600),
]

TEST_PAYLOADS = [
    {
        "prompt": "A calm lake at sunrise with a lone fisherman.",
        "model": "test-model-a",
        "version": "v0.1.0",
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


def download_random_image(path, width, height):
    url = PICSUM_URL.format(width=width, height=height)
    request = urllib.request.Request(url, headers=USER_AGENT)

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        pytest.skip(f"Could not download random image: {e}")

    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.save(path, format="PNG")
    return path


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


def print_row(name, original_path, output_path, metrics):
    print(f"{name}")
    print(f"  original: {original_path}")
    print(f"  signed:   {output_path}")
    print(f"  MSE:             {metrics['mse']:.6f}")
    print(f"  PSNR:            {metrics['psnr']:.2f} dB")
    print(f"  MAE:             {metrics['mae']:.6f}")
    print(f"  max |diff|:      {metrics['max_diff']:.2f}")
    print(f"  changed pixels:  {metrics['changed_pixels'] * 100:.2f}%")


@pytest.mark.parametrize(
    "size",
    IMAGE_SIZES,
    ids=lambda size: f"{size[0]}x{size[1]}",
)
def test_watermark_does_not_change_image_much(tmp_path, size):
    width, height = size
    images_dir = tmp_path / "images"
    output_dir = tmp_path / "encodet"
    images_dir.mkdir()
    output_dir.mkdir()

    original_path = download_random_image(
        str(images_dir / "original.png"),
        width,
        height,
    )

    original = np.array(Image.open(original_path).convert("RGB"))

    for payload in TEST_PAYLOADS:
        name = f"{width}x{height} - {payload['model'] or '(empty)'}"
        output_path = str(output_dir / f"{payload['model'] or 'empty'}.png")

        encode.encode_image(
            input_path=original_path,
            output_path=output_path,
            prompt=payload["prompt"],
            model=payload["model"],
            version=payload["version"],
        )

        signed = np.array(Image.open(output_path).convert("RGB"))
        metrics = compute_metrics(original, signed)

        print_row(name, original_path, output_path, metrics)

        assert metrics["psnr"] > 20.0, f"PSNR too low for {name}"
        assert metrics["mse"] < 100.0, f"MSE too high for {name}"
        assert metrics["changed_pixels"] < 0.5, (
            f"Too many pixels changed for {name}"
        )