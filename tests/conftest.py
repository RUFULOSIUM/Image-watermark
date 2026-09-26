import struct
import zlib
import lzma

import numpy as np
import pytest
from PIL import Image

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


@pytest.fixture
def key_dir(tmp_path, monkeypatch):
    """Create an Ed25519 key pair in a temp dir and use it as the cwd."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    (tmp_path / "private_key.pem").write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    (tmp_path / "public_key.pem").write_bytes(
        public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    monkeypatch.chdir(tmp_path)

    return tmp_path, private_key


@pytest.fixture
def rgb_image(tmp_path):
    """512x512 RGB test image."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [120, 45, 200]
    path = tmp_path / "input.png"
    Image.fromarray(img).save(path)
    return str(path)


@pytest.fixture
def make_payload():
    """Build a signed payload with a given private key (deterministic)."""

    def _make(
        private_key,
        prompt="prompt",
        model="model",
        version="v1.0",
        image_id=b"\x00" * 16,
        timestamp=1700000000,
    ):
        model_bytes = model.encode("utf-8")
        version_bytes = version.encode("utf-8")
        compressed = zlib.compress(prompt.encode("utf-8"))

        body = (
            b"WM01"
            + bytes([len(model_bytes)]) + model_bytes
            + bytes([len(version_bytes)]) + version_bytes
            + image_id
            + struct.pack(">Q", timestamp)
            + struct.pack(">H", len(compressed))
            + compressed
        )

        signature = private_key.sign(body)

        compressed_body = lzma.compress(
            body, lzma.FORMAT_XZ, preset=9
        )

        return (
            b"WM01"
            + struct.pack(">H", len(compressed_body))
            + compressed_body
            + signature
        )

    return _make


@pytest.fixture
def make_body():
    """Build only the uncompressed, signed body (no LZMA container)."""

    def _make(
        prompt="prompt",
        model="model",
        version="v1.0",
        image_id=b"\x00" * 16,
        timestamp=1700000000,
    ):
        model_bytes = model.encode("utf-8")
        version_bytes = version.encode("utf-8")
        compressed = zlib.compress(prompt.encode("utf-8"))

        return (
            b"WM01"
            + bytes([len(model_bytes)]) + model_bytes
            + bytes([len(version_bytes)]) + version_bytes
            + image_id
            + struct.pack(">Q", timestamp)
            + struct.pack(">H", len(compressed))
            + compressed
        )

    return _make
