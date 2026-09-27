"""Ed25519 key handling.

v0.2 hard-coded ``private_key.pem`` / ``public_key.pem`` in the current
working directory and called ``sys.exit`` on every problem. Here the
paths are parameters, failures are exceptions, and the key type is
verified on load instead of blowing up later during ``sign()``.
"""

from __future__ import annotations

import os

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from image_watermark.errors import KeyFileError


DEFAULT_PRIVATE_KEY = "private_key.pem"
DEFAULT_PUBLIC_KEY = "public_key.pem"


def generate_key_pair(
    private_path: str | os.PathLike[str] = DEFAULT_PRIVATE_KEY,
    public_path: str | os.PathLike[str] = DEFAULT_PUBLIC_KEY,
) -> Ed25519PrivateKey:
    """Create a fresh key pair and write both PEM files."""

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    with open(private_path, "wb") as handle:
        handle.write(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )

    with open(public_path, "wb") as handle:
        handle.write(
            public_key.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    return private_key


def load_private_key(
    path: str | os.PathLike[str] = DEFAULT_PRIVATE_KEY,
) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM file."""

    try:
        with open(path, "rb") as handle:
            data = handle.read()

    except FileNotFoundError as e:
        raise KeyFileError(
            f"Private key not found: {path}"
        ) from e

    except OSError as e:
        raise KeyFileError(
            f"Could not read private key {path}: {e}"
        ) from e

    try:
        key = serialization.load_pem_private_key(
            data, password=None
        )

    except Exception as e:
        raise KeyFileError(
            f"{path} is not a readable private key: {e}"
        ) from e

    if not isinstance(key, Ed25519PrivateKey):
        raise KeyFileError(
            f"{path} is a "
            f"{type(key).__name__}, but this library only signs "
            f"with Ed25519."
        )

    return key


def load_public_key(
    path: str | os.PathLike[str] = DEFAULT_PUBLIC_KEY,
) -> Ed25519PublicKey:
    """Load an Ed25519 public key from a PEM file."""

    try:
        with open(path, "rb") as handle:
            data = handle.read()

    except FileNotFoundError as e:
        raise KeyFileError(
            f"Public key not found: {path}"
        ) from e

    except OSError as e:
        raise KeyFileError(
            f"Could not read public key {path}: {e}"
        ) from e

    try:
        key = serialization.load_pem_public_key(data)

    except Exception as e:
        raise KeyFileError(
            f"{path} is not a readable public key: {e}"
        ) from e

    if not isinstance(key, Ed25519PublicKey):
        raise KeyFileError(
            f"{path} is a {type(key).__name__}, but this library "
            f"only verifies Ed25519 signatures."
        )

    return key


def derive_public_key(
    private_key: Ed25519PrivateKey,
) -> Ed25519PublicKey:
    return private_key.public_key()


def ensure_key_pair(
    private_path: str | os.PathLike[str] = DEFAULT_PRIVATE_KEY,
    public_path: str | os.PathLike[str] = DEFAULT_PUBLIC_KEY,
) -> Ed25519PrivateKey:
    """Return the private key, generating a pair if none exists yet."""

    if not os.path.exists(private_path):
        return generate_key_pair(private_path, public_path)

    return load_private_key(private_path)
