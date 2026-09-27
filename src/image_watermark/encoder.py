"""The watermark encoder.

``WatermarkEncoder`` owns a :class:`WatermarkSettings` object and exposes
the whole pipeline: payload -> body -> compressed container ->
Reed-Solomon -> DCT embedding. Every step is a separate method, so a
caller who needs a different order can compose them itself.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from reedsolo import RSCodec

from image_watermark import container as cont
from image_watermark import dct
from image_watermark.errors import (
    CapacityError,
    ConfigError,
    ImageError,
    PayloadError,
)
from image_watermark.keys import ensure_key_pair, load_private_key
from image_watermark.payload import Payload
from image_watermark.schema import PayloadSchema
from image_watermark.settings import WatermarkSettings


# ============================================================
# RESULT
# ============================================================

@dataclass
class EncodeResult:
    """Everything the encoder did, for inspection and for the CLI."""

    output_path: str
    input_path: str
    payload: Payload
    body: bytes
    container_bytes: int
    codeword_bytes: int
    bit_count: int
    tiles_embedded: int
    tiles_available: int
    settings: WatermarkSettings

    def describe(self) -> str:
        schema_id = (
            self.settings.schema.id
            if self.settings.schema
            else "(none)"
        )

        lines = [
            "",
            "=" * 62,
            " WATERMARK CREATED",
            "=" * 62,
            f"Input        {self.input_path}",
            f"Output       {self.output_path}",
            f"Schema       {schema_id}",
            f"Container    {self.container_bytes} B",
            f"Body         {len(self.body)} B",
            f"Codeword     {self.codeword_bytes} B",
            f"Bits         {self.bit_count}",
            f"Tiles        {self.tiles_embedded}"
            f" / {self.tiles_available}",
            "=" * 62,
            "",
            self.payload.format(),
            "",
            "=" * 62,
        ]

        return "\n".join(lines)


# ============================================================
# ENCODER
# ============================================================

class WatermarkEncoder:
    """Embeds a signed payload into an image.

    >>> encoder = WatermarkEncoder()                     # doctest: +SKIP
    >>> result = encoder.encode(                         # doctest: +SKIP
    ...     "in.png", "out.png", prompt="a cat"
    ... )

    Settings are passed to the constructor and can be replaced per
    instance, or overridden for one call with ``settings=``.
    """

    def __init__(
        self,
        settings: WatermarkSettings | None = None,
        schema: PayloadSchema | None = None,
        private_key_path: str | os.PathLike[str] | None = None,
        public_key_path: str | os.PathLike[str] | None = None,
    ) -> None:

        base = settings or WatermarkSettings()

        changes: dict[str, Any] = {}

        if schema is not None:
            changes["schema"] = schema

        if private_key_path is not None:
            changes["private_key_path"] = str(private_key_path)

        if public_key_path is not None:
            changes["public_key_path"] = str(public_key_path)

        if changes:
            base = base.evolve(**changes)

        self.settings = base

    # --------------------------------------------------------
    # Introspection
    # --------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Max container bytes that fit into one tile."""
        return self.settings.max_payload

    def describe(self) -> str:
        return self.settings.summary()

    # --------------------------------------------------------
    # Pipeline steps
    # --------------------------------------------------------

    def build_payload(
        self,
        values: Mapping[str, Any] | Payload | None = None,
        **kwargs: Any,
    ) -> Payload:
        """Create a payload bound to the configured schema."""

        schema = self.settings.require_schema()

        if isinstance(values, Payload):
            return values

        return Payload(schema, values, **kwargs)

    def pack(
        self, payload: Payload
    ) -> tuple[bytes, bytes]:
        """Serialise and frame a payload.

        Returns ``(raw_body, container_bytes)``.
        """

        settings = self.settings

        raw_body = payload.serialize()

        signature = None

        if settings.sign:

            private_key = load_private_key(
                settings.private_key_path
            )

            signature = private_key.sign(raw_body)

        box = cont.Container.build(
            schema_id=settings.require_schema().id,
            raw_body=raw_body,
            compression=settings.compression,
            signature=signature,
            lzma_preset=settings.lzma_preset,
        )

        packed = box.pack()

        self._check_capacity(packed)

        return raw_body, packed

    def reed_solomon(self, data: bytes) -> bytes:
        """Encode with the configured parity and verify the length."""

        settings = self.settings

        codec = RSCodec(settings.rs_parity)

        try:
            codeword = bytes(codec.encode(data))

        except Exception as e:
            raise PayloadError(
                f"Reed-Solomon encoding failed: {e}"
            ) from e

        expected = settings.geometry.encoded_length(
            len(data)
        )

        if len(codeword) != expected:
            raise ConfigError(
                f"reedsolo produced {len(codeword)} bytes but "
                f"{expected} were expected for a {len(data)} byte "
                f"payload with {settings.rs_parity} parity bytes. "
                f"The reedsolo version does not match this build."
            )

        return codeword

    def _check_capacity(self, packed: bytes) -> None:
        """Reject a container that cannot survive one tile."""

        settings = self.settings

        if len(packed) > settings.max_payload:

            raise CapacityError(
                f"Container is {len(packed)} bytes but one "
                f"{settings.tile_size}x{settings.tile_size} tile "
                f"holds at most {settings.max_payload} bytes "
                f"({settings.capacity} byte codeword, "
                f"{settings.schema_overhead} bytes of container "
                f"overhead). Shorten the payload, lower "
                f"rs_parity, use a smaller block_size or a larger "
                f"tile_size."
            )

        bits = len(packed) * 8

        if bits > settings.block_count:

            raise CapacityError(
                f"Container needs {bits} bits but a tile has only "
                f"{settings.block_count} blocks."
            )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    def embed_array(
        self,
        image: np.ndarray,
        packed: bytes,
    ) -> tuple[np.ndarray, int, int]:
        """Embed a packed container into an RGB array.

        Returns ``(output, tiles_embedded, tiles_available)``.
        """

        settings = self.settings

        height, width = image.shape[:2]

        tile = settings.tile_size

        if width < tile or height < tile:
            raise ImageError(
                f"Image must be at least {tile}x{tile} pixels, "
                f"got {width}x{height}."
            )

        self._check_capacity(packed)

        codeword = self.reed_solomon(packed)
        bits = dct.bytes_to_bits(codeword)

        output = image.copy()

        tiles_x, tiles_y = settings.tiles_across(width, height)

        positions = [
            (ty, tx)
            for ty in range(tiles_y)
            for tx in range(tiles_x)
        ]

        # embed_offset rotates where the sweep starts, so a payload can
        # dodge a known crop. It must not shift every tile: doing that
        # walks the grid off its own edge and drops the last row.
        if positions and any(settings.embed_offset):
            # positions is row major, so the index of tile (y, x) is
            # y * tiles_x + x and embed_offset is (y, x) as well.
            offset = (
                settings.embed_offset[0] * tiles_x
                + settings.embed_offset[1]
            ) % len(positions)
            positions = positions[offset:] + positions[:offset]

        if not settings.redundant and positions:
            positions = positions[:1]

        if settings.max_tiles:
            positions = positions[: settings.max_tiles]

        embedded = 0

        for ty, tx in positions:

            y = ty * tile
            x = tx * tile

            output[y:y + tile, x:x + tile] = dct.embed_tile(
                output[y:y + tile, x:x + tile],
                bits,
                settings.coef_a,
                settings.coef_b,
                settings.strength,
                settings.block_size,
            )

            embedded += 1

        return output, embedded, tiles_x * tiles_y

    # --------------------------------------------------------
    # Public entry points
    # --------------------------------------------------------

    def encode(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        values: Mapping[str, Any] | Payload | None = None,
        payload: Payload | None = None,
        progress: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> EncodeResult:
        """Encode an image file end to end."""

        report = progress or (lambda _message: None)

        report("[1/4] Loading image")

        image = dct.load_image(str(input_path))

        report("[2/4] Building payload")

        if payload is None:
            payload = self.build_payload(values, **kwargs)

        raw_body, packed = self.pack(payload)

        report("[3/4] Embedding")

        output, used, available = self.embed_array(image, packed)

        report("[4/4] Writing")

        dct.save_image(str(output_path), output)

        return EncodeResult(
            output_path=str(output_path),
            input_path=str(input_path),
            payload=payload,
            body=raw_body,
            container_bytes=len(packed),
            codeword_bytes=self.settings.geometry.encoded_length(
                len(packed)
            ),
            bit_count=len(packed) * 8,
            tiles_embedded=used,
            tiles_available=available,
            settings=self.settings,
        )

    def encode_array(
        self,
        image: np.ndarray,
        values: Mapping[str, Any] | None = None,
        payload: Payload | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, EncodeResult]:
        """Encode an in-memory array, useful for tests and pipelines."""

        if payload is None:
            payload = self.build_payload(values, **kwargs)

        raw_body, packed = self.pack(payload)

        output, used, available = self.embed_array(image, packed)

        result = EncodeResult(
            output_path="(memory)",
            input_path="(memory)",
            payload=payload,
            body=raw_body,
            container_bytes=len(packed),
            codeword_bytes=self.settings.geometry.encoded_length(
                len(packed)
            ),
            bit_count=len(packed) * 8,
            tiles_embedded=used,
            tiles_available=available,
            settings=self.settings,
        )

        return output, result

    # --------------------------------------------------------
    # Keys
    # --------------------------------------------------------

    def ensure_keys(self) -> None:
        """Generate the key pair if it does not exist yet."""
        ensure_key_pair(
            self.settings.private_key_path,
            self.settings.public_key_path,
        )
