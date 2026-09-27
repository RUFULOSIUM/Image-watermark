"""All tunable settings of the watermark, in one place.

Everything that was a module level constant in v0.2 is a field here, and
nothing is read from the environment implicitly. A settings object is
cheap to copy, so experimenting with a variant reads naturally::

    base = WatermarkSettings()
    stealthy = replace(base, strength=8.0, compression="zlib")
    strong = replace(base, rs_parity=64)

Derived values such as the tile capacity and the maximum payload are
computed from the fields rather than hard-coded, which is what makes
changing ``tile_size`` or ``rs_parity`` actually work.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from image_watermark.container import (
    HEADER_FIXED_SIZE,
    SIGNATURE_SIZE,
    compression_choices,
)
from image_watermark.dct import (
    count_blocks,
    validate_coefficients,
)
from image_watermark.errors import ConfigError
from image_watermark.keys import (
    DEFAULT_PRIVATE_KEY,
    DEFAULT_PUBLIC_KEY,
)
from image_watermark.rs import DEFAULT_GEOMETRY, RSGeometry
from image_watermark.schema import PayloadSchema


# ============================================================
# SETTINGS
# ============================================================

@dataclass
class WatermarkSettings:
    """Configuration for encoding and decoding."""

    # --------------------------------------------------------
    # Payload
    # --------------------------------------------------------

    schema: PayloadSchema | None = None

    compression: str = "lzma"

    lzma_preset: int = 9

    sign: bool = True

    private_key_path: str = DEFAULT_PRIVATE_KEY

    public_key_path: str = DEFAULT_PUBLIC_KEY

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    tile_size: int = 512

    block_size: int = 8

    coef_a: tuple[int, int] = (3, 4)

    coef_b: tuple[int, int] = (4, 3)

    strength: float = 12.0

    # --------------------------------------------------------
    # Error correction
    # --------------------------------------------------------

    rs_parity: int = DEFAULT_GEOMETRY.parity

    # --------------------------------------------------------
    # Redundancy
    # --------------------------------------------------------

    #: Embed the payload into every tile, or only into the first one.
    redundant: bool = True

    #: Offset of the first embedded tile as ``(y, x)`` in tile units,
    #: useful to dodge a known crop. ``(1, 0)`` therefore starts on the
    #: second row. Rotates the sweep only, it never drops a tile.
    embed_offset: tuple[int, int] = (0, 0)

    #: Restrict embedding to this many tiles, 0 means "no limit".
    max_tiles: int = 0

    _geometry: RSGeometry = field(
        default=DEFAULT_GEOMETRY,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:

        if self.compression not in compression_choices():
            raise ConfigError(
                f"compression must be one of "
                f"{', '.join(compression_choices())}, got "
                f"{self.compression!r}"
            )

        if self.block_size < 2:
            raise ConfigError(
                f"block_size must be >= 2, got {self.block_size}"
            )

        if self.tile_size % self.block_size:
            raise ConfigError(
                f"tile_size {self.tile_size} is not a multiple of "
                f"block_size {self.block_size}"
            )

        if self.strength <= 0:
            raise ConfigError(
                f"strength must be > 0, got {self.strength}"
            )

        if self.max_tiles < 0:
            raise ConfigError(
                f"max_tiles must be >= 0, got {self.max_tiles}"
            )

        validate_coefficients(
            self.coef_a, self.coef_b, self.block_size
        )

        self._geometry = RSGeometry(self.rs_parity)

    # --------------------------------------------------------
    # Derived capacity
    # --------------------------------------------------------

    @property
    def geometry(self) -> RSGeometry:
        return self._geometry

    @property
    def block_count(self) -> int:
        """Blocks per tile, one bit each."""
        return count_blocks(
            self.tile_size, self.tile_size, self.block_size
        )

    @property
    def capacity(self) -> int:
        """Codeword bytes that fit into one tile."""
        return self.block_count // 8

    @property
    def max_payload(self) -> int:
        """Largest container that survives one Reed-Solomon round trip."""
        return self.geometry.max_message_length(
            self.capacity
        )

    @property
    def signature_size(self) -> int:
        return SIGNATURE_SIZE if self.sign else 0

    @property
    def schema_overhead(self) -> int:
        """Container bytes that are not payload."""

        if self.schema is None:
            return 0

        id_bytes = len(self.schema.id.encode("utf-8"))

        return (
            HEADER_FIXED_SIZE
            + id_bytes
            + self.signature_size
        )

    @property
    def max_raw_body(self) -> int:
        """Largest uncompressed body that still fits into a tile.

        Only meaningful for ``compression="none"``, where the body is
        stored verbatim. For the compressing algorithms the real limit
        depends on how well the data compresses, so this is an upper
        bound and the encoder reports the true number if the result
        overflows anyway.
        """

        return self.max_payload - self.schema_overhead

    def tiles_across(self, width: int, height: int):
        """Number of whole tiles in an image, as ``(tiles_x, tiles_y)``."""

        return (
            width // self.tile_size,
            height // self.tile_size,
        )

    # --------------------------------------------------------
    # Copying
    # --------------------------------------------------------

    def evolve(self, **changes: Any) -> WatermarkSettings:
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)

    def with_schema(self, schema: PayloadSchema) -> WatermarkSettings:
        """Return a copy bound to ``schema``."""
        return replace(self, schema=schema)

    def require_schema(self) -> PayloadSchema:

        if self.schema is None:
            raise ConfigError(
                "No payload schema set. Pass schema=... to the "
                "settings or to the encoder."
            )

        return self.schema

    # --------------------------------------------------------
    # Presentation
    # --------------------------------------------------------

    def summary(self) -> str:
        """Multi-line overview, used by the CLI output."""

        schema_id = (
            self.schema.id if self.schema else "(none)"
        )

        rows = [
            ("schema", schema_id),
            ("compression", self.compression),
            ("signed", str(self.sign)),
            ("tile", f"{self.tile_size}x{self.tile_size}"),
            ("block", f"{self.block_size}x{self.block_size}"),
            (
                "coefficients",
                f"a={self.coef_a} b={self.coef_b}",
            ),
            ("strength", f"{self.strength:g}"),
            ("rs parity", f"{self.rs_parity}"),
            ("blocks/tile", f"{self.block_count}"),
            ("capacity", f"{self.capacity} B codeword"),
            ("max payload", f"{self.max_payload} B"),
            ("max raw body", f"{self.max_raw_body} B"),
        ]

        width = max(len(k) for k, _ in rows)

        return "\n".join(
            f"{k:<{width}}  {v}" for k, v in rows
        )
