"""The watermark decoder.

The decoder mirrors :class:`~image_watermark.encoder.WatermarkEncoder`:
it reads a container header, sizes its Reed-Solomon read from it, and
then resolves the schema the container names. A container whose schema
is unknown is reported with the id it carries rather than being
misparsed with whatever schema happens to be configured.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from reedsolo import RSCodec

from image_watermark import container as cont
from image_watermark import dct
from image_watermark.errors import (
    ConfigError,
    ContainerError,
    DecodingError,
    ImageError,
    SchemaError,
)
from image_watermark.keys import load_public_key
from image_watermark.legacy import (
    LEGACY_SCHEMA_ID,
    parse_legacy_body,
)
from image_watermark.payload import Payload
from image_watermark.schema import PayloadSchema
from image_watermark.settings import WatermarkSettings


# ============================================================
# SCHEMA RESOLUTION
# ============================================================

class SchemaRegistry:
    """Maps schema ids to :class:`PayloadSchema` objects.

    The container carries the id, the registry supplies the layout. That
    split is what lets a watermark be read without any external hints
    while still allowing arbitrary user defined schemes.
    """

    def __init__(
        self, schemas: list[PayloadSchema] | None = None
    ) -> None:
        self._schemas: dict[str, PayloadSchema] = {}

        for schema in schemas or ():
            self.register(schema)

    def register(self, schema: PayloadSchema) -> PayloadSchema:

        if not isinstance(schema, PayloadSchema):
            raise SchemaError(
                f"Expected a PayloadSchema, got "
                f"{type(schema).__name__}"
            )

        existing = self._schemas.get(schema.id)

        if existing is not None and existing != schema:
            raise SchemaError(
                f"Schema id {schema.id!r} is already registered "
                f"with a different layout."
            )

        self._schemas[schema.id] = schema

        return schema

    def load(self, path: str | os.PathLike[str]) -> PayloadSchema:
        return self.register(PayloadSchema.load(path))

    def load_dir(
        self, path: str | os.PathLike[str]
    ) -> list[PayloadSchema]:
        """Register every ``*.json`` in a directory."""

        if not os.path.isdir(path):
            raise SchemaError(
                f"Not a directory: {path}"
            )

        loaded = []

        for name in sorted(os.listdir(path)):

            if not name.endswith(".json"):
                continue

            loaded.append(
                self.load(os.path.join(path, name))
            )

        return loaded

    def get(self, schema_id: str) -> PayloadSchema | None:
        return self._schemas.get(schema_id)

    def require(self, schema_id: str) -> PayloadSchema:

        schema = self.get(schema_id)

        if schema is None:
            known = ", ".join(sorted(self._schemas)) or "(none)"

            raise SchemaError(
                f"The watermark names schema {schema_id!r}, which is "
                f"not registered. Known schemas: {known}. Pass the "
                f"schema file to the decoder or add it to the "
                f"registry."
            )

        return schema

    def ids(self) -> list[str]:
        return sorted(self._schemas)

    def __len__(self) -> int:
        return len(self._schemas)

    def __contains__(self, schema_id: object) -> bool:
        return schema_id in self._schemas

    def __iter__(self):
        return iter(self._schemas.values())


# ============================================================
# RESULT
# ============================================================

@dataclass
class TileCandidate:
    """One tile that produced a Reed-Solomon decodable container."""

    x: int
    y: int
    data: bytes
    version: int
    schema_id: str

    def __repr__(self) -> str:
        return (
            f"<TileCandidate ({self.x},{self.y}) "
            f"v{self.version} {self.schema_id!r} "
            f"{len(self.data)}B>"
        )


@dataclass
class DecodeResult:
    """The outcome of decoding an image."""

    payload: Payload
    valid: bool
    tile: tuple[int, int]
    container: cont.Container
    version: int
    schema_id: str
    settings: WatermarkSettings
    candidates: list[TileCandidate] = field(
        default_factory=list, repr=False
    )

    def describe(self) -> str:
        lines = [
            "",
            "=" * 62,
            " WATERMARK FOUND",
            "=" * 62,
            f"Tile         ({self.tile[0]},{self.tile[1]})",
            f"Container    v{self.version} ({self.schema_id})",
            f"Compression  {self.container.compression}",
            f"Signature    "
            f"{'VALID' if self.valid else 'INVALID'}",
            "=" * 62,
            "",
            self.payload.format(),
            "",
            "=" * 62,
        ]

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        data = self.payload.to_dict()
        data["_signature_valid"] = self.valid
        data["_schema_id"] = self.schema_id
        data["_container_version"] = self.version
        return data


# ============================================================
# DECODER
# ============================================================

class WatermarkDecoder:
    """Recovers a payload from an image.

    >>> decoder = WatermarkDecoder(schema_registry)       # doctest: +SKIP
    >>> result = decoder.decode("out.png")                # doctest: +SKIP
    >>> result.payload["prompt"]                           # doctest: +SKIP
    'a cat'
    """

    def __init__(
        self,
        settings: WatermarkSettings | None = None,
        schema: PayloadSchema | None = None,
        registry: SchemaRegistry | None = None,
        public_key_path: str | os.PathLike[str] | None = None,
    ) -> None:

        base = settings or WatermarkSettings()
        changes: dict[str, Any] = {}

        if schema is not None:
            changes["schema"] = schema

        if public_key_path is not None:
            changes["public_key_path"] = str(public_key_path)

        if changes:
            base = base.evolve(**changes)

        self.settings = base
        self.registry = registry or SchemaRegistry()

        if base.schema is not None:
            self.registry.register(base.schema)

    # --------------------------------------------------------
    # Introspection
    # --------------------------------------------------------

    def register(self, schema: PayloadSchema) -> PayloadSchema:
        return self.registry.register(schema)

    def load_schema(self, path: str | os.PathLike[str]) -> PayloadSchema:
        return self.registry.load(path)

    def describe(self) -> str:
        return (
            f"{self.settings.summary()}\n"
            f"schemas       {', '.join(self.registry.ids())}"
        )

    # --------------------------------------------------------
    # Tile scanning
    # --------------------------------------------------------

    def scan_tiles(
        self, image: np.ndarray
    ) -> list[TileCandidate]:
        """Read every tile and keep the ones that yield a container."""

        settings = self.settings

        height, width = image.shape[:2]
        tile = settings.tile_size

        if width < tile or height < tile:
            raise ImageError(
                f"Image must be at least {tile}x{tile} pixels, "
                f"got {width}x{height}."
            )

        tiles_x, tiles_y = settings.tiles_across(width, height)

        codec = RSCodec(settings.rs_parity)

        candidates: list[TileCandidate] = []

        for ty in range(tiles_y):

            for tx in range(tiles_x):

                y = ty * tile
                x = tx * tile

                block = image[y:y + tile, x:x + tile]

                found = self._read_tile(block, codec, tx, ty)

                if found is not None:
                    candidates.append(found)

        return candidates

    def _read_tile(
        self,
        block: np.ndarray,
        codec: RSCodec,
        tx: int,
        ty: int,
    ) -> TileCandidate | None:
        """Try to pull one container out of a single tile."""

        settings = self.settings

        # The header and the schema id are small and uncompressed, so
        # they can be read before anything else is known. SNIFF_SIZE is
        # the worst case across both framings.
        probe = self._read_bytes(block, cont.SNIFF_SIZE, settings)

        if probe is None:
            return None

        sniffed = cont.read_length(probe)

        if sniffed is None:
            return None

        total_length, version = sniffed

        # WM01 does not record whether a signature follows the body, so
        # the declared length may be off by 64 bytes. Try the primary
        # reading first and fall back to the unsigned one.
        for length in (total_length, *cont.alternate_lengths(probe)):

            found = self._read_at_length(
                block, codec, length, version, tx, ty
            )

            if found is not None:
                return found

        return None

    def _read_at_length(
        self,
        block: np.ndarray,
        codec: RSCodec,
        total_length: int,
        version: int,
        tx: int,
        ty: int,
    ) -> TileCandidate | None:
        """Try one concrete container length."""

        settings = self.settings

        if total_length < 0:
            return None

        codeword_length = settings.geometry.encoded_length(
            total_length
        )

        if codeword_length > settings.capacity:
            return None

        raw = self._read_bytes(block, codeword_length, settings)

        if raw is None:
            return None

        try:
            message = bytes(codec.decode(raw)[0])

        except Exception:
            return None

        if len(message) != total_length:
            return None

        schema_id = ""

        if version != cont.LEGACY_VERSION:

            try:
                header = cont.read_header(message)
                schema_id = header.schema_id

            except ContainerError:
                return None

        return TileCandidate(
            x=tx,
            y=ty,
            data=message,
            version=version,
            schema_id=schema_id,
        )

    def _read_bytes(
        self,
        block: np.ndarray,
        count: int,
        settings: WatermarkSettings,
    ) -> bytes | None:
        """Extract ``count`` bytes from a tile, or ``None``."""

        bits = dct.extract_tile(
            block,
            count * 8,
            settings.coef_a,
            settings.coef_b,
            settings.block_size,
        )

        if len(bits) < count * 8:
            return None

        return dct.bits_to_bytes(bits)

    # --------------------------------------------------------
    # Candidate handling
    # --------------------------------------------------------

    def _interpret(
        self, candidate: TileCandidate
    ) -> tuple[Payload, cont.Container, bool]:
        """Unpack, decompress, parse and verify one candidate."""

        settings = self.settings

        box, body, signature = cont.Container.unpack(
            candidate.data, candidate.version
        )

        raw_body = cont.decompress(
            body,
            box.compression,
            expected_raw_length=box.raw_length or None,
        )

        if candidate.version == cont.LEGACY_VERSION:
            values = parse_legacy_body(raw_body)
            payload = Payload(
                self.settings.schema
                or PayloadSchema.from_dict(
                    {
                        "id": LEGACY_SCHEMA_ID,
                        "fields": [
                            {"name": "model", "type": "string"},
                            {
                                "name": "version",
                                "type": "string",
                            },
                            {"name": "image_id", "type": "string"},
                            {"name": "timestamp", "type": "uint64"},
                            {
                                "name": "prompt",
                                "type": "string",
                                "compression": "zlib",
                            },
                        ],
                    }
                ),
                values,
            )

        else:
            schema = self.registry.require(
                candidate.schema_id
            )

            payload = Payload.deserialize(schema, raw_body)

        payload.body = raw_body
        payload.signature = signature or None

        return payload, box, self.verify(payload)

    def verify(self, payload: Payload) -> bool:
        """Check the payload signature, if there is one."""

        if payload.signature is None or payload.body is None:
            return False

        try:
            public_key = load_public_key(
                self.settings.public_key_path
            )

        except Exception:
            return False

        try:
            public_key.verify(
                payload.signature, payload.body
            )

        except Exception:
            return False

        return True

    # --------------------------------------------------------
    # Public entry points
    # --------------------------------------------------------

    def decode(
        self,
        path: str | os.PathLike[str],
        progress: Callable[[str], None] | None = None,
    ) -> DecodeResult:
        """Decode an image file and return the best candidate."""

        report = progress or (lambda _message: None)

        report("[1/3] Loading image")

        image = dct.load_image(str(path))

        return self.decode_array(image, progress=report)

    def decode_array(
        self,
        image: np.ndarray,
        progress: Callable[[str], None] | None = None,
    ) -> DecodeResult:
        """Decode an in-memory array."""

        report = progress or (lambda _message: None)

        report("[2/3] Scanning tiles")

        candidates = self.scan_tiles(image)

        if not candidates:
            raise DecodingError(
                "No watermark found in this image."
            )

        report("[3/3] Verifying")

        parsed: list[
            tuple[TileCandidate, Payload, cont.Container, bool]
        ] = []
        problems: list[str] = []

        for candidate in candidates:

            try:
                payload, box, valid = self._interpret(candidate)

            except (ContainerError, SchemaError) as e:
                problems.append(str(e))
                continue

            except Exception as e:
                problems.append(
                    f"{type(e).__name__}: {e}"
                )
                continue

            parsed.append(
                (candidate, payload, box, valid)
            )

        if not parsed:
            detail = "; ".join(problems[:3])

            raise DecodingError(
                "Watermark data was found but no candidate could "
                f"be decoded. {detail}"
            )

        # A validly signed payload always wins over a coincidentally
        # decodable one, no matter which tile was scanned first.
        for entry in parsed:

            if entry[3]:
                return self._result(entry, candidates)

        return self._result(parsed[0], candidates)

    def _result(
        self,
        entry: tuple[
            TileCandidate, Payload, cont.Container, bool
        ],
        candidates: list[TileCandidate],
    ) -> DecodeResult:

        candidate, payload, box, valid = entry

        return DecodeResult(
            payload=payload,
            valid=valid,
            tile=(candidate.x, candidate.y),
            container=box,
            version=candidate.version,
            schema_id=(
                candidate.schema_id or LEGACY_SCHEMA_ID
            ),
            settings=self.settings,
            candidates=candidates,
        )
