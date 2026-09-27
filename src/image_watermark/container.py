"""The binary container that frames a watermark inside a tile.

Layout of container version 1 (magic ``WM02``)::

    offset  size  field
    0       4     magic            b"WM02"
    4       2     container version (uint16, currently 1)
    6       2     compression id   (uint16)
    8       2     schema id length (uint16)
    10      2     flags            (uint16, bit 0: signature present)
    12      4     body length      (uint32, as stored, post compression)
    16      4     raw length       (uint32, before compression)
    20      N     schema id        (utf-8)
    20+N    M     body
    20+N+M  64    signature        (only when the flag is set)

The body length is stored uncompressed because the decoder has to know
the exact Reed-Solomon codeword length *before* it can pull bytes out of
the image, which happens long before it could decompress anything. The
same is true of the signature flag: whether the 64 trailing bytes exist
changes the total length, and the decoder must get that right from the
header alone.

Container version 0 was ``WM01`` and had neither a version field nor a
schema id. It is still readable so images produced by v0.1 and v0.2 keep
decoding; :func:`read_length` auto-detects which framing a tile uses.
"""

from __future__ import annotations

import lzma
import struct
import zlib

from dataclasses import dataclass
from typing import Any

from image_watermark.errors import ContainerError


# ============================================================
# CONSTANTS
# ============================================================

MAGIC_V1 = b"WM02"
MAGIC_LEGACY = b"WM01"
CONTAINER_VERSION = 1
LEGACY_VERSION = 0

SIGNATURE_SIZE = 64

#: magic + version + compression + id length + flags + body length + raw length
HEADER_FIXED_SIZE = 20

#: Legacy ``WM01``: magic + 2 byte body length
LEGACY_HEADER_SIZE = 6

#: Bit 0 of the flags field: a 64 byte signature follows the body.
FLAG_SIGNED = 1 << 0

#: Longest schema id the framing accepts. Bounding this is what lets
#: the decoder sniff a container from a fixed-size probe: header plus
#: the id, and nothing more, has to be readable before the length is
#: known.
MAX_SCHEMA_ID_LENGTH = 64

COMPRESSION_NONE = 0
COMPRESSION_ZLIB = 1
COMPRESSION_LZMA = 2

COMPRESSION_BY_ID = {
    COMPRESSION_NONE: "none",
    COMPRESSION_ZLIB: "zlib",
    COMPRESSION_LZMA: "lzma",
}

COMPRESSION_ID = {
    name: cid for cid, name in COMPRESSION_BY_ID.items()
}


def compression_choices() -> tuple[str, ...]:
    """Names of every supported body compression."""
    return tuple(COMPRESSION_ID)

DEFAULT_LZMA_PRESET = 9


# ============================================================
# COMPRESSION
# ============================================================

def compress(
    data: bytes,
    algorithm: str = "lzma",
    lzma_preset: int = DEFAULT_LZMA_PRESET,
) -> bytes:
    """Compress the body according to ``algorithm``."""

    if algorithm == "none":
        return data

    if algorithm == "zlib":
        return zlib.compress(data)

    if algorithm == "lzma":
        return lzma.compress(
            data, lzma.FORMAT_XZ, preset=lzma_preset
        )

    raise ContainerError(
        f"Unknown compression {algorithm!r}. Known: "
        f"{', '.join(compression_choices())}"
    )


def decompress(
    data: bytes,
    algorithm: str,
    expected_raw_length: int | None = None,
) -> bytes:
    """Strictly decompress a body.

    ``lzma.decompress`` ignores trailing bytes and cannot report a
    truncated stream, so an ``LZMADecompressor`` is used and both
    conditions are checked. An ``expected_raw_length`` additionally
    guards against a stream that expands to the wrong size.
    """

    if algorithm == "none":
        result = data

    elif algorithm == "zlib":

        try:
            result = zlib.decompress(data)

        except zlib.error as e:
            raise ContainerError(
                f"Body decompression failed: {e}"
            ) from e

    elif algorithm == "lzma":

        decompressor = lzma.LZMADecompressor(
            format=lzma.FORMAT_XZ
        )

        try:
            result = decompressor.decompress(data)

        except lzma.LZMAError as e:
            raise ContainerError(
                f"Body decompression failed: {e}"
            ) from e

        if not decompressor.eof:
            raise ContainerError(
                "Incomplete LZMA stream."
            )

        if decompressor.unused_data:
            raise ContainerError(
                "Trailing data after LZMA stream."
            )

    else:
        raise ContainerError(
            f"Unknown compression {algorithm!r}"
        )

    if (
        expected_raw_length is not None
        and len(result) != expected_raw_length
    ):
        raise ContainerError(
            f"Body expanded to {len(result)} bytes, "
            f"header declared {expected_raw_length}."
        )

    return result


# ============================================================
# HEADER
# ============================================================

@dataclass(frozen=True)
class Header:
    """The fixed part of a version 1 container."""

    version: int
    compression: str
    schema_id: str
    body_length: int
    raw_length: int
    signed: bool = True

    @property
    def size(self) -> int:
        """Total header size including the schema id."""
        return HEADER_FIXED_SIZE + len(
            self.schema_id.encode("utf-8")
        )

    @property
    def total_length(self) -> int:
        """Bytes the whole container occupies, signature included.

        This is the number the decoder needs before it can size its
        Reed-Solomon read, so it must account for the signature only
        when the header says one is present.
        """
        return (
            self.size
            + self.body_length
            + (SIGNATURE_SIZE if self.signed else 0)
        )

    def pack(self) -> bytes:
        schema_id = self.schema_id.encode("utf-8")

        if len(schema_id) > MAX_SCHEMA_ID_LENGTH:
            raise ContainerError(
                f"Schema id is too long: {len(schema_id)} > "
                f"{MAX_SCHEMA_ID_LENGTH} bytes"
            )

        return (
            MAGIC_V1
            + struct.pack(">H", self.version)
            + struct.pack(
                ">H", COMPRESSION_ID[self.compression]
            )
            + struct.pack(">H", len(schema_id))
            + struct.pack(
                ">H", FLAG_SIGNED if self.signed else 0
            )
            + struct.pack(">I", self.body_length)
            + struct.pack(">I", self.raw_length)
            + schema_id
        )


@dataclass(frozen=True)
class LegacyHeader:
    """The v0.1 / v0.2 ``WM01`` framing."""

    body_length: int
    signed: bool = True

    @property
    def total_length(self) -> int:
        return (
            LEGACY_HEADER_SIZE
            + self.body_length
            + (SIGNATURE_SIZE if self.signed else 0)
        )


# ============================================================
# READ HEADER
# ============================================================

def peek_magic(data: bytes) -> bytes:
    """The first bytes of ``data``, for magic sniffing."""
    return bytes(data[:4])


#: How many bytes the decoder has to pull out of a tile before it can
#: tell whether a container is there and how long it is: the legacy
#: header, or a version 1 header plus the longest schema id allowed.
SNIFF_SIZE = max(LEGACY_HEADER_SIZE, HEADER_FIXED_SIZE + MAX_SCHEMA_ID_LENGTH)


def read_header(data: bytes) -> Header:
    """Parse a version 1 header from the start of ``data``."""

    if len(data) < HEADER_FIXED_SIZE:
        raise ContainerError(
            f"Container too small: need {HEADER_FIXED_SIZE} bytes, "
            f"got {len(data)}"
        )

    if data[:4] != MAGIC_V1:
        raise ContainerError(
            f"Not a {MAGIC_V1!r} container "
            f"(magic {bytes(data[:4])!r})"
        )

    (
        version,
        compression_id,
        schema_id_length,
        flags,
        body_length,
        raw_length,
    ) = struct.unpack(">HHHHII", data[4:HEADER_FIXED_SIZE])

    if version != CONTAINER_VERSION:
        raise ContainerError(
            f"Unsupported container version {version}, "
            f"this build understands {CONTAINER_VERSION}"
        )

    if compression_id not in COMPRESSION_BY_ID:
        raise ContainerError(
            f"Unknown compression id {compression_id}"
        )

    if flags & ~FLAG_SIGNED:
        raise ContainerError(
            f"Unknown container flags 0x{flags:04x}"
        )

    if schema_id_length == 0:
        raise ContainerError("Empty schema id")

    if schema_id_length > MAX_SCHEMA_ID_LENGTH:
        raise ContainerError(
            f"Schema id is too long: {schema_id_length} > "
            f"{MAX_SCHEMA_ID_LENGTH} bytes"
        )

    end = HEADER_FIXED_SIZE + schema_id_length

    if end > len(data):
        raise ContainerError(
            f"Truncated schema id: need {schema_id_length} bytes"
        )

    try:
        schema_id = data[
            HEADER_FIXED_SIZE:end
        ].decode("utf-8")

    except UnicodeDecodeError as e:
        raise ContainerError(
            f"Schema id is not valid utf-8: {e}"
        ) from e

    if body_length == 0:
        raise ContainerError("Body length is zero")

    return Header(
        version=version,
        compression=COMPRESSION_BY_ID[compression_id],
        schema_id=schema_id,
        body_length=body_length,
        raw_length=raw_length,
        signed=bool(flags & FLAG_SIGNED),
    )


def read_legacy_header(data: bytes) -> LegacyHeader:
    """Parse the v0.2 ``WM01`` framing from the start of ``data``."""

    if len(data) < LEGACY_HEADER_SIZE:
        raise ContainerError(
            f"Legacy container too small: need "
            f"{LEGACY_HEADER_SIZE} bytes, got {len(data)}"
        )

    body_length = struct.unpack(
        ">H", data[4:LEGACY_HEADER_SIZE]
    )[0]

    if body_length == 0:
        raise ContainerError("Body length is zero")

    return LegacyHeader(body_length=body_length)


def read_length(
    data: bytes,
) -> tuple[int, int] | None:
    """Sniff the framing and return ``(declared_length, version)``.

    ``None`` means the data does not start with a watermark container.

    This only reports what the header *claims*; it deliberately does
    not check the length against ``len(data)``. The decoder sniffs a
    small probe first -- it does not know the container size yet, and
    finding that out is the whole point of reading the header. Callers
    must therefore validate the returned length against the capacity
    they actually have; :meth:`Container.unpack` does that when it
    insists on an exact slice.
    """

    if len(data) < 4:
        return None

    magic = data[:4]

    if magic == MAGIC_V1:

        try:
            header = read_header(data)

        except ContainerError:
            return None

        return header.total_length, header.version

    if magic == MAGIC_LEGACY:

        try:
            header = read_legacy_header(data)

        except ContainerError:
            return None

        # v0.2 always signed, so that is the primary reading. The
        # unsigned alternative is offered by alternate_lengths() in
        # case the primary does not survive Reed-Solomon.
        return header.total_length, LEGACY_VERSION

    return None


def alternate_lengths(data: bytes) -> tuple[int, ...]:
    """Extra container lengths to try when the primary one fails.

    Only ``WM01`` is ambiguous: v0.2 always appended a 64 byte
    signature, but a container written without one is still a valid
    Reed-Solomon read, just 64 bytes shorter. ``WM02`` records the
    signature in its flags, so it never needs a second guess.
    """

    if data[:4] != MAGIC_LEGACY:
        return ()

    try:
        header = read_legacy_header(data)

    except ContainerError:
        return ()

    unsigned = LEGACY_HEADER_SIZE + header.body_length

    if unsigned == header.total_length:
        return ()

    return (unsigned,)


# ============================================================
# CONTAINER
# ============================================================

@dataclass
class Container:
    """A packed watermark ready for Reed-Solomon encoding."""

    schema_id: str
    body: bytes
    raw_length: int
    compression: str = "lzma"
    signature: bytes | None = None
    version: int = CONTAINER_VERSION

    @property
    def header(self) -> Header:
        return Header(
            version=self.version,
            compression=self.compression,
            schema_id=self.schema_id,
            body_length=len(self.body),
            raw_length=self.raw_length,
            signed=self.signature is not None,
        )

    @property
    def signed(self) -> bool:
        return self.signature is not None

    def pack(self) -> bytes:
        """Serialise the container to bytes."""

        data = self.header.pack() + self.body

        if self.signature is not None:

            if len(self.signature) != SIGNATURE_SIZE:
                raise ContainerError(
                    f"Signature must be {SIGNATURE_SIZE} bytes, "
                    f"got {len(self.signature)}"
                )

            data += self.signature

        return data

    # --------------------------------------------------------
    # Constructors
    # --------------------------------------------------------

    @classmethod
    def build(
        cls,
        schema_id: str,
        raw_body: bytes,
        compression: str = "lzma",
        signature: bytes | None = None,
        lzma_preset: int = DEFAULT_LZMA_PRESET,
    ) -> Container:
        """Compress ``raw_body`` and frame it."""

        body = compress(
            raw_body, compression, lzma_preset
        )

        return cls(
            schema_id=schema_id,
            body=body,
            raw_length=len(raw_body),
            compression=compression,
            signature=signature,
        )

    @classmethod
    def unpack(
        cls, data: bytes, version: int
    ) -> tuple[Container, bytes, bytes]:
        """Split a container into ``(container, body, signature)``.

        The body is returned still compressed; call
        :func:`decompress` with the header information to expand it.

        The slice must be exactly one container. Trailing bytes mean the
        caller read a wrong length somewhere, so they are rejected
        rather than dropped.
        """

        if version == LEGACY_VERSION:
            return cls._unpack_legacy(data)

        header = read_header(data)

        expected = header.total_length

        if expected != len(data):
            raise ContainerError(
                f"Container is {len(data)} bytes but its header "
                f"describes {expected}."
            )

        pos = header.size
        end = pos + header.body_length

        body = data[pos:end]

        signature = b""

        if header.signed:
            signature = data[end:end + SIGNATURE_SIZE]

        container = cls(
            schema_id=header.schema_id,
            body=body,
            raw_length=header.raw_length,
            compression=header.compression,
            signature=signature or None,
            version=header.version,
        )

        return container, body, signature

    @staticmethod
    def _unpack_legacy(
        data: bytes,
    ) -> tuple[Container, bytes, bytes]:
        """Split a v0.2 ``WM01`` container."""

        header = read_legacy_header(data)

        pos = LEGACY_HEADER_SIZE
        end = pos + header.body_length

        if end > len(data):
            raise ContainerError(
                f"Truncated body: need {header.body_length} bytes, "
                f"got {len(data) - pos}"
            )

        signature = data[end:end + SIGNATURE_SIZE]

        container = Container(
            schema_id="",
            body=data[pos:end],
            raw_length=0,
            compression="lzma",
            signature=signature or None,
            version=LEGACY_VERSION,
        )

        return container, data[pos:end], signature

    # --------------------------------------------------------
    # Presentation
    # --------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "schema_id": self.schema_id or "(legacy WM01)",
            "compression": self.compression,
            "raw_length": self.raw_length,
            "body_length": len(self.body),
            "signed": self.signed,
        }
