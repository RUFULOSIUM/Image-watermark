"""Reed-Solomon geometry of the watermark container.

``reedsolo`` never encodes one big codeword. It splits the message into
chunks of ``nsize - nsym`` bytes and appends ``nsym`` parity bytes to
*every* chunk. A 300 byte payload with 32 parity bytes is therefore two
chunks and occupies 300 + 2 * 32 = 364 bytes, not 300 + 32 = 332.

The encoder and the decoder must agree on the encoded length, otherwise
the decoder slices the wrong number of bytes out of the tile and every
payload larger than a single chunk becomes undecodable. This happened in
v0.1 and is the reason this module exists.
"""

N_SIZE = 255
DEFAULT_PARITY = 32


class RSGeometry:
    """Chunk arithmetic for one Reed-Solomon parity setting.

    Instantiate once per parity value and reuse it; the encoder and the
    decoder must share the same instance (or at least the same parity)
    for the format to round-trip.
    """

    __slots__ = ("parity", "chunk")

    def __init__(self, parity: int = DEFAULT_PARITY):

        if not 1 <= parity < N_SIZE:
            raise ValueError(
                f"rs parity must be in 1..{N_SIZE - 1}, "
                f"got {parity}"
            )

        self.parity = parity
        self.chunk = N_SIZE - parity

    def __repr__(self) -> str:
        return f"RSGeometry(parity={self.parity})"

    def chunk_count(self, message_length: int) -> int:
        """Number of chunks a message of this size is split into."""
        return -(-message_length // self.chunk)

    def encoded_length(self, message_length: int) -> int:
        """Length ``reedsolo`` actually emits for this message length."""
        return (
            message_length
            + self.chunk_count(message_length) * self.parity
        )

    def max_message_length(self, capacity: int) -> int:
        """Largest message whose codeword still fits ``capacity`` bytes.

        ``encoded_length`` is non-decreasing, so the scan can stop at
        the first length that no longer fits.
        """
        best = 0

        for length in range(0, capacity + 1):
            if self.encoded_length(length) > capacity:
                break

            best = length

        return best


# Default geometry, kept as a module level singleton for the common
# 32 parity case and for backwards compatibility with v0.2 imports.
DEFAULT_GEOMETRY = RSGeometry(DEFAULT_PARITY)

#: Parity of the default geometry, the v0.2 ``RS_PARITY`` constant.
RS_PARITY = DEFAULT_PARITY

#: Bytes of message per chunk at the default parity, the v0.2 constant.
RS_CHUNK = DEFAULT_GEOMETRY.chunk


# ============================================================
# MODULE LEVEL SHORTCUTS
#
# v0.1 and v0.2 exposed these three functions and the tests of that
# era import them directly. They are kept so the historic API keeps
# working, but the encoder and the decoder use the RSGeometry instance
# from their settings instead, which is what makes a non-default parity
# possible.
# ============================================================

def chunk_count(message_length: int) -> int:
    """Chunks a message is split into at the default parity."""
    return DEFAULT_GEOMETRY.chunk_count(message_length)


def rs_len(message_length: int) -> int:
    """Length ``reedsolo`` emits at the default parity."""
    return DEFAULT_GEOMETRY.encoded_length(message_length)


def max_message_length(capacity: int) -> int:
    """Largest message that fits ``capacity`` at the default parity."""
    return DEFAULT_GEOMETRY.max_message_length(capacity)
