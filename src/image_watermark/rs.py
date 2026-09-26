"""Reed-Solomon geometry of the watermark container.

``reedsolo`` never encodes one big codeword. It splits the message into
chunks of ``nsize - nsym`` bytes and appends ``nsym`` parity bytes to
*every* chunk. A 300 byte payload is therefore two chunks and occupies
300 + 2 * 32 = 364 bytes, not 300 + 32 = 332.

The encoder and the decoder must agree on the encoded length, otherwise
the decoder slices the wrong number of bytes out of the tile and every
payload larger than a single chunk becomes undecodable.
"""

N_SIZE = 255
RS_PARITY = 32
RS_CHUNK = N_SIZE - RS_PARITY


def chunk_count(message_length: int) -> int:
    """Number of Reed-Solomon chunks a message of this size is split into."""
    return -(-message_length // RS_CHUNK)


def rs_len(message_length: int) -> int:
    """Encoded length reedsolo actually emits for this message length."""
    return message_length + chunk_count(message_length) * RS_PARITY


def max_message_length(capacity: int) -> int:
    """Largest message whose codeword still fits into ``capacity`` bytes.

    ``rs_len`` is non-decreasing, so the scan can stop at the first
    length that no longer fits.
    """
    best = 0

    for length in range(0, capacity + 1):
        if rs_len(length) > capacity:
            break

        best = length

    return best
