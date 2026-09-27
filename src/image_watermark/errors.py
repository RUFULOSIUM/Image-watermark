"""Exception hierarchy for the watermark library.

Up to v0.2 every failure path called ``sys.exit()``, which made the
code impossible to use as a library and impossible to assert against in
tests. The classes in this module raise instead; the legacy
``encode.py`` / ``decode.py`` wrappers translate them into the old
``SystemExit`` behaviour so existing callers keep working.
"""


class WatermarkError(Exception):
    """Base class for every error raised by this library."""


class ConfigError(WatermarkError):
    """A setting or schema value is unusable."""


class SchemaError(ConfigError):
    """A payload schema is malformed or internally inconsistent."""


class KeyFileError(WatermarkError):
    """A key could not be generated, loaded or validated."""


class PayloadError(WatermarkError):
    """A payload could not be built from the given values."""


class CapacityError(PayloadError):
    """The payload does not fit into a single tile.

    Kept separate from ``PayloadError`` because it is the one failure
    a user can usually fix by choosing a different schema, a stronger
    compression or a larger tile.
    """


class ContainerError(WatermarkError):
    """The binary container framing is invalid."""


class ImageError(WatermarkError):
    """The image could not be read, decoded or written."""


class DecodingError(WatermarkError):
    """No usable watermark could be recovered from an image."""
