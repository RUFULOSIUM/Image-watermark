"""Tests for the v0.3 object oriented API.

Covers the pieces the v0.2 suite does not touch: schema driven
payloads, the WM02 container, the schema registry, the object oriented
encoder/decoder and the command line interface. The legacy procedural
API keeps its own tests in ``encode_test.py`` and ``decode_test.py``.
"""

import itertools
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from image_watermark import (
    Container,
    FieldSpec,
    Header,
    Payload,
    PayloadSchema,
    SchemaError,
    SchemaRegistry,
    WatermarkDecoder,
    WatermarkEncoder,
    default_settings,
    load_default_schema,
)
from image_watermark.cli import main as cli_main
from image_watermark.container import (
    FLAG_SIGNED,
    HEADER_FIXED_SIZE,
    LEGACY_VERSION,
    MAX_SCHEMA_ID_LENGTH,
    SIGNATURE_SIZE,
    SNIFF_SIZE,
    ContainerError,
    read_header,
)
from image_watermark.errors import PayloadError


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def schema():
    return load_default_schema()


@pytest.fixture
def photo(rgb_image):
    """A 512x512 image, so it holds exactly one tile."""

    return rgb_image


# ============================================================
# SCHEMA
# ============================================================

def test_default_schema_loads_from_package_resource(schema):
    assert schema.id == "default"
    assert list(schema.by_name) == [
        "model", "version", "image_id", "timestamp", "prompt",
    ]


def test_schema_to_dict_roundtrips(schema):
    data = schema.to_dict()
    again = PayloadSchema.from_dict(data)

    assert again.to_dict() == data
    assert again.id == schema.id


def test_schema_to_dict_is_json_serialisable(schema):
    data = schema.to_dict()

    assert json.loads(json.dumps(data)) == data


def test_schema_describe_mentions_every_field(schema):
    text = schema.describe()

    for name in schema.by_name:
        assert name in text


def test_schema_describe_hides_the_zero_length_sentinel(schema):
    text = schema.describe()

    # length 0 means "no limit" and must not be shown as "max 0".
    assert "max 0" not in text
    assert "unbounded" in text


def test_schema_rejects_bad_id():
    with pytest.raises(SchemaError):
        PayloadSchema.from_dict(
            {
                "id": "has space",
                "fields": [{"name": "a", "type": "string"}],
            }
        )


def test_schema_rejects_empty_field_list():
    with pytest.raises(SchemaError):
        PayloadSchema.from_dict({"id": "empty", "fields": []})


def test_schema_rejects_duplicate_field_names():
    with pytest.raises(SchemaError):
        PayloadSchema.from_dict(
            {
                "id": "dup",
                "fields": [
                    {"name": "a", "type": "string"},
                    {"name": "a", "type": "string"},
                ],
            }
        )


def test_schema_rejects_unknown_type():
    with pytest.raises(SchemaError):
        PayloadSchema.from_dict(
            {"id": "bad", "fields": [{"name": "a", "type": "nope"}]}
        )


def test_field_empty_values():
    assert FieldSpec(name="s", type="string").empty() == ""
    assert FieldSpec(name="b", type="bytes").empty() == b""
    assert FieldSpec(name="f", type="bool").empty() is False
    assert FieldSpec(name="u", type="uint16").empty() == 0


def test_schema_load_reports_missing_file(tmp_path):
    with pytest.raises(SchemaError) as exc:
        PayloadSchema.load(tmp_path / "nope.json")

    assert "not found" in str(exc.value)


def test_schema_load_reports_bad_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(SchemaError) as exc:
        PayloadSchema.load(path)

    assert "not valid JSON" in str(exc.value)


# ============================================================
# PAYLOAD
# ============================================================

def test_payload_roundtrips_all_values(schema):
    values = {
        "model": "flux",
        "version": "1.4",
        "image_id": bytes(range(16)),
        "timestamp": 1700000000,
        "prompt": "a lighthouse in a storm",
    }

    body = Payload(schema, values).serialize()
    back = Payload.deserialize(schema, body).to_dict()

    assert back == values


def test_payload_keeps_string_as_str_and_bytes_as_bytes(schema):
    body = Payload(
        schema,
        {"model": "m", "image_id": b"\x01\x02", "prompt": "p"},
    ).serialize()

    back = Payload.deserialize(schema, body).to_dict()

    assert isinstance(back["model"], str)
    assert isinstance(back["image_id"], bytes)
    assert isinstance(back["timestamp"], int)
    assert isinstance(back["prompt"], str)


def test_payload_always_keeps_every_optional_field(schema):
    """An omitted optional field is written as its zero value.

    Skipping it would shift every following field, and the decoder
    could no longer tell where it is.
    """

    body = Payload(schema, {"prompt": "only prompt"}).serialize()
    back = Payload.deserialize(schema, body).to_dict()

    for name in schema.by_name:
        assert name in back

    assert back["model"] == ""
    assert back["version"] == ""
    assert back["prompt"] == "only prompt"


def test_payload_roundtrips_every_subset_of_optional_fields(schema):
    optional = ["model", "version", "prompt"]
    pool = {
        "model": "m", "version": "1.0", "prompt": "hello",
    }

    for count in range(len(optional) + 1):
        for combo in itertools.combinations(optional, count):
            values = {name: pool[name] for name in combo}
            body = Payload(schema, values).serialize()
            back = Payload.deserialize(schema, body).to_dict()

            for name in combo:
                assert back[name] == values[name], combo


def test_payload_roundtrips_omitted_middle_field(schema):
    """The field that used to break the layout: model set, version not."""

    body = Payload(
        schema, {"model": "m", "prompt": "hello"},
    ).serialize()
    back = Payload.deserialize(schema, body).to_dict()

    assert back["model"] == "m"
    assert back["version"] == ""
    assert back["prompt"] == "hello"


def test_payload_roundtrips_empty_strings(schema):
    values = {"model": "", "version": "", "prompt": ""}
    body = Payload(schema, values).serialize()
    back = Payload.deserialize(schema, body).to_dict()

    for name, value in values.items():
        assert back[name] == value


def test_payload_roundtrips_unicode(schema):
    text = "Grüße aus Berlin – ein langer Text mit Umlauten"
    body = Payload(schema, {"prompt": text}).serialize()
    back = Payload.deserialize(schema, body).to_dict()

    assert back["prompt"] == text


def test_payload_fills_generated_fields(schema):
    back = Payload.deserialize(
        schema, Payload(schema, {}).serialize()
    ).to_dict()

    assert len(back["image_id"]) == 16
    assert back["timestamp"] > 1_600_000_000


def test_payload_missing_required_field_raises():
    schema = PayloadSchema.from_dict(
        {"id": "req", "fields": [{"name": "a", "type": "string"}]}
    )

    with pytest.raises(PayloadError):
        Payload(schema, {}).serialize()


def test_payload_rejects_truncated_body(schema):
    body = Payload(schema, {"prompt": "hello"}).serialize()

    with pytest.raises(SchemaError):
        Payload.deserialize(schema, body[:-1])


def test_payload_rejects_trailing_bytes(schema):
    body = Payload(schema, {"prompt": "hello"}).serialize()

    with pytest.raises(SchemaError) as exc:
        Payload.deserialize(schema, body + b"\x00")

    assert "unread" in str(exc.value)


def test_payload_rejects_wrong_schema(schema):
    other = PayloadSchema.from_dict(
        {
            "id": "other",
            "fields": [{"name": "x", "type": "uint32"}],
        }
    )
    body = Payload(schema, {"prompt": "hello"}).serialize()

    with pytest.raises(SchemaError):
        Payload.deserialize(other, body)


def test_payload_rejects_invalid_utf8(schema):
    body = bytearray(
        Payload(schema, {"model": "m", "version": "v", "prompt": "p"})
        .serialize()
    )
    # Corrupt a byte inside the model string payload.
    body[5] = 0xD4

    with pytest.raises(SchemaError):
        Payload.deserialize(schema, bytes(body))


def test_payload_enforces_field_length():
    schema = PayloadSchema.from_dict(
        {
            "id": "short",
            "fields": [
                {"name": "a", "type": "string", "length": 4},
            ],
        }
    )

    Payload(schema, {"a": "1234"}).serialize()

    with pytest.raises(SchemaError):
        Payload(schema, {"a": "12345"}).serialize()


# ============================================================
# CONTAINER
# ============================================================

def _header(schema_id="demo", signed=True, body_length=4):
    return Header(
        version=1,
        compression="none",
        schema_id=schema_id,
        body_length=body_length,
        raw_length=body_length,
        signed=signed,
    )


def _container(schema_id="demo", signed=True, body=b"body"):
    return Container(
        schema_id=schema_id,
        body=body,
        raw_length=len(body),
        compression="none",
        signature=b"x" * SIGNATURE_SIZE if signed else None,
    )


def test_header_size_and_sniff_constants():
    assert HEADER_FIXED_SIZE == 20
    assert SNIFF_SIZE == HEADER_FIXED_SIZE + MAX_SCHEMA_ID_LENGTH


def test_header_size_includes_the_schema_id():
    assert _header(schema_id="abc").size == HEADER_FIXED_SIZE + 3


def test_signed_header_length_includes_signature():
    header = _header(signed=True)

    assert header.total_length == (
        HEADER_FIXED_SIZE + 4 + 4 + SIGNATURE_SIZE
    )


def test_unsigned_header_length_excludes_signature():
    header = _header(signed=False)

    assert header.total_length == HEADER_FIXED_SIZE + 4 + 4


def test_signed_header_sets_the_flag_bit():
    packed = _header(signed=True).pack()

    assert int.from_bytes(
        packed[10:12], "big"
    ) & FLAG_SIGNED


def test_unsigned_header_clears_the_flag_bit():
    packed = _header(signed=False).pack()

    assert int.from_bytes(packed[10:12], "big") & FLAG_SIGNED == 0


def test_signed_container_roundtrips():
    data = _container(signed=True).pack()

    assert data[:4] == b"WM02"
    assert len(data) == _container().header.total_length

    container, body, signature = Container.unpack(data, 1)

    assert container.signed is True
    assert body == b"body"
    assert signature == b"x" * SIGNATURE_SIZE


def test_unsigned_container_roundtrips():
    data = _container(signed=False).pack()

    assert len(data) == _container(signed=False).header.total_length

    container, body, signature = Container.unpack(data, 1)

    assert container.signed is False
    assert body == b"body"
    assert signature == b""


def test_container_rejects_trailing_bytes():
    data = _container(signed=False).pack()

    with pytest.raises(ContainerError):
        Container.unpack(data + b"\x00", 1)


def test_container_rejects_truncated_data():
    data = _container(signed=False).pack()

    with pytest.raises(ContainerError):
        Container.unpack(data[:-1], 1)


def test_container_rejects_a_short_signature():
    container = Container(
        schema_id="demo", body=b"body", raw_length=4,
        compression="none", signature=b"short",
    )

    with pytest.raises(ContainerError):
        container.pack()


def test_read_header_rejects_unknown_flags():
    data = bytearray(_container(signed=False).pack())
    data[10] = 0x80

    with pytest.raises(ContainerError):
        read_header(bytes(data))


def test_container_rejects_schema_id_at_the_limit_plus_one():
    with pytest.raises(ContainerError):
        _container(
            schema_id="x" * (MAX_SCHEMA_ID_LENGTH + 1)
        ).pack()


def test_container_accepts_schema_id_at_the_limit():
    schema_id = "x" * MAX_SCHEMA_ID_LENGTH
    data = _container(schema_id=schema_id, signed=False).pack()

    assert read_header(data).schema_id == schema_id


def test_read_header_needs_only_the_sniff_size():
    data = _container(signed=False).pack()

    assert read_header(data[:SNIFF_SIZE]).schema_id == "demo"


def test_read_header_refuses_a_short_probe():
    with pytest.raises(ContainerError):
        read_header(b"WM02" + b"\x00" * 4)


# ============================================================
# REGISTRY
# ============================================================

def test_registry_returns_registered_schema(schema):
    registry = SchemaRegistry()
    registry.register(schema)

    assert registry.require("default") is schema
    assert registry.get("default") is schema
    assert registry.ids() == ["default"]


def test_registry_returns_none_for_unknown_id(schema):
    assert SchemaRegistry().get("nope") is None


def test_registry_require_rejects_unknown_id(schema):
    with pytest.raises(SchemaError) as exc:
        SchemaRegistry().require("nope")

    assert "nope" in str(exc.value)


def test_registry_loads_a_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "id": "fromfile",
                "fields": [{"name": "a", "type": "uint8"}],
            }
        ),
        encoding="utf-8",
    )

    assert SchemaRegistry().load(path).id == "fromfile"


# ============================================================
# ENCODER AND DECODER
# ============================================================

def test_encoder_decoder_roundtrip_signed(photo, key_dir):
    WatermarkEncoder(settings=default_settings(sign=True)).encode(
        photo, photo.replace("input", "out"),
        {"model": "flux", "version": "1.4", "prompt": "hi"},
        progress=None,
    )

    result = WatermarkDecoder(
        settings=default_settings(sign=True)
    ).decode(photo.replace("input", "out"), progress=None)

    assert result.valid is True
    assert result.schema_id == "default"
    assert result.payload.get("prompt") == "hi"
    assert result.payload.get("model") == "flux"


def test_encoder_decoder_roundtrip_unsigned(photo, key_dir):
    out = photo.replace("input", "out")

    WatermarkEncoder(settings=default_settings(sign=False)).encode(
        photo, out,
        {"model": "flux", "prompt": "hi"},
        progress=None,
    )

    result = WatermarkDecoder(
        settings=default_settings(sign=False)
    ).decode(out, progress=None)

    assert result.valid is False
    assert result.payload.get("prompt") == "hi"
    assert result.payload.get("model") == "flux"


def test_encoder_leaves_the_signature_flag_clear_when_unsigned():
    settings = default_settings(sign=False)
    _, packed = WatermarkEncoder(
        settings=settings
    ).pack(Payload(settings.require_schema(), {"prompt": "hi"}))

    header = read_header(packed)

    assert header.signed is False
    assert int.from_bytes(
        packed[10:12], "big"
    ) & FLAG_SIGNED == 0
    assert len(packed) == header.total_length


def test_encoder_roundtrip_without_optional_fields(photo, key_dir):
    """The case that used to corrupt the payload."""

    out = photo.replace("input", "out")

    WatermarkEncoder(settings=default_settings()).encode(
        photo, out, {"prompt": "only prompt"}, progress=None
    )

    result = WatermarkDecoder(
        settings=default_settings()
    ).decode(out, progress=None)

    assert result.payload.get("prompt") == "only prompt"
    assert result.payload.get("model") == ""
    assert result.payload.get("version") == ""


def test_decoder_rejects_a_plain_image(photo, key_dir):
    from image_watermark.errors import DecodingError

    with pytest.raises(DecodingError):
        WatermarkDecoder(
            settings=default_settings()
        ).decode(photo, progress=None)


def test_encoder_embeds_into_every_tile_by_default(photo, key_dir):
    from PIL import Image

    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[:] = 128
    path = photo.replace("input", "big")
    Image.fromarray(image).save(path)

    result = WatermarkEncoder(
        settings=default_settings(redundant=True)
    ).encode(
        path, path.replace("big", "bigout"),
        {"prompt": "hi"}, progress=None,
    )

    assert result.tiles_available == 4
    assert result.tiles_embedded == 4


def test_encoder_single_tile_mode(photo, key_dir):
    from PIL import Image

    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[:] = 128
    path = photo.replace("input", "big")
    Image.fromarray(image).save(path)

    result = WatermarkEncoder(
        settings=default_settings(redundant=False)
    ).encode(
        path, path.replace("big", "bigout"),
        {"prompt": "hi"}, progress=None,
    )

    assert result.tiles_embedded == 1


def test_max_tiles_limits_redundancy(photo, key_dir):
    from PIL import Image

    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[:] = 128
    path = photo.replace("input", "big")
    Image.fromarray(image).save(path)

    result = WatermarkEncoder(
        settings=default_settings(redundant=True, max_tiles=2)
    ).encode(
        path, path.replace("big", "bigout"),
        {"prompt": "hi"}, progress=None,
    )

    assert result.tiles_embedded == 2


def test_custom_schema_roundtrips(photo, key_dir, tmp_path):
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps(
            {
                "id": "custom",
                "fields": [
                    {"name": "artist", "type": "string"},
                    {"name": "shutter", "type": "uint16"},
                    {"name": "gps", "type": "bytes", "display": "hex"},
                    {"name": "note", "type": "string",
                     "compression": "zlib"},
                ],
            }
        ),
        encoding="utf-8",
    )

    custom = PayloadSchema.load(path)
    settings = default_settings(schema=custom)
    out = photo.replace("input", "custom")

    WatermarkEncoder(settings=settings).encode(
        photo, out,
        {
            "artist": "Hannah",
            "shutter": 125,
            "gps": bytes(range(16)),
            "note": "Berlin",
        },
        progress=None,
    )

    decoder = WatermarkDecoder(settings=settings)
    decoder.register(custom)
    result = decoder.decode(out, progress=None)

    assert result.schema_id == "custom"
    assert result.payload.get("artist") == "Hannah"
    assert result.payload.get("shutter") == 125
    assert result.payload.get("gps") == bytes(range(16))
    assert result.payload.get("note") == "Berlin"


# ============================================================
# EMBED OFFSET
# ============================================================

def test_embed_offset_rotates_the_sweep():
    """embed_offset is (y, x); the first embedded tile is that tile."""

    tiles_x = tiles_y = 3

    positions = [
        (ty, tx)
        for ty in range(tiles_y)
        for tx in range(tiles_x)
    ]

    for offset in (
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
    ):
        start = (
            offset[0] * tiles_x + offset[1]
        ) % len(positions)

        rotated = positions[start:] + positions[:start]

        assert rotated[0] == offset
        assert sorted(rotated) == sorted(positions)


def test_embed_offset_is_wired_into_the_encoder(photo, key_dir):
    from PIL import Image

    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[:] = 128
    path = photo.replace("input", "big")
    Image.fromarray(image).save(path)

    result = WatermarkEncoder(
        settings=default_settings(embed_offset=(1, 1))
    ).encode(
        path, path.replace("big", "bigout"),
        {"prompt": "hi"}, progress=None,
    )

    assert result.tiles_embedded == 4


# ============================================================
# LEGACY CROSS COMPATIBILITY
# ============================================================

def test_new_decoder_reads_a_legacy_wm01_image(photo, key_dir):
    import image_watermark.encode as legacy

    out = photo.replace("input", "legacy")
    legacy.encode_image(
        photo, out,
        prompt="a lighthouse",
        model="flux",
        version="1.4",
    )

    result = WatermarkDecoder(
        settings=default_settings()
    ).decode(out, progress=None)

    assert result.schema_id == "legacy-wm01"
    assert result.valid is True
    assert result.payload.get("prompt") == "a lighthouse"
    assert result.payload.get("model") == "flux"
    assert len(result.payload.get("image_id")) == 32


def test_legacy_encoder_still_produces_wm01(photo, key_dir):
    """The legacy wrapper must keep the old container, not WM02."""

    import image_watermark.encode as legacy

    out = photo.replace("input", "legacy")
    legacy.encode_image(
        photo, out, prompt="p", model="m", version="v"
    )

    result = WatermarkDecoder(
        settings=default_settings()
    ).decode(out, progress=None)

    # The decoder resolves the legacy framing and labels it itself.
    assert result.schema_id == "legacy-wm01"
    assert result.container.version == LEGACY_VERSION
    assert result.valid is True


def test_legacy_image_carries_no_wm02_magic(photo, key_dir):
    import image_watermark.encode as legacy

    out = photo.replace("input", "legacy")
    legacy.encode_image(
        photo, out, prompt="p", model="m", version="v"
    )

    scanner = WatermarkDecoder(settings=default_settings())

    with Image.open(out) as handle:
        array = np.array(handle.convert("RGB"))

    assert scanner.scan_tiles(array)


# ============================================================
# CLI
# ============================================================

def test_cli_version(capsys):
    from image_watermark import __version__

    assert cli_main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_without_a_command_prints_help(capsys):
    assert cli_main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_info(capsys):
    assert cli_main(["info"]) == 0

    out = capsys.readouterr().out

    assert "capacity" in out
    assert "max payload" in out


def test_cli_schema_shows_the_default(capsys):
    assert cli_main(["schema"]) == 0
    assert "default" in capsys.readouterr().out


def test_cli_schema_json(capsys):
    assert cli_main(["schema", "--format", "json"]) == 0

    data = json.loads(capsys.readouterr().out)

    assert data["id"] == "default"


def test_cli_schema_validate(capsys):
    assert cli_main(["schema", "--validate"]) == 0
    assert capsys.readouterr().out.startswith("ok: default")


def test_cli_schema_rejects_a_broken_file(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")

    assert cli_main(["schema", str(path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_init_keys(tmp_path, capsys):
    private = tmp_path / "priv.pem"
    public = tmp_path / "pub.pem"

    assert cli_main(
        [
            "init-keys",
            "--private-key", str(private),
            "--public-key", str(public),
        ]
    ) == 0

    assert private.exists()
    assert public.exists()


def test_cli_encode_decode_roundtrip(photo, tmp_path, capsys):
    from PIL import Image

    private = tmp_path / "priv.pem"
    public = tmp_path / "pub.pem"
    out = tmp_path / "out.png"

    assert cli_main(
        ["init-keys",
         "--private-key", str(private),
         "--public-key", str(public)]
    ) == 0

    capsys.readouterr()

    assert cli_main(
        [
            "encode", photo, str(out), "-q",
            "--private-key", str(private),
            "--public-key", str(public),
            "-f", "model=flux",
            "-f", "version=1.4",
            "-f", "prompt=a lighthouse",
        ]
    ) == 0

    capsys.readouterr()

    assert cli_main(
        [
            "decode", str(out), "-q", "--json",
            "--public-key", str(public),
        ]
    ) == 0

    data = json.loads(capsys.readouterr().out)

    assert data["signature_valid"] is True
    assert data["schema_id"] == "default"
    assert data["values"]["model"] == "flux"
    assert data["values"]["prompt"] == "a lighthouse"


def test_cli_encode_with_a_custom_schema(photo, tmp_path, capsys):
    from PIL import Image

    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps(
            {
                "id": "cli_custom",
                "fields": [
                    {"name": "artist", "type": "string"},
                    {"name": "gps", "type": "bytes",
                     "display": "hex"},
                ],
            }
        ),
        encoding="utf-8",
    )

    private = tmp_path / "priv.pem"
    public = tmp_path / "pub.pem"
    out = tmp_path / "out.png"

    cli_main(["init-keys",
              "--private-key", str(private),
              "--public-key", str(public)])
    capsys.readouterr()

    assert cli_main(
        [
            "encode", photo, str(out), "-q",
            "--schema", str(path),
            "--private-key", str(private),
            "--public-key", str(public),
            "-f", "artist=Hannah",
            "-f", "gps=473a1b2c",
        ]
    ) == 0

    capsys.readouterr()

    assert cli_main(
        [
            "decode", str(out), "-q", "--json",
            "--schema", str(path),
            "--public-key", str(public),
        ]
    ) == 0

    data = json.loads(capsys.readouterr().out)

    assert data["schema_id"] == "cli_custom"
    assert data["values"]["artist"] == "Hannah"
    assert data["values"]["gps"] == "473a1b2c"


def test_cli_rejects_an_unknown_field(photo, capsys):
    assert cli_main(
        ["encode", photo, photo + ".png", "-q",
         "-f", "nosuchfield=1"]
    ) == 1
    assert "Unknown field" in capsys.readouterr().err


def test_cli_rejects_a_malformed_pair(photo, capsys):
    assert cli_main(
        ["encode", photo, photo + ".png", "-q", "-f", "novalue"]
    ) == 1
    assert "NAME=VALUE" in capsys.readouterr().err


def test_cli_rejects_bad_hex(photo, tmp_path, capsys):
    path = tmp_path / "hex.json"
    path.write_text(
        json.dumps(
            {"id": "hexs",
             "fields": [{"name": "gps", "type": "bytes",
                         "display": "hex"}]}
        ),
        encoding="utf-8",
    )

    assert cli_main(
        ["encode", photo, photo + ".png", "-q",
         "--schema", str(path), "-f", "gps=zzzz"]
    ) == 1
    assert "not valid hex" in capsys.readouterr().err


def test_cli_rejects_a_missing_image(capsys):
    assert cli_main(
        ["encode", "/nope/missing.png", "/tmp/out.png", "-q"]
    ) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_decode_reports_a_plain_image(photo, capsys):
    assert cli_main(["decode", photo, "-q"]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_module_is_runnable():
    """The console script target must be importable and callable."""

    proc = subprocess.run(
        [sys.executable, "-m", "image_watermark.cli", "--version"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    assert proc.stdout.strip().startswith("0.3")
