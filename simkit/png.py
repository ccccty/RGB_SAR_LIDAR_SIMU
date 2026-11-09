"""Minimal dependency-free PNG writer (8/16-bit gray, 8-bit RGB)."""
import struct
import zlib

import numpy as np


def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


#: Fixed English tEXt keywords, in a fixed order so the bytes are reproducible.
#: width/height/bit_depth/color_type are filled in by _write_png itself.
_TEXT_KEYS = ("modality", "width", "height", "bit_depth", "color_type",
              "gsd_m_per_px", "fov_deg", "range_m",
              "radiometric_calibration", "grid")


def _format(value):
    """Deterministic text for one metadata value (no timestamps / paths / seeds)."""
    if isinstance(value, bytes):
        return value.decode("latin-1")
    if isinstance(value, float):
        return "%.6g" % value
    return str(value)


def _latin1(text):
    """tEXt values are Latin-1; anything else is ASCII-escaped deterministically."""
    if not isinstance(text, str):
        text = str(text)
    try:
        return text.encode("latin-1")
    except UnicodeEncodeError:
        return text.encode("ascii", "backslashreplace")


def _text_entries(meta, w, h, bit_depth, color_type):
    """Ordered (keyword, value) pairs; missing optional keys are simply skipped."""
    values = dict(width=w, height=h, bit_depth=bit_depth, color_type=color_type)
    for key in _TEXT_KEYS:
        if key not in values and meta.get(key) is not None:
            values[key] = meta[key]
    return [(key, _format(values[key])) for key in _TEXT_KEYS if key in values]


def _pixels_per_meter(meta):
    """pHYs density: explicit pixels_per_meter wins, else derive from GSD."""
    ppm = meta.get("pixels_per_meter")
    if ppm is not None:
        return ppm
    gsd = meta.get("gsd_m_per_px")
    if gsd is None:
        return None
    try:
        gsd = float(gsd)
    except (TypeError, ValueError):
        return None
    return 1.0 / gsd if gsd > 0 else None


def _write_png(path, arr, color_type, bit_depth, nch, meta=None):
    h, w = arr.shape[0], arr.shape[1]
    if bit_depth == 16:
        rows = arr.astype(">u2").reshape(h, w * nch).view(np.uint8).reshape(h, -1)
    else:
        rows = arr.astype(np.uint8).reshape(h, w * nch)
    raw = bytearray()
    for i in range(h):
        raw.append(0)  # filter type 0
        raw += rows[i].tobytes()
    out = b"\x89PNG\r\n\x1a\n"
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bit_depth, color_type, 0, 0, 0))
    if meta is not None:
        ppm = _pixels_per_meter(meta)
        if ppm is not None:
            if isinstance(ppm, (tuple, list)):
                x_ppu, y_ppu = int(ppm[0]), int(ppm[1])
            else:
                x_ppu = y_ppu = int(ppm)
            # unit = 1 -> metre; pHYs must sit between IHDR and IDAT.
            out += _chunk(b"pHYs", struct.pack(">IIB", x_ppu, y_ppu, 1))
    out += _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    if meta is not None:
        for keyword, value in _text_entries(meta, w, h, bit_depth, color_type):
            out += _chunk(b"tEXt", keyword.encode("ascii") + b"\0" + _latin1(value))
    out += _chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(out)


def write_gray8(path, a, meta=None):
    _write_png(path, np.clip(a, 0, 255).astype(np.uint8), 0, 8, 1, meta=meta)


def write_gray16(path, a, meta=None):
    _write_png(path, np.clip(a, 0, 65535).astype(np.uint16), 0, 16, 1, meta=meta)


def write_rgb8(path, a, meta=None):
    _write_png(path, np.clip(a, 0, 255).astype(np.uint8), 2, 8, 3, meta=meta)


def read_meta(path):
    """Parse the pHYs/tEXt chunks written by write_* and return them as a dict.

    tEXt values come back as strings; pHYs yields ``pixels_per_meter`` = [x, y]
    and ``pixels_per_meter_unit`` (1 == metre). Unknown chunks are skipped.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    out = {}
    pos = 8  # PNG signature
    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        end = pos + 8 + length
        if end + 4 > len(data):
            break
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:end]
        pos = end + 4  # skip the CRC
        if tag == b"pHYs" and len(body) >= 9:
            x, y, unit = struct.unpack(">IIB", body[:9])
            out["pixels_per_meter"] = [x, y]
            out["pixels_per_meter_unit"] = unit
        elif tag == b"tEXt":
            keyword, sep, value = body.partition(b"\0")
            if sep:
                out[keyword.decode("latin-1")] = value.decode("latin-1")
        elif tag == b"IEND":
            break
    return out


#: Downstream readers may prefer the longer name.
read_text_meta = read_meta
