"""Minimal dependency-free PNG writer (8/16-bit gray, 8-bit RGB)."""
import struct
import zlib

import numpy as np


def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _write_png(path, arr, color_type, bit_depth, nch):
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
    out += _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    out += _chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(out)


def write_gray8(path, a):
    _write_png(path, np.clip(a, 0, 255).astype(np.uint8), 0, 8, 1)


def write_gray16(path, a):
    _write_png(path, np.clip(a, 0, 65535).astype(np.uint16), 0, 16, 1)


def write_rgb8(path, a):
    _write_png(path, np.clip(a, 0, 255).astype(np.uint8), 2, 8, 3)
