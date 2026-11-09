# -*- coding: utf-8 -*-
"""PNG pHYs/tEXt metadata: byte-for-byte backward compatibility and readback.

The hard contract: passing ``meta=None`` (or omitting it) must produce exactly
the legacy three-chunk PNG (IHDR/IDAT/IEND), because several tests compare
read_bytes() across runs. Passing a dict adds pHYs (before IDAT) and tEXt
(after IDAT) chunks whose bytes are fully deterministic.
"""
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from simkit import png


def _legacy_bytes(arr, color_type, bit_depth, nch):
    """Reproduce the pre-metadata writer byte for byte, independently."""
    h, w = arr.shape[0], arr.shape[1]
    if bit_depth == 16:
        rows = arr.astype(">u2").reshape(h, w * nch).view(np.uint8).reshape(h, -1)
    else:
        rows = arr.astype(np.uint8).reshape(h, w * nch)
    raw = bytearray()
    for i in range(h):
        raw.append(0)
        raw += rows[i].tobytes()

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bit_depth,
                                         color_type, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))


def _chunks(path):
    data = Path(path).read_bytes()
    out, pos = [], 8
    while pos + 12 <= len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        end = pos + 8 + n
        if end + 4 > len(data):
            break
        out.append((data[pos + 4:pos + 8], data[pos + 8:end],
                    struct.unpack(">I", data[end:end + 4])[0],
                    data[pos + 4:end]))
        pos = end + 4
    return out


FULL_META = dict(
    pixels_per_meter=(100, 200),
    gsd_m_per_px=0.05,
    fov_deg=30.0,
    range_m=1500.0,
    modality="sar",
    radiometric_calibration="gamma0 product; 8-bit display stretch",
    grid="1024x256",
)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_meta_none_is_byte_for_byte_the_legacy_writer(self):
        cases = [
            (png.write_gray8, np.arange(12, dtype=np.uint8).reshape(3, 4) * 17,
             0, 8, 1),
            (png.write_gray16, np.arange(12, dtype=np.uint16).reshape(3, 4) * 1000,
             0, 16, 1),
            (png.write_rgb8, np.arange(18, dtype=np.uint8).reshape(3, 2, 3) * 11,
             2, 8, 3),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for i, (fn, arr, ct, bd, nch) in enumerate(cases):
                expected = _legacy_bytes(arr, ct, bd, nch)
                plain = Path(tmp) / ("plain%d.png" % i)
                explicit = Path(tmp) / ("explicit%d.png" % i)
                fn(str(plain), arr)
                fn(str(explicit), arr, meta=None)
                self.assertEqual(plain.read_bytes(), expected, fn.__name__)
                self.assertEqual(explicit.read_bytes(), expected, fn.__name__)
                # Exactly three chunks, in the legacy order.
                self.assertEqual([c[0] for c in _chunks(plain)],
                                 [b"IHDR", b"IDAT", b"IEND"])

    def test_meta_changes_the_bytes(self):
        arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.png", Path(tmp) / "b.png"
            png.write_gray8(str(a), arr)
            png.write_gray8(str(b), arr, meta=dict(modality="ir"))
            self.assertNotEqual(a.read_bytes(), b.read_bytes())


class MetadataLayoutTests(unittest.TestCase):
    def test_chunk_order_and_crc(self):
        arr = np.arange(24, dtype=np.uint16).reshape(4, 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.png"
            png.write_gray16(str(path), arr, meta=FULL_META)
            chunks = _chunks(path)
            tags = [c[0] for c in chunks]
            self.assertEqual(tags[0], b"IHDR")
            self.assertEqual(tags[1], b"pHYs")
            self.assertIn(b"IDAT", tags)
            self.assertEqual(tags[-1], b"IEND")
            idat = tags.index(b"IDAT")
            self.assertTrue(all(t == b"tEXt" for t in tags[idat + 1:-1]))
            self.assertGreater(len(tags), idat + 2)
            # Every chunk CRC is over tag+data and is correct.
            for tag, _data, crc, tag_data in chunks:
                self.assertEqual(crc, zlib.crc32(tag_data) & 0xFFFFFFFF)

    def test_phys_is_nine_bytes_and_metre_unit(self):
        arr = np.zeros((2, 2), np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.png"
            png.write_gray8(str(path), arr, meta=dict(pixels_per_meter=(100, 200)))
            phys = [c for c in _chunks(path) if c[0] == b"pHYs"]
            self.assertEqual(len(phys), 1)
            self.assertEqual(len(phys[0][1]), 9)
            self.assertEqual(struct.unpack(">IIB", phys[0][1]), (100, 200, 1))

    def test_scalar_pixels_per_meter(self):
        arr = np.zeros((2, 2), np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.png"
            png.write_gray8(str(path), arr, meta=dict(pixels_per_meter=250))
            self.assertEqual(png.read_meta(str(path))["pixels_per_meter"], [250, 250])

    def test_gsd_derives_a_phys_chunk(self):
        arr = np.zeros((2, 2), np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.png"
            png.write_gray8(str(path), arr, meta=dict(gsd_m_per_px=0.05))
            self.assertEqual(png.read_meta(str(path))["pixels_per_meter"], [20, 20])

    def test_no_phys_when_no_density_key(self):
        arr = np.zeros((2, 2), np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.png"
            png.write_gray8(str(path), arr, meta=dict(modality="ir"))
            self.assertNotIn("pixels_per_meter", png.read_meta(str(path)))
            self.assertNotIn(b"pHYs", [c[0] for c in _chunks(path)])


class ReadbackTests(unittest.TestCase):
    def test_read_meta_returns_all_written_entries(self):
        arr = np.zeros((7, 9), np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.png"
            png.write_gray8(str(path), arr, meta=FULL_META)
            got = png.read_meta(str(path))
        self.assertEqual(got["pixels_per_meter"], [100, 200])
        self.assertEqual(got["pixels_per_meter_unit"], 1)
        self.assertEqual(got["modality"], "sar")
        self.assertEqual(got["grid"], "1024x256")
        self.assertEqual(got["radiometric_calibration"],
                         "gamma0 product; 8-bit display stretch")
        self.assertEqual(float(got["gsd_m_per_px"]), 0.05)
        self.assertEqual(float(got["fov_deg"]), 30.0)
        self.assertEqual(float(got["range_m"]), 1500.0)
        # width/height/bit_depth/color_type come from the writer itself.
        self.assertEqual(got["width"], "9")
        self.assertEqual(got["height"], "7")
        self.assertEqual(got["bit_depth"], "8")
        self.assertEqual(got["color_type"], "0")

    def test_all_three_writers_report_their_own_depth_and_colour_type(self):
        gray8 = np.zeros((3, 5), np.uint8)
        gray16 = np.zeros((3, 5), np.uint16)
        rgb8 = np.zeros((3, 5, 3), np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            cases = [(png.write_gray8, gray8, "8", "0"),
                     (png.write_gray16, gray16, "16", "0"),
                     (png.write_rgb8, rgb8, "8", "2")]
            for i, (fn, arr, bd, ct) in enumerate(cases):
                path = Path(tmp) / ("%d.png" % i)
                fn(str(path), arr, meta=dict(modality="test"))
                got = png.read_meta(str(path))
                self.assertEqual(got["bit_depth"], bd)
                self.assertEqual(got["color_type"], ct)
                self.assertEqual(got["width"], "5")
                self.assertEqual(got["height"], "3")

    def test_optional_keys_are_skipped_when_absent(self):
        arr = np.zeros((2, 2), np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.png"
            png.write_gray8(str(path), arr, meta=dict(modality="ir", grid="4x2"))
            got = png.read_meta(str(path))
        self.assertEqual(got["modality"], "ir")
        self.assertEqual(got["grid"], "4x2")
        for key in ("gsd_m_per_px", "fov_deg", "range_m",
                    "radiometric_calibration", "pixels_per_meter"):
            self.assertNotIn(key, got)

    def test_read_text_meta_alias(self):
        self.assertIs(png.read_text_meta, png.read_meta)


class DeterminismTests(unittest.TestCase):
    def test_two_writes_with_same_meta_are_byte_equal(self):
        arr = np.arange(40, dtype=np.uint8).reshape(5, 8)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.png", Path(tmp) / "b.png"
            png.write_gray8(str(a), arr, meta=FULL_META)
            png.write_gray8(str(b), arr, meta=dict(FULL_META))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(png.read_meta(str(a)), png.read_meta(str(b)))

    def test_non_latin1_value_is_safely_transliterated_and_stable(self):
        meta = dict(modality="ir", radiometric_calibration="gamma0 产品 / 8-bit 拉伸")
        arr = np.zeros((2, 2), np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.png", Path(tmp) / "b.png"
            png.write_gray16(str(a), arr, meta=meta)
            png.write_gray16(str(b), arr, meta=dict(meta))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            value = png.read_meta(str(a))["radiometric_calibration"]
            # tEXt is Latin-1: the escaped text must be pure ASCII.
            value.encode("ascii")
            self.assertIn("\\u", value)

    def test_keyword_is_english_and_has_no_null(self):
        arr = np.zeros((2, 2), np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.png"
            png.write_gray8(str(path), arr, meta=dict(modality="ir", grid="2x2"))
            texts = [c[1] for c in _chunks(path) if c[0] == b"tEXt"]
        for body in texts:
            keyword, sep, _ = body.partition(b"\0")
            self.assertEqual(sep, b"\0")
            self.assertTrue(1 <= len(keyword) <= 79)
            keyword.decode("latin-1")


if __name__ == "__main__":
    unittest.main()
