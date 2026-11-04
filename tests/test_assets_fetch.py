"""Offline regression tests for simkit.assets_fetch.

No test touches the network: a FakeFetcher implements the two-method seam and
every cache directory points into a TemporaryDirectory.
"""
import hashlib
import io
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock
from urllib.error import URLError

from simkit import assets_fetch

HDR_BYTES = (
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 4\n" + bytes(range(64))
)


def download_url(asset_id, size):
    return "https://cdn.example.test/%s_%s.hdr" % (asset_id, size)


def hdri_files_payload(asset_id, size, payload, declared_size=None):
    return {"hdri": {size: {"hdr": {
        "url": download_url(asset_id, size),
        "size": len(payload) if declared_size is None else declared_size,
    }}}}


def assets_payload(**download_counts):
    return {
        asset_id: {"name": asset_id.replace("_", " ").title(),
                   "download_count": count, "categories": []}
        for asset_id, count in download_counts.items()
    }


def material_zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class FakeFetcher:
    """Two-method fetcher double; never opens a socket."""

    def __init__(self, json_map=None, downloads=None, json_error=None, download_error=None):
        self.json_map = json_map or {}
        self.downloads = downloads or {}
        self.json_error = json_error
        self.download_error = download_error
        self.calls = []

    def get_json(self, url):
        self.calls.append(("json", url))
        if self.json_error is not None:
            raise self.json_error
        if url not in self.json_map:
            raise URLError("unexpected JSON URL %s" % url)
        return self.json_map[url]

    def download(self, url, fileobj, *, max_bytes):
        self.calls.append(("download", url))
        if self.download_error is not None:
            raise self.download_error
        payload = self.downloads.get(url)
        if payload is None:
            raise URLError("unexpected download URL %s" % url)
        for offset in range(0, len(payload), 97):
            fileobj.write(payload[offset:offset + 97])
        return len(payload)


class ExplodingFetcher:
    """Fails the test if any network seam is used."""

    def __init__(self):
        self.calls = []

    def get_json(self, url):
        self.calls.append(("json", url))
        raise AssertionError("network access attempted: %s" % url)

    def download(self, url, fileobj, *, max_bytes):
        self.calls.append(("download", url))
        raise AssertionError("network access attempted: %s" % url)


class AssetsFetchTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.hdri_dir = root/"hdri"
        self.material_dir = root/"materials"
        self.catalog_path = root/"catalog.json"
        self._patches = [
            mock.patch.object(assets_fetch, "HDRI_DIR", str(self.hdri_dir)),
            mock.patch.object(assets_fetch, "MATERIAL_DIR", str(self.material_dir)),
            mock.patch.object(assets_fetch, "CATALOG_PATH", str(self.catalog_path)),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        self._temp.cleanup()

    def write_hdri(self, asset_id, size="2k", payload=HDR_BYTES):
        self.hdri_dir.mkdir(parents=True, exist_ok=True)
        path = self.hdri_dir/("%s_%s.hdr" % (asset_id, size))
        path.write_bytes(payload)
        return path

    def fake_weather(self, category, asset_id, count=10, size="2k"):
        fetcher = FakeFetcher(
            json_map={
                assets_fetch._assets_url(category): assets_payload(**{asset_id: count}),
                assets_fetch._files_url(asset_id): hdri_files_payload(asset_id, size, HDR_BYTES),
            },
            downloads={download_url(asset_id, size): HDR_BYTES})
        return fetcher

    # ---------------------------------------------------------------- weather mapping

    def test_clear_weather_picks_pure_skies_and_best_downloaded(self):
        fetcher = FakeFetcher(
            json_map={
                assets_fetch._assets_url("pure skies"):
                    assets_payload(pure_plain=5, pure_best=99),
                assets_fetch._files_url("pure_best"):
                    hdri_files_payload("pure_best", "2k", HDR_BYTES),
            },
            downloads={download_url("pure_best", "2k"): HDR_BYTES})

        result = assets_fetch.hdri_for_weather("clear", "2k", fetcher=fetcher)

        self.assertEqual(result["id"], "pure_best")
        self.assertEqual(result["source"], "polyhaven")
        self.assertEqual(result["license"], "CC0")
        self.assertFalse(result["cached"])
        self.assertTrue(Path(result["path"]).is_file())
        self.assertEqual(Path(result["path"]).read_bytes(), HDR_BYTES)
        self.assertEqual(fetcher.calls[0],
                         ("json", assets_fetch._assets_url("pure skies")))

        # The catalog now satisfies the same weather request without any network.
        offline = assets_fetch.hdri_for_weather("clear", "2k", fetcher=ExplodingFetcher())
        self.assertTrue(offline["cached"])
        self.assertEqual(offline["path"], result["path"])

    def test_cloudy_weather_prefers_partly_cloudy(self):
        fetcher = FakeFetcher(
            json_map={
                assets_fetch._assets_url("partly cloudy"):
                    assets_payload(partly_sky=20),
                assets_fetch._files_url("partly_sky"):
                    hdri_files_payload("partly_sky", "2k", HDR_BYTES),
            },
            downloads={download_url("partly_sky", "2k"): HDR_BYTES})

        result = assets_fetch.hdri_for_weather("cloudy", fetcher=fetcher)

        self.assertEqual(result["id"], "partly_sky")
        self.assertIn(("json", assets_fetch._assets_url("partly cloudy")), fetcher.calls)

    def test_rain_and_fog_map_to_overcast(self):
        for weather, asset_id in (("rain", "rain_overcast"), ("fog", "fog_overcast")):
            with self.subTest(weather=weather):
                # Rain and fog share the overcast cache, so start each case clean.
                if self.hdri_dir.exists():
                    for cached_file in self.hdri_dir.glob("*"):
                        cached_file.unlink()
                if self.catalog_path.exists():
                    self.catalog_path.unlink()
                fetcher = self.fake_weather("overcast", asset_id)
                result = assets_fetch.hdri_for_weather(weather, fetcher=fetcher)
                self.assertEqual(result["id"], asset_id)
                self.assertEqual(fetcher.calls[0],
                                 ("json", assets_fetch._assets_url("overcast")))

    def test_overcast_cache_is_shared_by_rain_and_fog(self):
        fetcher = self.fake_weather("overcast", "shared_overcast")
        first = assets_fetch.hdri_for_weather("rain", fetcher=fetcher)

        second = assets_fetch.hdri_for_weather("fog", fetcher=ExplodingFetcher())

        self.assertTrue(second["cached"])
        self.assertEqual(second["path"], first["path"])

    def test_unknown_weather_returns_none_without_network(self):
        fetcher = ExplodingFetcher()
        for weather in ("snow", "mixed", "evolve", None, ""):
            with self.subTest(weather=weather):
                self.assertIsNone(assets_fetch.hdri_for_weather(weather, fetcher=fetcher))
        self.assertEqual(fetcher.calls, [])

    def test_weather_accepts_studio_preset_dict(self):
        fetcher = self.fake_weather("overcast", "dict_sky")
        result = assets_fetch.hdri_for_weather({"mode": "fixed", "preset": "rain"},
                                               fetcher=fetcher)
        self.assertEqual(result["id"], "dict_sky")

    # ---------------------------------------------------------------- cache hits

    def test_catalog_weather_cache_hit_does_not_network(self):
        payload = HDR_BYTES
        path = self.write_hdri("cached_sky")
        assets_fetch.write_catalog({
            "version": 1,
            "hdri": {path.name: {
                "id": "cached_sky", "size": "2k", "source": "polyhaven",
                "license": "CC0", "weather": ["cloudy"],
                "downloaded": "2026-01-01T00:00:00Z",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }},
            "materials": {},
        })
        fetcher = ExplodingFetcher()

        result = assets_fetch.hdri_for_weather("cloudy", "2k", fetcher=fetcher)

        self.assertTrue(result["cached"])
        self.assertEqual(result["path"], str(path.resolve()))
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(fetcher.calls, [])

    def test_download_hdri_cache_hit_does_not_network(self):
        path = self.write_hdri("known_asset")
        fetcher = ExplodingFetcher()

        result = assets_fetch.download_hdri("known_asset", "2k", fetcher=fetcher)

        self.assertEqual(result, str(path.resolve()))
        self.assertEqual(fetcher.calls, [])

    def test_plain_id_filename_is_also_a_cache_hit(self):
        self.hdri_dir.mkdir(parents=True, exist_ok=True)
        path = self.hdri_dir/"legacy_asset.hdr"
        path.write_bytes(HDR_BYTES)
        fetcher = ExplodingFetcher()

        result = assets_fetch.download_hdri("legacy_asset", "2k", fetcher=fetcher)

        self.assertEqual(result, str(path.resolve()))
        self.assertEqual(fetcher.calls, [])

    def test_list_cached_reads_files_even_without_catalog(self):
        payload = HDR_BYTES + b"extra"
        path = self.write_hdri("manual_asset", payload=payload)

        entries = assets_fetch.list_cached()

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "hdri")
        self.assertEqual(entries[0]["id"], "manual_asset")
        self.assertEqual(entries[0]["size"], "2k")
        self.assertEqual(entries[0]["path"], str(path.resolve()))
        self.assertEqual(entries[0]["bytes"], len(payload))
        self.assertEqual(entries[0]["source"], "polyhaven")
        self.assertEqual(entries[0]["license"], "CC0")

    # ---------------------------------------------------------------- failures

    def test_download_failure_raises_assets_unavailable_and_leaves_no_partial(self):
        size = "2k"
        asset_id = "broken_asset"
        url = download_url(asset_id, size)
        fetcher = FakeFetcher(
            json_map={assets_fetch._files_url(asset_id):
                      hdri_files_payload(asset_id, size, HDR_BYTES)},
            downloads={url: HDR_BYTES},
            download_error=URLError("offline"))

        with self.assertRaises(assets_fetch.AssetsUnavailable) as raised:
            assets_fetch.download_hdri(asset_id, size, fetcher=fetcher)

        self.assertIn("offline", str(raised.exception))
        self.assertFalse((self.hdri_dir/("%s_2k.hdr" % asset_id)).exists())
        self.assertEqual(list(self.hdri_dir.glob("*")), [])

    def test_json_failure_is_wrapped(self):
        fetcher = FakeFetcher(json_error=URLError("dns down"))
        with self.assertRaisesRegex(assets_fetch.AssetsUnavailable, "dns down"):
            assets_fetch.download_hdri("some_asset", "2k", fetcher=fetcher)

    def test_truncated_download_is_rejected(self):
        asset_id = "short_asset"
        size = "2k"
        url = download_url(asset_id, size)
        fetcher = FakeFetcher(
            json_map={assets_fetch._files_url(asset_id):
                      hdri_files_payload(asset_id, size, HDR_BYTES,
                                         declared_size=len(HDR_BYTES) + 10)},
            downloads={url: HDR_BYTES})

        with self.assertRaisesRegex(assets_fetch.AssetsUnavailable, "truncated"):
            assets_fetch.download_hdri(asset_id, size, fetcher=fetcher)

        self.assertFalse((self.hdri_dir/("%s_2k.hdr" % asset_id)).exists())
        self.assertEqual(list(self.hdri_dir.glob("*")), [])

    def test_size_cap_aborts_and_removes_temporary_file(self):
        asset_id = "huge_asset"
        size = "2k"
        payload = HDR_BYTES * 4
        url = download_url(asset_id, size)
        fetcher = FakeFetcher(
            json_map={assets_fetch._files_url(asset_id):
                      hdri_files_payload(asset_id, size, payload)},
            downloads={url: payload})

        with self.assertRaisesRegex(assets_fetch.AssetsUnavailable, "size limit"):
            assets_fetch.download_hdri(asset_id, size, fetcher=fetcher, max_bytes=32)

        self.assertFalse((self.hdri_dir/("%s_2k.hdr" % asset_id)).exists())
        self.assertEqual(list(self.hdri_dir.glob("*")), [])

    def test_invalid_asset_id_and_size_are_rejected_before_network(self):
        fetcher = ExplodingFetcher()
        with self.assertRaises(ValueError):
            assets_fetch.download_hdri("../escape", "2k", fetcher=fetcher)
        with self.assertRaises(ValueError):
            assets_fetch.download_hdri("asset", "3k", fetcher=fetcher)
        self.assertEqual(fetcher.calls, [])

    # ---------------------------------------------------------------- catalog

    def test_catalog_roundtrip_and_atomic_write(self):
        sample = {
            "version": 1,
            "hdri": {"x_2k.hdr": {
                "id": "x", "size": "2k", "source": "polyhaven", "license": "CC0",
                "downloaded": "2026-01-01T00:00:00Z", "sha256": "ab" * 32,
            }},
            "materials": {},
        }

        written = assets_fetch.write_catalog(sample)

        self.assertEqual(Path(written), self.catalog_path.resolve())
        self.assertEqual(assets_fetch.read_catalog(), sample)
        on_disk = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["hdri"]["x_2k.hdr"]["sha256"], "ab" * 32)
        self.assertEqual(list(self.catalog_path.parent.glob(".catalog-*")), [])

    def test_corrupt_catalog_is_ignored_with_warning(self):
        self.catalog_path.write_text("{not json", encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            catalog = assets_fetch.read_catalog()
            self.assertEqual(assets_fetch.list_cached(), [])
        self.assertEqual(catalog["hdri"], {})
        self.assertTrue(any("unreadable" in str(item.message) for item in caught))

    def test_download_records_catalog_and_list_cached(self):
        asset_id = "recorded_sky"
        size = "2k"
        fetcher = self.fake_weather("overcast", asset_id)

        result = assets_fetch.hdri_for_weather("fog", size, fetcher=fetcher)

        entry = assets_fetch.read_catalog()["hdri"][Path(result["path"]).name]
        self.assertEqual(entry["id"], asset_id)
        self.assertEqual(entry["size"], size)
        self.assertEqual(entry["source"], "polyhaven")
        self.assertEqual(entry["license"], "CC0")
        self.assertEqual(entry["bytes"], len(HDR_BYTES))
        self.assertEqual(entry["sha256"], hashlib.sha256(HDR_BYTES).hexdigest())
        self.assertTrue(entry["downloaded"].endswith("Z"))

        cached = assets_fetch.list_cached()
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0]["kind"], "hdri")
        self.assertEqual(cached[0]["path"], result["path"])
        self.assertEqual(cached[0]["sha256"], entry["sha256"])

    # ---------------------------------------------------------------- materials

    def material_asset(self, asset_id="Asphalt033", size="1k"):
        url = "https://cdn.example.test/%s_%s-JPG.zip" % (asset_id, size.upper())
        return {
            "assetId": asset_id,
            "downloadCount": 100,
            "downloadFolders": {"default": {"downloadFiletypeCategories": {"zip": {
                "downloads": [{
                    "attribute": size.upper() + "-JPG",
                    "fileName": "%s_%s-JPG.zip" % (asset_id, size.upper()),
                    "downloadLink": url,
                    "fullDownloadPath": url,
                    "size": len(MATERIAL_ZIP),
                }]
            }}}},
        }, url

    def test_material_for_downloads_extracts_and_caches_offline(self):
        asset, url = self.material_asset()
        fetcher = FakeFetcher(
            json_map={assets_fetch._material_search_url("asphalt"):
                      {"foundAssets": [asset]}},
            downloads={url: MATERIAL_ZIP})

        result = assets_fetch.material_for("asphalt", "1k", fetcher=fetcher)

        self.assertEqual(result["id"], "Asphalt033")
        self.assertEqual(result["source"], "ambientcg")
        self.assertEqual(result["license"], "CC0")
        self.assertFalse(result["cached"])
        self.assertEqual(Path(result["path"]).name, "Asphalt033_1K_Color.jpg")
        self.assertEqual(Path(result["path"]).read_bytes(), b"color pixels")
        self.assertEqual(result["sha256"], hashlib.sha256(MATERIAL_ZIP).hexdigest())

        zip_key = "%s_%s.zip" % (result["id"], result["size"])
        entry = assets_fetch.read_catalog()["materials"][zip_key]
        self.assertEqual(entry["id"], "Asphalt033")
        self.assertEqual(entry["kinds"], ["asphalt"])
        self.assertEqual(entry["size"], "1k")

        offline = assets_fetch.material_for("asphalt", "1k", fetcher=ExplodingFetcher())
        self.assertTrue(offline["cached"])
        self.assertEqual(offline["path"], result["path"])
        self.assertEqual([item["kind"] for item in assets_fetch.list_cached()], ["material"])

    def test_material_for_unknown_kind_returns_none_without_network(self):
        fetcher = ExplodingFetcher()
        self.assertIsNone(assets_fetch.material_for("lava", fetcher=fetcher))
        self.assertEqual(fetcher.calls, [])

    def test_material_download_failure_raises(self):
        asset, url = self.material_asset()
        fetcher = FakeFetcher(
            json_map={assets_fetch._material_search_url("asphalt"):
                      {"foundAssets": [asset]}},
            downloads={url: MATERIAL_ZIP},
            download_error=URLError("offline"))

        with self.assertRaises(assets_fetch.AssetsUnavailable):
            assets_fetch.material_for("asphalt", "1k", fetcher=fetcher)

    # ---------------------------------------------------------------- configuration

    def test_defaults_document_timeout_and_size_cap(self):
        self.assertEqual(assets_fetch.DEFAULT_TIMEOUT, 20.0)
        self.assertGreater(assets_fetch.MAX_DOWNLOAD_BYTES, 0)
        fetcher = assets_fetch.UrllibFetcher()
        self.assertEqual(fetcher.timeout, 20.0)
        self.assertEqual(fetcher.max_bytes, assets_fetch.MAX_DOWNLOAD_BYTES)


MATERIAL_ZIP = material_zip({
    "Asphalt033_1K_Color.jpg": b"color pixels",
    "Asphalt033_1K_NormalGL.jpg": b"normal pixels",
    "Asphalt033_1K_Roughness.jpg": b"rough pixels",
})


if __name__ == "__main__":
    unittest.main()
