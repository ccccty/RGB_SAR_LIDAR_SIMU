"""The local workbench must not be drivable by a web page.

It has no login, so the only thing standing between a random tab and a
20000-frame render on the user's disk is a set of origin checks. Before this
suite existed, ``fetch(..., {mode: "no-cors", body: '...'})`` with
``Content-Type: text/plain`` reached /api/generate; a DNS-rebinding page could
read every dataset because Host was never checked; and an output path could be
anywhere on the machine, including an UNC share.
"""
from __future__ import annotations

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from simkit import studio


class OriginGuardTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), studio.Handler)
        cls.port = cls.srv.server_port
        cls.thread = threading.Thread(target=cls.srv.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def _request(self, method, path, body=b"", headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=8)
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for key, value in (headers or {}).items():
            conn.putheader(key, value)
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status

    def test_dns_rebinding_host_is_refused(self):
        status = self._request(
            "GET", "/api/recipes",
            headers={"Host": "attacker.test:%d" % self.port})
        self.assertEqual(status, 421)

    def test_text_plain_csrf_post_is_refused(self):
        status = self._request(
            "POST", "/api/generate", body=b'{"frames": 20000}',
            headers={"Host": "127.0.0.1:%d" % self.port,
                     "Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_cross_origin_json_post_is_refused(self):
        body = json.dumps({"scene": "x"}, ensure_ascii=False).encode()
        status = self._request(
            "POST", "/api/plan", body=body,
            headers={"Host": "127.0.0.1:%d" % self.port,
                     "Content-Type": "application/json",
                     "Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_loopback_json_get_still_works(self):
        status = self._request(
            "GET", "/api/recipes",
            headers={"Host": "127.0.0.1:%d" % self.port})
        self.assertEqual(status, 200)


class OutputSandboxTests(unittest.TestCase):

    def test_unc_path_is_refused(self):
        with self.assertRaises(studio.BadRequest):
            studio._safe_out_dir("\\\\attacker\\share\\out")

    def test_absolute_path_outside_the_project_is_refused(self):
        with self.assertRaises(studio.BadRequest):
            studio._safe_out_dir("C:/Windows/Temp/simkit_out")

    def test_relative_escape_is_refused(self):
        with self.assertRaises(studio.BadRequest):
            studio._safe_out_dir("../../escaped_out")

    def test_project_root_itself_is_refused(self):
        with self.assertRaises(studio.BadRequest):
            studio._safe_out_dir(studio.ROOT)

    def test_default_output_is_allowed(self):
        self.assertIsNone(studio._safe_out_dir(None))
        self.assertIsNone(studio._safe_out_dir(""))


if __name__ == "__main__":
    unittest.main()
