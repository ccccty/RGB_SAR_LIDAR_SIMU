"""Offline readiness check: run this on the air-gapped machine.

    env\Scripts\python.exe tools\offline_check.py
"""
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def line(ok, msg):
    print(("  OK   " if ok else "  FAIL ") + msg)
    return ok


def main():
    ok = True
    print("simkit offline check  root=%s" % ROOT)

    try:
        import numpy as np
        ok &= line(True, "numpy %s (%s)" % (np.__version__, os.path.dirname(np.__file__)))
    except Exception as e:
        ok &= line(False, "numpy import failed: %s" % e)

    ok &= line(os.path.isdir(os.path.join(ROOT, "simkit")), "simkit package present")
    ok &= line(os.path.isdir(os.path.join(ROOT, "scenes")), "scenes present")

    from simkit import optical
    blender = optical.find_blender()
    ok &= line(blender is not None, "Blender: %s" % (blender or "not found (optical disabled)"))

    # no network dependency: imports must succeed with sockets blocked
    class _Blocked(socket.socket):
        def __init__(self, *a, **k):
            raise OSError("network disabled for offline check")

    orig = socket.socket
    socket.socket = _Blocked
    try:
        from simkit import dataset, ir, radar, thermal  # noqa: F401
        ok &= line(True, "core modules import with sockets blocked")
    except Exception as e:
        ok &= line(False, "import with sockets blocked: %s" % e)
    finally:
        socket.socket = orig

    print("\nselftest:")
    from simkit.selftest import run
    ok &= bool(run())

    print("\nRESULT:", "READY FOR OFFLINE USE" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
