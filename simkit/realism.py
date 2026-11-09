"""One place that turns the "already implemented but switched off" realism
features on for every generation path (CLI, studio, batch, validate).

Background (PLAN.md, 客户视角改进清单 §5 item 0): the visible-band ISP chain,
the HDR optical sensor model, motion blur, the IR display window and the
height-profile atmosphere were all written, tested and then left disabled.
A dataset generated from a scene JSON therefore looked like a clean game render
even though the physics was sitting in the repository. Fixing that by editing
44 scene files one by one would drift as soon as a 45th scene appears, so the
defaults are injected at the single configuration entry point instead.

Contract
    apply(cfg) is called by scene.load_config(). It is idempotent and only
    fills keys a scene did not declare: an explicit value always wins. A scene
    can opt out entirely with ``{"realism": {"enabled": false}}``; individual
    switches can be overridden with a normal ``optical.camera.<stage>.enabled``
    or ``optical.sensor.enabled`` block.

Why jpeg is NOT put in optical.camera.jpeg
    camera.jpeg_roundtrip needs Pillow, and Blender's bundled Python does not
    ship it -- enabling that stage makes every Cycles render raise. The lossy
    codec is therefore delivered by ``output.jpeg`` (dataset.py, main process,
    Pillow present), which writes a real .jpg alongside the lossless PNG.
"""
from __future__ import annotations

# Stages of simkit/camera.py that only need numpy and can run inside Blender.
ISP_STAGES = ("vignetting", "chromatic_aberration", "flare")
JPEG_QUALITY = 90
PREVIEW_SPAN_C = 20.0
RGB_EXPOSURE_S = 0.004
SAR_DISPLAY_ANCHOR_PCT = 1.5


def _sub(cfg, key):
    """Return a mutable dict under cfg[key], creating it when absent."""
    got = cfg.get(key)
    if not isinstance(got, dict):
        got = {}
        cfg[key] = got
    return got


def _default(d, key, value):
    """Fill key only when it is absent or explicitly null."""
    if d.get(key) is None:
        d[key] = value


def apply(cfg):
    """Inject realism defaults into a scene config, in place. Returns cfg."""
    realism = cfg.get("realism")
    if isinstance(realism, dict) and realism.get("enabled") is False:
        return cfg
    if realism is False:
        return cfg

    optical = _sub(cfg, "optical")

    # 1. Lens / sensor ISP: diffraction + sensor are already on by default in
    #    camera.DEFAULTS; these three were off. Do not touch camera.DEFAULTS
    #    itself (tests/test_camera.py pins the module-level defaults).
    cam = _sub(optical, "camera")
    for stage in ISP_STAGES:
        s = _sub(cam, stage)
        _default(s, "enabled", True)
    # jpeg lives in the main process; see module docstring. It is opt-in
    # because "output.jpeg off by default" is a pinned contract in
    # tests/test_jpeg_output.py: writing an extra .jpg into every historical
    # run would break that. Enable per scene with {"realism":{"jpeg":true}} or
    # by declaring output.jpeg explicitly.
    if isinstance(realism, dict) and realism.get("jpeg") is True:
        _default(_sub(cfg, "output"), "jpeg", JPEG_QUALITY)

    # 2. Full HDR optical sensor chain (atmosphere + PSF + electron/count
    #    model). render_optical.py gates this on optical.sensor.enabled and
    #    skips the overlapping camera.py sensor/diffraction stages for us.
    sensor = _sub(optical, "sensor")
    _default(sensor, "enabled", True)

    # 3. Exposure-synchronised motion blur. The correct key is
    #    acquisition.rgb_exposure_s (plan.py feeds it into shutter_samples);
    #    PLAN.md's "optical.exposure_s" was never a real key and is ignored.
    _default(optical, "motion_blur", True)
    frames = cfg.get("frames") if isinstance(cfg.get("frames"), dict) else {}
    dt = float(frames.get("dt") or 0.05)
    exp_s = min(RGB_EXPOSURE_S, max(dt * 0.5, 1e-4))
    acq = _sub(cfg, "acquisition")
    _default(acq, "rgb_exposure_s", exp_s)
    if sensor.get("enabled") is True and optical.get("motion_blur"):
        # The electron model divides by reference_exposure_s, so the two keys
        # must describe the same integration window -- but they must follow the
        # shutter that is ACTUALLY in force. A blanket `= exp_s` rewrote the 9
        # delivered scenes that declare 0.001 s to 0.004 s, and the target then
        # rendered four times darker than its photographed plate. An explicit
        # pair is a contract and is never touched.
        _default(sensor, "reference_exposure_s", float(acq["rgb_exposure_s"]))

    # 4. IR 8-bit preview window: without a cap it stretches to the 1/99.5
    #    percentiles (~232 mK per code), which makes a 30 mK NETD invisible.
    _default(_sub(cfg, "ir"), "preview_span_c", PREVIEW_SPAN_C)

    # 5. Altitude-aware atmosphere (visible + IR path radiance / extinction).
    _default(_sub(cfg, "atmosphere"), "profile", "exponential")

    # 5b. A real background plate plus auto_exposure is the exact combination
    #     that flattened the 2.2 deg seeker frame: the exposure solver targets
    #     mean luma 0.48 and erases the plate's contrast. Only applies when the
    #     scene actually carries a plate.
    ground_cfg = cfg.get("ground")
    sky_plate = (cfg.get("optical") or {}).get("sky_image")
    ir_plate = (cfg.get("ir") or {}).get("background_image")
    if ((isinstance(ground_cfg, dict) and ground_cfg.get("backdrop"))
            or sky_plate or ir_plate):
        _default(optical, "auto_exposure", False)

    # 6. SAR display stretch anchor (the kernel default is now 1.5 as well;
    #    this keeps a scene that carries an explicit sensors.sar block aligned).
    sar = (_sub(cfg, "sensors")).get("sar")
    if isinstance(sar, dict):
        _default(sar, "display_anchor_pct", SAR_DISPLAY_ANCHOR_PCT)

    # 7. Let thermal.solve() actually decide surface temperature. A per-object
    #    temp_c overrides the 60-iteration radiative balance with one constant,
    #    which is why a driving rover rendered as a featureless white ellipse.
    #    ECM entities keep their temp_c (a 2000 C flare must stay hot), and
    #    air.temp_c is the ambient input, not an override -- both untouched.
    for obj in cfg.get("objects") or []:
        if isinstance(obj, dict) and obj.get("temp_c") is not None:
            obj["temp_c"] = None
    ground = cfg.get("ground")
    if isinstance(ground, dict) and ground.get("temp_c") is not None:
        ground["temp_c"] = None

    return cfg


def summary(cfg):
    """Human-readable one-liner for logs / preflight output."""
    after = apply(dict(cfg)) if isinstance(cfg, dict) else {}
    optical = after.get("optical") or {}
    cam = optical.get("camera") or {}
    active = [s for s in ISP_STAGES if (cam.get(s) or {}).get("enabled")]
    return ("realism: isp=%s sensor=%s motion_blur=%s preview_span_c=%s "
            "atmosphere=%s output_jpeg=%s" % (
                ",".join(active) or "off",
                bool((optical.get("sensor") or {}).get("enabled")),
                bool(optical.get("motion_blur")),
                (after.get("ir") or {}).get("preview_span_c"),
                (after.get("atmosphere") or {}).get("profile"),
                (after.get("output") or {}).get("jpeg")))
