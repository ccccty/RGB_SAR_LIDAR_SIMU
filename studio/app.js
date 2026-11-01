(function () {
  "use strict";

  // Single source of truth for the form. Every widget writes one key here and
  // nothing reads the DOM back, so "what will be generated" is always exactly
  // what the summary strip and the payload agree on.
  var S = {
    scene: "", size: "medium", pxLo: 18, pxHi: 42,
    closing: "in", closingMps: 0, rngOn: true, rngLo: 1500, rngHi: 16000,
    viewsMode: "single", nFov: 2.2, nW: 960, nH: 720, wFov: 18, wW: 1280, wH: 960,
    sar: false, sarBand: "X", sarBw: 100, swathC: 4000, swathW: 1200,
    sarAlt: 3000, sarSpeed: 150, sarSquint: 0, sarPol: "VV",
    traj: "platform", pAlt: 1200, pSpeed: 180, pHeading: 90, pDive: -14,
    pWeave: 18, pWeaveHz: 0.22, pBack: 3000, pStandoff: 260,
    weather: "mixed", sensors: ["ir"], frames: 240, fps: 10, fpc: 120, seed: 7, name: ""
  };

  var OPT = { scenes: [], sarBands: {}, limits: {}, sizePx: {} };
  var lastPlan = null;
  var active = null;         // job id the gallery is following
  var pollTimer = null;
  var planTimer = null;

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt !== undefined && txt !== null) e.textContent = String(txt);
    return e;
  }
  function get(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) {
      return r.json().then(function (b) {
        if (!r.ok) throw new Error(b.error || ("HTTP " + r.status));
        return b;
      });
    });
  }
  function post(url, body) {
    return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(body) }).then(function (r) {
      return r.json().then(function (b) {
        if (!r.ok) throw new Error(b.error || ("HTTP " + r.status));
        return b;
      });
    });
  }
  function num(v, def) {
    var x = parseFloat(v);
    return isFinite(x) ? x : def;
  }
  function fmt(v, d) {
    if (v === null || v === undefined || v === "") return "—";
    if (typeof v !== "number") return String(v);
    return v.toFixed(d === undefined ? 1 : d);
  }

  function alert_(msg) {
    var box = $("formErr");
    if (!msg) { box.hidden = true; box.textContent = ""; return; }
    box.hidden = false;
    box.textContent = String(msg);
  }

  // ---------------------------------------------------------------- widgets
  function chips(host, items, key, multi) {
    host.innerHTML = "";
    items.forEach(function (it) {
      var b = el("button", "", it.label);
      b.type = "button";
      b.setAttribute("data-v", it.value);
      var on = multi ? (S[key].indexOf(it.value) >= 0) : (S[key] === it.value);
      if (on) b.className = "on";
      b.onclick = function () {
        if (multi) {
          var i = S[key].indexOf(it.value);
          if (i >= 0) { if (S[key].length > 1) S[key].splice(i, 1); }
          else S[key].push(it.value);
        } else {
          S[key] = it.value;
        }
        chips(host, items, key, multi);
        changed();
      };
      host.appendChild(b);
    });
  }

  function bindNum(id, key, lo, hi) {
    var e = $(id);
    e.value = S[key];
    e.addEventListener("input", function () {
      var v = num(e.value, S[key]);
      if (lo !== undefined && v < lo) v = lo;
      if (hi !== undefined && v > hi) v = hi;
      S[key] = v;
      changed();
    });
  }
  function bindText(id, key) {
    var e = $(id);
    e.value = S[key];
    e.addEventListener("input", function () { S[key] = e.value; changed(); });
  }

  function refreshGates() {
    $("wideBox").classList.toggle("off", S.viewsMode !== "dual");
    $("sarBox").classList.toggle("off", !S.sar);
    $("platBox").classList.toggle("off", S.traj !== "platform");
    var band = OPT.sarBands[S.sarBand];
    $("sarBandNote").textContent = band ? (band.fc_ghz + " GHz · " + band.note) : "";
    $("sarRes").textContent = (150 / Math.max(S.sarBw, 1e-6)).toFixed(2) + " m";
  }

  // ---------------------------------------------------------------- payload
  function payload() {
    var b = {
      scene: S.scene, name: S.name, size: S.size,
      target_px: [Math.min(S.pxLo, S.pxHi), Math.max(S.pxLo, S.pxHi)],
      closing: S.closing, closing_mps: S.closingMps > 0 ? S.closingMps : null,
      views_mode: S.viewsMode,
      narrow_fov: S.nFov, narrow_width: S.nW, narrow_height: S.nH,
      wide_fov: S.wFov, wide_width: S.wW, wide_height: S.wH,
      traj_mode: S.traj,
      platform_alt: S.pAlt, platform_speed: S.pSpeed, platform_heading: S.pHeading,
      platform_dive: S.pDive, platform_weave: S.pWeave, platform_weave_hz: S.pWeaveHz,
      platform_back: S.pBack, platform_standoff: S.pStandoff,
      sar_enabled: S.sar, sar_band: S.sarBand, sar_bandwidth_mhz: S.sarBw,
      sar_swath_center_m: S.swathC, sar_swath_width_m: S.swathW,
      sar_altitude_m: S.sarAlt, sar_speed_mps: S.sarSpeed,
      sar_squint_deg: S.sarSquint, sar_polarization: S.sarPol,
      weather: S.weather, sensors: S.sensors,
      frames: S.frames, fps: S.fps, frames_per_clip: Math.min(S.fpc, S.frames),
      seed: S.seed
    };
    if (S.rngLo > 0 && S.rngHi > 0) b.range_m = [S.rngLo, S.rngHi];
    return b;
  }

  // ---------------------------------------------------------------- summary
  var SENSOR_CN = { optical: "可见光", ir: "红外", radar: "雷达", lidar: "激光雷达", sar: "SAR" };
  var WEATHER_CN = { clear: "晴", cloudy: "多云", rain: "雨", fog: "雾", mixed: "混合", evolve: "演变" };
  var TRAJ_CN = { platform: "平台俯冲+蛇形", seeker: "相机静立·目标飞行", world: "世界轨迹" };

  function localSummary() {
    var clips = Math.max(1, Math.ceil(S.frames / Math.max(1, Math.min(S.fpc, S.frames))));
    var per = Math.min(S.fpc, S.frames);
    var secs = S.frames / Math.max(S.fps, 0.1);
    var lo = Math.min(S.pxLo, S.pxHi), hi = Math.max(S.pxLo, S.pxHi);
    var txt = "本次将生成 " + clips + " 段 × " + per + " 帧（共 " + S.frames + " 帧 / " +
      secs.toFixed(1) + " 秒 @ " + S.fps + " fps），窄视场目标 " + lo + "–" + hi + " px（占宽 " +
      (100 * lo / S.nW).toFixed(1) + "%–" + (100 * hi / S.nW).toFixed(1) + "%）";
    if (S.viewsMode === "dual") {
      var k = S.wFov / S.nFov;
      txt += "，宽视场约 " + fmt(lo / k, 1) + "–" + fmt(hi / k, 1) + " px";
    }
    txt += "；" + S.sensors.map(function (s) { return SENSOR_CN[s] || s; }).join("+") +
      "；" + TRAJ_CN[S.traj] + "；天气 " + (WEATHER_CN[S.weather] || S.weather) +
      "；SAR " + (S.sar ? ("开（" + S.sarBand + " · " + S.sarBw + " MHz · 距离分辨率 " +
      (150 / S.sarBw).toFixed(2) + " m）") : "关");
    return txt;
  }

  function paintSummary() {
    var strip = $("summary");
    if (lastPlan && lastPlan.summary) {
      var m = lastPlan.summary;
      var t = "本次将生成 " + m.clips + " 段 × " + m.frames_per_clip + " 帧（共 " + m.frames +
        " 帧 / " + m.seconds + " 秒 @ " + fmt(m.fps, 2) + " fps）";
      if (m.px_abs) {
        t += "，窄视场目标 " + fmt(m.px_abs[0], 1) + "–" + fmt(m.px_abs[1], 1) + " px（占宽 " +
          fmt(m.px_min, 1) + "%–" + fmt(m.px_max, 1) + "%）";
      }
      if (lastPlan.clips.length && lastPlan.clips[0].view_px && lastPlan.clips[0].view_px.wide) {
        var wl = [], wh = [];
        lastPlan.clips.forEach(function (c) {
          wl.push(c.view_px.wide[0]); wh.push(c.view_px.wide[1]);
        });
        t += "，宽视场 " + fmt(Math.min.apply(null, wl), 1) + "–" +
          fmt(Math.max.apply(null, wh), 1) + " px";
      }
      t += "；" + m.sensors.map(function (s) { return SENSOR_CN[s] || s; }).join("+") +
        "；" + TRAJ_CN[Q.scene.traj] + "；天气 " + (WEATHER_CN[m.weather] || m.weather) +
        "；SAR " + (m.sar ? ("开（" + Q.sar.band + " · 距离分辨率 " + fmt(Q.sar.range_res_m, 2) + " m）") : "关");
      strip.textContent = t;
      strip.classList.remove("stale");
    } else {
      strip.textContent = localSummary();
      strip.classList.add("stale");
    }
  }
  var Q = { scene: {}, sar: {} };

  function changed() {
    refreshGates();
    paintSummary();
    clearTimeout(planTimer);
    planTimer = setTimeout(loadPlan, 550);
  }

  // ---------------------------------------------------------------- scenes
  function renderScenes() {
    var sel = $("scene");
    sel.innerHTML = "";
    OPT.scenes.forEach(function (s) {
      var o = el("option", "", s.title + "  ·  " + s.key);
      o.value = s.key;
      if (s.key === S.scene) o.selected = true;
      sel.appendChild(o);
      OPT.byKey = OPT.byKey || {};
      OPT.byKey[s.key] = s;
    });
    $("sceneCount").textContent = OPT.scenes.length + " 个场景";
    var info = (OPT.byKey || {})[S.scene];
    $("sceneNote").textContent = info
      ? (info.note + " · 传感器 " + info.sensors.join("/") +
         (info.has_sar ? " · 自带 SAR 段" : ""))
      : "";
  }

  function sceneDefaults() {
    var info = (OPT.byKey || {})[S.scene];
    if (!info) return;
    if (info.target_px && info.target_px.length === 2) {
      S.pxLo = Math.round(num(info.target_px[0], 18));
      S.pxHi = Math.round(num(info.target_px[1], 42));
      $("pxLo").value = S.pxLo;
      $("pxHi").value = S.pxHi;
    }
    if (info.range_m && info.range_m.length === 2) {
      S.rngLo = Math.round(num(info.range_m[0], 1500));
      S.rngHi = Math.round(num(info.range_m[1], 16000));
      $("rngLo").value = S.rngLo;
      $("rngHi").value = S.rngHi;
    }
    if (info.fov && info.fov > 0 && S.nFov !== info.fov) {
      S.nFov = num(info.fov, 2.2);
      $("nFov").value = S.nFov;
    }
    // Frame rate and lens sizes belong to the scene: a seeker clip authored at
    // dt=0.05 is 20 fps, and showing "10 fps" next to "dt=0.05" is a lie the
    // operator cannot resolve.
    if (info.fps && info.fps > 0) {
      S.fps = info.fps;
      var ef = $("fps");
      if (ef) ef.value = info.fps;
    }
    if (info.views && info.views.length) {
      var nv = info.views[0], wv = info.views[1];
      // width inputs only exist on some builds; never assume a control is there
      function put(id, v, k) {
        if (v) S[k] = Math.round(num(v, S[k]));
        var e = $(id);
        if (e) e.value = S[k];
      }
      function putF(id, v, k) {
        if (v) S[k] = num(v, S[k]);
        var e = $(id);
        if (e) e.value = S[k];
      }
      if (nv) { putF("nFov", nv.fov_deg, "nFov"); put("nWidth", nv.width, "nWidth"); }
      if (wv) { putF("wFov", wv.fov_deg, "wFov"); put("wWidth", wv.width, "wWidth"); }
    }
    if (info.closing && info.closing in { in: 1, out: 1, hold: 1 }) {
      S.closing = info.closing;
      chips($("closing"), [{ value: "in", label: "接近" }, { value: "out", label: "远离" },
                           { value: "hold", label: "保持" }], "closing", false);
    }
    // A scene with a seeker block gets the new platform trajectory by default;
    // a scene without one can only run its world trajectory.
    S.traj = info.has_seeker ? "platform" : "world";
    if (info.views && info.views.length > 1) S.viewsMode = "dual";
    $("viewsMode").querySelectorAll("button").forEach(function (b) {
      b.classList.toggle("on", b.getAttribute("data-v") === S.viewsMode);
    });
    $("traj").querySelectorAll("button").forEach(function (b) {
      b.classList.toggle("on", b.getAttribute("data-v") === S.traj);
    });
  }

  // ---------------------------------------------------------------- plan
  function loadPlan() {
    var body = payload();
    var t0 = Date.now();
    $("plan").classList.add("busy");
    post("/api/plan", body).then(function (d) {
      lastPlan = d;
      Q = { scene: d.scene || {}, sar: d.sar || {} };
      renderPlan(d, Date.now() - t0);
      paintSummary();
    }).catch(function (e) {
      lastPlan = null;
      $("plan").innerHTML = "";
      $("plan").appendChild(el("div", "alert", "轨迹预览失败：" + e.message));
      paintSummary();
    }).then(function () { $("plan").classList.remove("busy"); });
  }

  function renderPlan(d, ms) {
    var host = $("plan");
    host.innerHTML = "";
    if (!d.clips || !d.clips.length) {
      host.appendChild(el("div", "empty", "（" + (d.error || "没有片段") + "）"));
      return;
    }
    var facts = el("div", "facts");
    facts.appendChild(el("span", "fact", "目标参考尺寸 " + fmt(d.scene.size_m, 2) + " m"));
    facts.appendChild(el("span", "fact", "窄视场 " + fmt(d.views[0].fov_deg, 1) + "° @" +
      d.views[0].width + "×" + d.views[0].height));
    if (d.views[1]) {
      facts.appendChild(el("span", "fact", "宽视场 " + fmt(d.views[1].fov_deg, 1) + "° @" +
        d.views[1].width + "×" + d.views[1].height));
    }
    facts.appendChild(el("span", "fact", "dt " + fmt(d.summary.dt, 3) + " s"));
    facts.appendChild(el("span", "fact", d.summary.clips + " 段 × " +
      d.summary.frames_per_clip + " 帧"));
    facts.appendChild(el("span", "fact", "算完 " + ms + " ms"));
    if (d.summary.sar) {
      facts.appendChild(el("span", "fact", "SAR " + d.sar.band + " · " +
        d.sar.bandwidth_mhz + " MHz · 距离分辨率 " + fmt(d.sar.range_res_m, 2) + " m · 测绘带 " +
        d.sar.swath_center_m + "±" + (d.sar.swath_width_m / 2) + " m"));
    }
    host.appendChild(facts);

    var wrap = el("div", "tw");
    var t = el("table");
    var hr = el("tr");
    var heads = ["段", "天气", "距离 m", "窄视场 px", "窄占比", "宽视场 px", "宽占比",
                 "目标 m/s", "加速度 m/s²", "平台 m/s", "bank°", "模式"];
    heads.forEach(function (h) { hr.appendChild(el("th", "", h)); });
    t.appendChild(hr);
    d.clips.slice(0, 24).forEach(function (c) {
      var tr = el("tr");
      function td(v, mono) { tr.appendChild(el("td", mono ? "mono" : "", v)); }
      td(c.clip, true);
      td(c.weather);
      td(fmt(c.range_m[0], 0) + " → " + fmt(c.range_m[1], 0), true);
      td(fmt(c.view_px.seeker[0], 1) + " → " + fmt(c.view_px.seeker[1], 1), true);
      td(fmt(c.ratio.seeker[0], 2) + "% → " + fmt(c.ratio.seeker[1], 2) + "%", true);
      td(c.view_px.wide ? (fmt(c.view_px.wide[0], 1) + " → " + fmt(c.view_px.wide[1], 1)) : "—", true);
      td(c.ratio.wide ? (fmt(c.ratio.wide[0], 2) + "% → " + fmt(c.ratio.wide[1], 2) + "%") : "—", true);
      td(fmt(c.speed_mps, 1), true);
      td(fmt(c.accel_mps2, 1), true);
      td(c.platform_speed_mps === undefined || c.platform_speed_mps === null
         ? "—" : fmt(c.platform_speed_mps, 1), true);
      td(c.bank_deg === undefined || c.bank_deg === null ? "—" : fmt(c.bank_deg, 1), true);
      td(c.mode);
      t.appendChild(tr);
    });
    wrap.appendChild(t);
    host.appendChild(wrap);
    if (d.clips.length > 24) {
      host.appendChild(el("div", "hint", "只显示前 24 段，共 " + d.clips.length + " 段"));
    }
    var note = el("div", "hint");
    note.textContent = "占比 = 目标像素跨度 / 画面宽度。这些是 plan 的预测值；实测框以样张卡片为准。";
    host.appendChild(note);
  }

  // ---------------------------------------------------------------- jobs
  var STATE_CN = { queued: "排队", running: "生成中", done: "完成", failed: "失败", cancelled: "已取消" };

  function jobCard(j) {
    var card = el("div", "job " + j.state);
    var top = el("div", "job-top");
    top.appendChild(el("span", "job-id", j.id));
    top.appendChild(el("span", "state " + j.state, STATE_CN[j.state] || j.state));
    card.appendChild(top);

    var pct = j.total ? Math.min(100, Math.round(100 * j.done / j.total)) : 0;
    var bar = el("div", "bar");
    var fill = el("i");
    fill.style.width = pct + "%";
    bar.appendChild(fill);
    card.appendChild(bar);
    card.appendChild(el("div", "job-meta",
      (j.total ? ("第 " + j.done + " / " + j.total + " 帧") : "尚未输出帧") +
      " · " + j.seconds + " 秒" + (j.error ? " · " + j.error : "")));

    if (j.meta && j.meta.scene) {
      card.appendChild(el("div", "job-meta", "场景 " + j.meta.scene + " · 视场 " +
        (j.meta.views === "dual" ? "双" : "单") + " · " + (j.meta.sensors || []).join("+") +
        (j.meta.sar ? " · SAR" : "")));
    }
    if (j.error_line) {
      card.appendChild(el("div", "job-exc", j.error_line));
    }
    if (j.error_tail && j.error_tail.length) {
      var pre = el("pre", "log err", j.error_tail.join("\n"));
      card.appendChild(pre);
      card.appendChild(el("div", "hint", "以上是后端 stderr / stdout 的最后 " +
        j.error_tail.length + " 行"));
    } else if (j.log && j.log.length) {
      card.appendChild(el("pre", "log", j.log.slice(-6).join("\n")));
    }
    var acts = el("div", "job-acts");
    var b = el("button", "ghost small", "看样张");
    b.type = "button";
    b.onclick = function () { active = j.id; showGallery(j.id); };
    acts.appendChild(b);
    if (j.state === "running" || j.state === "queued") {
      var c = el("button", "ghost small", "停止");
      c.type = "button";
      c.onclick = function () {
        post("/api/job/" + j.id + "/cancel", {}).then(poll).catch(function (e) { alert_(e.message); });
      };
      acts.appendChild(c);
    }
    card.appendChild(acts);
    return card;
  }

  function poll() {
    get("/api/jobs").then(function (d) {
      var host = $("jobs");
      host.innerHTML = "";
      if (!d.jobs || !d.jobs.length) {
        host.appendChild(el("div", "empty", "还没有任务。选好参数后点「生成数据集」。"));
        $("jobHint").textContent = "";
        return;
      }
      d.jobs.slice(0, 4).forEach(function (j) { host.appendChild(jobCard(j)); });
      var running = d.jobs.filter(function (j) { return j.state === "running"; })[0];
      $("jobHint").textContent = running ? ("正在跑 " + running.id) : "";
      var focus = active || d.jobs[0].id;
      var fjob = d.jobs.filter(function (j) { return j.id === focus; })[0];
      if (focus && (!fjob || fjob.done > 0 || fjob.state === "done")) {
        showGallery(focus);
      }
      clearTimeout(pollTimer);
      if (running || (fjob && fjob.state === "queued")) {
        pollTimer = setTimeout(poll, 1200);
      } else {
        pollTimer = setTimeout(poll, 4000);
      }
    }).catch(function () {
      clearTimeout(pollTimer);
      pollTimer = setTimeout(poll, 4000);
    });
  }

  // ---------------------------------------------------------------- gallery
  function showGallery(jobId) {
    var host = $("gallery");
    get("/api/dataset/" + jobId).then(function (d) {
      var cards = d.cards || [];
      if (host.getAttribute("data-job") === jobId &&
          host.getAttribute("data-n") === String(cards.length)) return;
      // Build off-document and swap once: polling every 1.2 s would otherwise
      // blank the gallery and make the images flicker on every tick.
      var box = document.createElement("div");
      box.appendChild(el("div", "hint", jobId + " · " + d.dir));
      if (!cards.length) {
        box.appendChild(el("div", "empty", "这一轮还没有写出图（" + jobId + "）。"));
      } else {
        var grid = el("div", "cards");
        cards.slice(0, 12).forEach(function (c) {
          var fig = el("figure", "card");
          var img = el("img");
          img.loading = "lazy";
          img.src = c.url;
          img.alt = c.label;
          fig.appendChild(img);
          var cap = el("figcaption");
          cap.appendChild(el("strong", "", c.label));
          var rows = [
            (c.w || "?") + "×" + (c.h || "?") + " · 帧 " + (c.frame === null ? "?" : c.frame) +
              (c.fov_deg ? " · FOV " + fmt(c.fov_deg, 2) + "°" : ""),
            c.box ? ("实测框 " + fmt(c.box[0], 0) + "×" + fmt(c.box[1], 0) + " px · 占宽 " +
                     fmt(c.ratio, 1) + "%") : (c.target_px ?
                     ("目标 " + (typeof c.target_px === "number" ? fmt(c.target_px, 1) :
                      fmt(c.target_px[0], 1) + "–" + fmt(c.target_px[1], 1)) + " px") :
                     "无目标框（整幅成像）"),
            c.res_range_m ? ("分辨率 " + fmt(c.res_range_m, 2) + " m（距离）× " +
              fmt(c.res_az_m, 2) + " m（方位）") : "",
            c.swath_m && c.swath_m[0] ? ("测绘带 " + fmt(c.swath_m[0], 0) + "–" +
              fmt(c.swath_m[1], 0) + " m") : "",
            (c.range_m ? "视线 " + fmt(c.range_m, 0) + " m" : "") +
              (c.snr_db !== undefined && c.snr_db !== null ? " · SNR " + fmt(c.snr_db, 1) + " dB" :
               (c.peak_db !== undefined && c.peak_db !== null ? " · 峰值 " + fmt(c.peak_db, 1) + " dB" : ""))
          ];
          rows.forEach(function (r) { if (r) cap.appendChild(el("span", "", r)); });
          fig.appendChild(cap);
          grid.appendChild(fig);
        });
        box.appendChild(grid);
      }
      host.innerHTML = "";
      while (box.firstChild) host.appendChild(box.firstChild);
      host.setAttribute("data-job", jobId);
      host.setAttribute("data-n", String(cards.length));
    }).catch(function (e) {
      host.innerHTML = "";
      host.appendChild(el("div", "empty", "样张读取失败：" + e.message));
    });
  }

  // ---------------------------------------------------------------- datasets
  function loadDatasets() {
    get("/api/datasets").then(function (d) {
      var host = $("datasets");
      host.innerHTML = "";
      $("runsHint").textContent = (d.datasets || []).length + " 个运行目录";
      if (!d.datasets || !d.datasets.length) {
        host.appendChild(el("div", "empty", "还没有数据集。"));
        return;
      }
      var wrap = el("div", "tw");
      var t = el("table");
      var hr = el("tr");
      ["名称", "帧数", "模态", "红外", "激光", "雷达", "SAR", "大小 MB", ""].forEach(function (h) {
        hr.appendChild(el("th", "", h));
      });
      t.appendChild(hr);
      d.datasets.slice(0, 15).forEach(function (r) {
        var tr = el("tr");
        tr.appendChild(el("td", "mono", r.id));
        tr.appendChild(el("td", "mono", r.ok ? (r.frames === null ? "—" : r.frames) : "—"));
        tr.appendChild(el("td", "", r.ok ? ((r.modalities || []).join(",") || "—") : "未完成（无 stats.json）"));
        tr.appendChild(el("td", "mono", r.ok ? (r.ir || 0) : "—"));
        tr.appendChild(el("td", "mono", r.ok ? (r.lidar || 0) : "—"));
        tr.appendChild(el("td", "mono", r.ok ? (r.radar || 0) : "—"));
        tr.appendChild(el("td", "mono", r.ok ? (r.sar || 0) : "—"));
        tr.appendChild(el("td", "mono", r.size_mb));
        var td = el("td");
        var a = el("a", "", "看");
        a.href = "#";
        a.onclick = function (e) { e.preventDefault(); active = r.id; showGallery(r.id); };
        td.appendChild(a);
        tr.appendChild(td);
        t.appendChild(tr);
      });
      wrap.appendChild(t);
      host.appendChild(wrap);
      // A freshly restarted server has no jobs in memory but plenty of runs on
      // disk: show the newest one instead of an empty gallery.
      var hasJob = document.querySelectorAll("#jobs .job").length > 0;
      if (!active && !hasJob) {
        active = d.datasets[0].id;
        showGallery(active);
      }
    }).catch(function (e) { alert_(e.message); });
  }

  // ---------------------------------------------------------------- run
  function go(preview) {
    alert_("");
    var btn = preview ? $("btnPreview") : $("btnGen");
    var label = btn.textContent;
    btn.disabled = true;
    btn.textContent = "启动中…";
    post(preview ? "/api/preview" : "/api/generate", payload()).then(function (j) {
      active = j.id;
      $("gallery").removeAttribute("data-job");
      $("gallery").removeAttribute("data-n");
      btn.disabled = false;
      btn.textContent = label;
      clearTimeout(pollTimer);
      poll();
      setTimeout(function () { showGallery(j.id); }, 2500);
      setTimeout(loadDatasets, 3000);
    }).catch(function (e) {
      btn.disabled = false;
      btn.textContent = label;
      alert_(e.message);
    });
  }

  // ---------------------------------------------------------------- init
  function init() {
    bindNum("pxLo", "pxLo", 1, 4000);
    bindNum("pxHi", "pxHi", 1, 4000);
    bindNum("closingMps", "closingMps", 0, 3000);
    bindNum("rngLo", "rngLo", 1, 400000);
    bindNum("rngHi", "rngHi", 1, 400000);
    bindNum("nFov", "nFov", 0.1, 120);
    bindNum("nW", "nW", 64, 4096);
    bindNum("nH", "nH", 64, 4096);
    bindNum("wFov", "wFov", 0.1, 120);
    bindNum("wW", "wW", 64, 4096);
    bindNum("wH", "wH", 64, 4096);
    bindNum("sarBw", "sarBw", 1, 2000);
    bindNum("swathC", "swathC", 10, 200000);
    bindNum("swathW", "swathW", 10, 200000);
    bindNum("sarAlt", "sarAlt", 10, 30000);
    bindNum("sarSpeed", "sarSpeed", 0, 3000);
    bindNum("sarSquint", "sarSquint", -75, 75);
    bindNum("pAlt", "pAlt", 10, 30000);
    bindNum("pSpeed", "pSpeed", 0, 3000);
    bindNum("pHeading", "pHeading", 0, 360);
    bindNum("pDive", "pDive", -89, 89);
    bindNum("pWeave", "pWeave", 0, 500);
    bindNum("pWeaveHz", "pWeaveHz", 0, 5);
    bindNum("pBack", "pBack", 0, 200000);
    bindNum("pStandoff", "pStandoff", 10, 50000);
    bindNum("frames", "frames", 1, 20000);
    bindNum("fps", "fps", 0.5, 120);
    bindNum("fpc", "fpc", 1, 5000);
    bindNum("seed", "seed");
    bindText("name", "name");

    $("sarOn").addEventListener("change", function () {
      S.sar = this.checked;
      if (S.sar && S.sensors.indexOf("sar") < 0) S.sensors.push("sar");
      if (!S.sar) S.sensors = S.sensors.filter(function (s) { return s !== "sar"; });
      chips($("sensors"), sensorItems(), "sensors", true);
      changed();
    });
    $("sarPol").addEventListener("change", function () { S.sarPol = this.value; changed(); });
    $("scene").addEventListener("change", function () {
      S.scene = this.value;
      sceneDefaults();
      changed();
    });
    $("btnPlan").onclick = function () { loadPlan(); };
    $("btnPreview").onclick = function () { go(true); };
    $("btnGen").onclick = function () { go(false); };

    get("/api/presets").then(function (d) {
      OPT.scenes = d.scenes || [];
      (d.sar_bands || []).forEach(function (b) { OPT.sarBands[b.key] = b; });
      OPT.limits = d.limits || {};
      OPT.sizePx = d.size_px || {};
      if (!S.scene && OPT.scenes.length) S.scene = OPT.scenes[0].key;
      renderScenes();
      chips($("size"), [
        { value: "small", label: "小 8–20" },
        { value: "medium", label: "中 18–42" },
        { value: "large", label: "大 35–80" }
      ], "size", false);
      chips($("closing"), [{ value: "in", label: "接近" }, { value: "out", label: "远离" },
                           { value: "hold", label: "保持" }], "closing", false);
      chips($("viewsMode"), [{ value: "single", label: "单视场" },
                             { value: "dual", label: "双视场" }], "viewsMode", false);
      chips($("traj"), [{ value: "platform", label: "平台俯冲 + 蛇形" },
                        { value: "seeker", label: "相机静立 · 目标飞行" },
                        { value: "world", label: "世界轨迹（旧）" }], "traj", false);
      chips($("sarBand"), [{ value: "X", label: "X" }, { value: "C", label: "C" },
                           { value: "Ku", label: "Ku" }], "sarBand", false);
      chips($("weather"), (d.weather || []).map(function (w) {
        return { value: w.key, label: w.label };
      }), "weather", false);
      chips($("sensors"), sensorItems(), "sensors", true);
      $("size").addEventListener("click", function (ev) {
        var b = ev.target.closest("button");
        if (!b) return;
        var px = OPT.sizePx[b.getAttribute("data-v")];
        if (px) {
          S.pxLo = px[0];
          S.pxHi = px[1];
          $("pxLo").value = px[0];
          $("pxHi").value = px[1];
        }
      }, true);
      sceneDefaults();
      refreshGates();
      paintSummary();
      loadPlan();
      poll();
      loadDatasets();
    }).catch(function (e) {
      alert_("读取选项失败：" + e.message);
      $("summary").textContent = "服务未就绪：" + e.message;
    });
  }

  function sensorItems() {
    return [
      { value: "optical", label: "可见光" }, { value: "ir", label: "红外" },
      { value: "radar", label: "雷达" }, { value: "lidar", label: "激光雷达" },
      { value: "sar", label: "SAR" }
    ];
  }

  init();
})();
