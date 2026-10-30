(function () {
  "use strict";
  var state = { preset: "air_uav", size: "medium", closing: "in", weather: "mixed",
                sensors: ["ir"], band: "Ka", frames: 300, fps: 10, fpc: 50, seed: 7,
                name: "" };
  var presets = {};
  var active = null;
  var pollTimer = null;

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt !== undefined) e.textContent = txt;
    return e;
  }
  function post(url, body) {
    return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(body) }).then(function (r) { return r.json(); });
  }
  function get(url) { return fetch(url).then(function (r) { return r.json(); }); }

  function chips(host, items, key, multi) {
    host.innerHTML = "";
    items.forEach(function (it) {
      var b = el("button", "", it.label);
      b.setAttribute("data-v", it.value);
      var on = multi ? (state[key].indexOf(it.value) >= 0) : (state[key] === it.value);
      if (on) b.className = "on";
      b.onclick = function () {
        if (multi) {
          var i = state[key].indexOf(it.value);
          if (i >= 0) { if (state[key].length > 1) state[key].splice(i, 1); }
          else state[key].push(it.value);
        } else {
          state[key] = it.value;
        }
        chips(host, items, key, multi);
        refreshEstimate();
      };
      host.appendChild(b);
    });
  }

  function selected() {
    return { preset: state.preset, size: state.size, closing: state.closing,
             weather: state.weather, sensors: state.sensors, radar_band: state.band,
             frames: state.frames, fps: state.fps, frames_per_clip: state.fpc,
             seed: state.seed, name: state.name };
  }

  function readNumbers() {
    state.frames = Math.max(8, parseInt($("frames").value, 10) || 300);
    state.fps = Math.max(1, parseFloat($("fps").value) || 10);
    state.fpc = Math.max(2, parseInt($("fpc").value, 10) || 50);
    state.seed = parseInt($("seed").value, 10) || 0;
    state.name = $("name").value || "";
  }

  var RATE = { air_uav: 7.0, ground_vehicle: 9.5, sea_boat: 6.0 };

  function refreshEstimate() {
    readNumbers();
    var n = Math.floor(state.frames / Math.max(1, state.fpc));
    var per = RATE[state.preset] || 8;
    var mins = state.frames * per / 60.0;
    var sensors = state.sensors.length;
    mins = mins * (0.55 + 0.18 * sensors);
    $("estimate").textContent = "约 " + n + " 个连续片段 × " + state.fpc +
      " 帧（" + (state.fpc / state.fps).toFixed(1) + " 秒/段），预计 " +
      (mins < 1 ? "不到 1" : mins.toFixed(0)) + " 分钟（单进程；分块并行会更快）";
  }

  function renderPresets() {
    var items = Object.keys(presets).map(function (k) {
      return { value: k, label: presets[k].title };
    });
    chips($("preset"), items, "preset", false);
    var note = presets[state.preset] ? presets[state.preset].note : "";
    $("sizeHint").textContent = note + " · 目标在画面里“差不多能看见”，不是占满画面";
  }

  function jobCard(j) {
    var card = el("div", "job");
    var top = el("div", "top");
    top.appendChild(el("span", "id", j.id));
    var st = el("span", "state " + j.state, j.state);
    top.appendChild(st);
    card.appendChild(top);
    var bar = el("div", "bar");
    var fill = el("i");
    var pct = j.total ? Math.min(100, Math.round(100 * j.done / j.total)) : 0;
    fill.style.width = pct + "%";
    bar.appendChild(fill);
    card.appendChild(bar);
    var meta = el("div", "hint", j.done + "/" + (j.total || "?") + " 帧 · " +
      j.seconds + " 秒" + (j.error ? " · " + j.error : ""));
    card.appendChild(meta);
    if (j.log && j.log.length) {
      card.appendChild(el("pre", "log", j.log.slice(-10).join("\n")));
    }
    if (j.state === "running") {
      var b = el("button", "ghost", "停止");
      b.style.marginTop = "6px";
      b.onclick = function () {
        fetch("/api/job/" + j.id + "/cancel", { method: "POST" }).then(poll);
      };
      card.appendChild(b);
    }
    return card;
  }

  function poll() {
    get("/api/jobs").then(function (d) {
      var host = $("jobs");
      host.innerHTML = "";
      if (!d.jobs || !d.jobs.length) {
        host.appendChild(el("div", "empty", "还没有任务。选好参数后点“生成数据集”。"));
        return;
      }
      d.jobs.slice(0, 6).forEach(function (j) { host.appendChild(jobCard(j)); });
      var running = d.jobs.filter(function (j) { return j.state === "running"; })[0];
      var focus = active || (d.jobs[0] && d.jobs[0].id);
      var fjob = d.jobs.filter(function (j) { return j.id === focus; })[0];
      // Re-render until the run has finished: an early render sees an empty
      // directory and would otherwise be cached forever by data-job.
      var ready = fjob && fjob.state === "done";
      if (focus && ($("gallery").getAttribute("data-job") !== focus || !ready)) {
        showGallery(focus, !ready);
      }
      if (running !== undefined) {
        setTimeout(poll, 1500);
      } else if (!ready && focus) {
        setTimeout(poll, 2500);
      }
    });
  }

  function showGallery(jobId, quiet) {
    get("/api/dataset/" + jobId).then(function (d) {
      if (!d || !d.samples) return;
      var count = 0;
      Object.keys(d.samples).forEach(function (k) { count += (d.samples[k] || []).length; });
      if (!count) return;                 // nothing written yet; try again later
      var host = $("gallery");
      host.innerHTML = "";
      host.setAttribute("data-job", jobId);
      var head = el("div", "hint", (quiet ? "生成中… " : "") + "结果预览 · " + jobId + " · " + d.dir);
      host.appendChild(head);
      var grid = el("div", "gal");
      var names = { optical: "可见光", mwir: "中波红外", lwir: "长波红外",
                    radar_ra: "雷达 距离-角度", lidar: "激光雷达距离" };
      Object.keys(d.samples).forEach(function (mod) {
        var files = d.samples[mod] || [];
        if (!files.length) return;
        var pick = files[Math.min(2, files.length - 1)];
        var fig = el("figure");
        var img = el("img");
        img.loading = "lazy";
        img.src = "/api/img/" + jobId + "/" + (mod === "lidar" ? "lidar/range" :
                  mod === "mwir" ? "images/ir_preview_mwir" :
                  mod === "lwir" ? "images/ir_preview_lwir" :
                  mod === "radar_ra" ? "images/radar_ra" : "images/optical") + "/" + pick;
        fig.appendChild(img);
        fig.appendChild(el("figcaption", "", names[mod] + " · " + pick));
        grid.appendChild(fig);
      });
      host.appendChild(grid);
    });
  }

  function loadPlan() {
    readNumbers();
    get("/api/plan?preset=" + encodeURIComponent(state.preset) + "&frames=" + state.frames)
      .then(function (d) {
        var host = $("plan");
        host.innerHTML = "";
        if (!d.clips || !d.clips.length) {
          host.appendChild(el("div", "empty", "（" + (d.error || "没有片段") + "）"));
          return;
        }
        var t = el("table");
        var hr = el("tr");
        ["片段", "天气", "起始距离 m", "结束距离 m", "起始像素", "结束像素", "速度 m/s", "加速度 m/s²"]
          .forEach(function (h) { hr.appendChild(el("th", "", h)); });
        t.appendChild(hr);
        d.clips.slice(0, 12).forEach(function (c) {
          var tr = el("tr");
          [c.clip, c.weather, c.range_m[0], c.range_m[1], c.target_px[0], c.target_px[1],
           c.speed_mps, c.accel_mps2].forEach(function (v) {
            tr.appendChild(el("td", "mono", String(v)));
          });
          t.appendChild(tr);
        });
        host.appendChild(t);
      });
  }

  function loadDatasets() {
    get("/api/datasets").then(function (d) {
      var host = $("datasets");
      host.innerHTML = "";
      if (!d.datasets || !d.datasets.length) {
        host.appendChild(el("div", "empty", "还没有数据集。"));
        return;
      }
      var t = el("table");
      var hr = el("tr");
      ["名称", "帧数", "模态", "红外图", "激光图", "雷达帧", "大小 MB", ""].forEach(function (h) {
        hr.appendChild(el("th", "", h));
      });
      t.appendChild(hr);
      d.datasets.slice(0, 12).forEach(function (r) {
        var tr = el("tr");
        tr.appendChild(el("td", "", r.id));
        tr.appendChild(el("td", "mono", String(r.frames)));
        tr.appendChild(el("td", "", (r.modalities || []).join(",")));
        tr.appendChild(el("td", "mono", String(r.ir || 0)));
        tr.appendChild(el("td", "mono", String(r.lidar || 0)));
        tr.appendChild(el("td", "mono", String(r.radar || 0)));
        tr.appendChild(el("td", "mono", String(r.size_mb)));
        var td = el("td");
        var a = el("a", "", "看");
        a.href = "#";
        a.onclick = function (e) { e.preventDefault(); showGallery(r.id); };
        td.appendChild(a);
        tr.appendChild(td);
        t.appendChild(tr);
      });
      host.appendChild(t);
    });
  }

  function go(preview) {
    readNumbers();
    var btn = preview ? $("btnPreview") : $("btnGen");
    btn.disabled = true;
    var label = btn.textContent;
    btn.textContent = "启动中…";
    post(preview ? "/api/preview" : "/api/generate", selected()).then(function (j) {
      active = j.id;
      $("gallery").removeAttribute("data-job");
      btn.disabled = false;
      btn.textContent = label;
      poll();
      setTimeout(function () { showGallery(j.id); }, 2500);
      setTimeout(poll, 4000);
      setTimeout(loadDatasets, 6000);
    }).catch(function () {
      btn.disabled = false;
      btn.textContent = label;
    });
  }

  function init() {
    get("/api/presets").then(function (d) {
      (d.presets || []).forEach(function (p) { presets[p.key] = p; });
      renderPresets();
      chips($("weather"), (d.weather || []).map(function (w) {
        var lbl = { clear: "晴", cloudy: "多云", rain: "雨", fog: "雾",
                    mixed: "混合", evolve: "演变" }[w] || w;
        return { value: w, label: lbl };
      }), "weather", false);
      chips($("sensors"), (d.sensors || []).map(function (s) {
        var lbl = { optical: "可见光", ir: "红外", radar: "雷达", lidar: "激光雷达" }[s] || s;
        return { value: s, label: lbl };
      }), "sensors", true);
      var sel = $("band");
      (d.radar_bands || []).forEach(function (b) {
        var o = el("option", "", b);
        o.value = b;
        if (b === state.band) o.selected = true;
        sel.appendChild(o);
      });
      sel.onchange = function () { state.band = sel.value; };
      refreshEstimate();
      loadPlan();
      loadDatasets();
      poll();
    });

    $("band").addEventListener("change", refreshEstimate);
    ["frames", "fps", "fpc", "seed"].forEach(function (id) {
      $(id).addEventListener("change", function () { refreshEstimate(); loadPlan(); });
    });
    $("btnPreview").onclick = function () { go(true); };
    $("btnGen").onclick = function () { go(false); };
    $("preset").addEventListener("click", function () {
      setTimeout(function () { loadPlan(); }, 0);
    });
  }

  init();
})();
