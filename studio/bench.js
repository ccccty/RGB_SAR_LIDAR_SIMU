/* Machine benchmark + parallel worker picker.
 *
 * Self-contained on purpose: it injects its own floating card so nothing in
 * app.js has to change, and reads the same /api/bench endpoint the CLI uses.
 * The recommendation is a measurement, not a guess from the core count: the
 * optical pass saturates near 2x because each Blender instance already fills
 * the CPU, while a pure IR batch keeps scaling.
 */
(function () {
  "use strict";
  var state = { workers: 1, recommended: 1, busy: false, result: null };

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function api(path, body) {
    return fetch(path, {
      method: body ? "POST" : "GET",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
        return j;
      });
    });
  }

  function build() {
    var box = el("div", "bench-box");
    box.id = "benchBox";

    var head = el("div", "bench-head");
    head.appendChild(el("span", "bench-title", "性能检测与并行"));
    var toggle = el("button", "bench-min", "–");
    toggle.type = "button";
    toggle.title = "收起";
    head.appendChild(toggle);
    box.appendChild(head);

    var body = el("div", "bench-body");
    box.appendChild(body);

    var hint = el("p", "bench-hint",
      "先测一下这台机器能同时跑几帧，再决定并行度。检测约需 5–20 秒。");
    body.appendChild(hint);

    var run = el("button", "btn small primary", "检测本机性能");
    run.type = "button";
    body.appendChild(run);

    var out = el("div", "bench-out");
    body.appendChild(out);

    var pick = el("div", "bench-pick");
    var lab = el("label", null, "并行度");
    var range = el("input");
    range.type = "range";
    range.min = "1"; range.max = "16"; range.step = "1";
    range.value = "1";
    var num = el("input", "bench-num");
    num.type = "number";
    num.min = "1"; num.max = "16";
    num.value = "1";
    var auto = el("button", "btn small quiet", "按建议值");
    auto.type = "button";
    var note = el("p", "bench-note",
      "并行度会写进生成请求；1 = 单进程（最省内存）。");
    pick.appendChild(lab);
    pick.appendChild(range);
    pick.appendChild(num);
    pick.appendChild(auto);
    body.appendChild(pick);
    body.appendChild(note);

    function setWorkers(v) {
      v = Math.max(1, Math.min(16, parseInt(v, 10) || 1));
      state.workers = v;
      range.value = String(v);
      num.value = String(v);
      try { window.BENCH_WORKERS = v; } catch (e) {}
    }
    range.addEventListener("input", function () { setWorkers(range.value); });
    num.addEventListener("change", function () { setWorkers(num.value); });
    auto.addEventListener("click", function () { setWorkers(state.recommended); });

    toggle.addEventListener("click", function () {
      box.classList.toggle("bench-collapsed");
      toggle.textContent = box.classList.contains("bench-collapsed") ? "+" : "–";
    });

    run.addEventListener("click", function () {
      if (state.busy) return;
      state.busy = true;
      run.disabled = true;
      run.textContent = "检测中…";
      out.textContent = "";
      var t0 = Date.now();
      api("/api/bench", { sensor: "ir", workers: [1, 2, 4, 8] })
        .then(function (d) {
          state.result = d;
          state.recommended = d.recommended;
          out.textContent = "";
          var p = el("p", "bench-sum",
            "推荐 " + d.recommended + " 路并行（上限 " + d.ceiling + "）");
          out.appendChild(p);
          var tbl = el("table", "bench-tbl");
          var hr = el("tr");
          ["并行", "每帧", "加速"].forEach(function (h) {
            hr.appendChild(el("th", null, h));
          });
          tbl.appendChild(hr);
          var base = null;
          (d.bench || []).forEach(function (r) {
            if (r.per_frame && base == null) base = r.per_frame;
            var tr = el("tr");
            tr.appendChild(el("td", null, r.workers + " 路"));
            tr.appendChild(el("td", null, r.per_frame ? r.per_frame.toFixed(2) + " s" : "—"));
            tr.appendChild(el("td", null,
              r.per_frame && base ? (base / r.per_frame).toFixed(2) + "x" : "—"));
            if (r.workers === d.recommended) tr.className = "bench-best";
            tbl.appendChild(tr);
          });
          out.appendChild(tbl);
          var info = el("p", "bench-hint",
            (d.probe ? d.probe.logical_cores + " 逻辑核 · " +
              Math.round(d.probe.ram_available_mb / 1024) + " GB 可用" : "") +
            " · 用时 " + ((Date.now() - t0) / 1000).toFixed(1) + " 秒");
          out.appendChild(info);
          (d.reason || []).forEach(function (line) {
            out.appendChild(el("p", "bench-why", "· " + line));
          });
          setWorkers(d.recommended);
        })
        .catch(function (e) {
          out.textContent = "";
          out.appendChild(el("p", "bench-err", "检测失败：" + e.message));
        })
        .then(function () {
          state.busy = false;
          run.disabled = false;
          run.textContent = "重新检测";
        });
    });

    return box;
  }

  function mount() {
    if (document.getElementById("benchBox")) return;
    document.body.appendChild(build());
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();