/* simkit studio — 三模态对齐面板
   一次运行结束后，把同一帧同一目标的光学框、红外框、雷达距离/径向速度/SNR
   并排放在一张表里，直接回答“三个模态是不是都看到了这个目标”。

   数据全部来自运行目录里已经写好的 labels/*.json、annotations_radar.json 与
   meta.json；这里不重新投影、不重新渲染，也不做检测。这样面板上的一行就是
   数据集里真实存在的一行标注，而不是界面对仿真结果的第二种解释。

   本文件独立于 app.js（后者是 3000 多行的历史工作台），只通过
   window.STUDIO_ALIGN.mount(host, api, runId) 接入，避免把表格逻辑塞进闭包里。 */
(function (global) {
  "use strict";

  var PAGE = 20;

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function clear(node) {
    while (node && node.firstChild) { node.removeChild(node.firstChild); }
  }

  function ajax(url, cb) {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", url, true);
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) { return; }
      var data = null;
      try { data = JSON.parse(xhr.responseText || "null"); } catch (e) { data = null; }
      if (xhr.status >= 200 && xhr.status < 300) { cb(null, data); return; }
      cb(data || { error: "无法读取 " + url }, null);
    };
    xhr.onerror = function () { cb({ error: "连接中断：" + url }, null); };
    xhr.send(null);
  }

  function fmt(v, digits) {
    if (v === null || v === undefined || v === "") { return "—"; }
    var n = Number(v);
    if (!isFinite(n)) { return "—"; }
    var d = digits === undefined ? 1 : digits;
    var s = n.toFixed(d);
    if (s.indexOf(".") >= 0) { s = s.replace(/0+$/, "").replace(/\.$/, ""); }
    return s;
  }

  function pxText(box) {
    return fmt(box.center[0], 1) + "," + fmt(box.center[1], 1) + " · " +
      fmt(box.size[0], 1) + "×" + fmt(box.size[1], 1);
  }

  /* --------------------------------------------------------------- 对齐表 */

  function renderTable(state, body) {
    var data = state.data || {};
    var all = data.frames || [];
    var only = state.onlyMiss;
    var rows = [];
    var i, k;

    if (only) {
      for (i = 0; i < all.length; i++) {
        var kept = [];
        for (k = 0; k < all[i].rows.length; k++) {
          if (!all[i].rows[k].all3) { kept.push(all[i].rows[k]); }
        }
        if (kept.length) { rows.push({ frame: all[i].frame, t: all[i].t, rows: kept }); }
      }
    } else {
      rows = all;
    }

    var table = el("table", "align-table");
    var thead = el("thead");
    var tr = el("tr");
    var heads = ["目标", "光学 中心 / 尺寸 (px)", "红外 中心 / 尺寸 (px)",
                 "雷达距离 (m)", "径向速度 (m/s)", "SNR (dB)", "三模态"];
    for (i = 0; i < heads.length; i++) { tr.appendChild(el("th", "", heads[i])); }
    thead.appendChild(tr);
    table.appendChild(thead);

    var tbody = el("tbody");
    var shownRows = 0;
    for (i = 0; i < rows.length; i++) {
      var fr = rows[i];
      var ok = 0;
      for (k = 0; k < fr.rows.length; k++) { if (fr.rows[k].all3) { ok++; } }
      var head = el("tr", "align-frame");
      var td = el("td", "", "帧 " + fr.frame +
        (fr.t === null || fr.t === undefined ? "" : " · t=" + fmt(fr.t, 3) + " s") +
        " · " + ok + "/" + fr.rows.length + " 目标三模态对齐");
      td.colSpan = heads.length;
      head.appendChild(td);
      tbody.appendChild(head);

      for (k = 0; k < fr.rows.length; k++) {
        var row = fr.rows[k];
        var rtr = el("tr", row.all3 ? "align-row" : "align-row miss");
        shownRows++;

        var nameCell = el("td", "align-name");
        nameCell.appendChild(el("b", "", row.name));
        if (row.cls) { nameCell.appendChild(el("span", "align-cls", row.cls)); }

        var optCell = el("td", "");
        if (!row.optical || !row.optical.length) {
          optCell.appendChild(el("span", "align-miss", "—"));
        } else {
          for (var oi = 0; oi < row.optical.length; oi++) {
            var ob = row.optical[oi];
            var ol = el("div", ob.visible ? "align-val" : "align-val dim");
            ol.textContent = (row.optical.length > 1 || ob.view !== "seeker" ? ob.view + " " : "") +
              pxText(ob) + (ob.px !== null && ob.px !== undefined ? " · " + ob.px + " px" : "");
            optCell.appendChild(ol);
          }
        }

        var irCell = el("td", "");
        if (!row.ir || !row.ir.length) {
          irCell.appendChild(el("span", "align-miss", "—"));
        } else {
          for (var ii = 0; ii < row.ir.length; ii++) {
            var ib = row.ir[ii];
            var il = el("div", ib.visible ? "align-val" : "align-val dim");
            il.textContent = (ib.band || "ir") + " " + pxText(ib);
            irCell.appendChild(il);
          }
        }

        var rad = row.radar;
        var rangeCell = el("td", "");
        var vrCell = el("td", "");
        var snrCell = el("td", "");
        if (!rad) {
          rangeCell.appendChild(el("span", "align-miss", "—"));
          vrCell.appendChild(el("span", "align-miss", "—"));
          snrCell.appendChild(el("span", "align-miss", "—"));
        } else {
          var rv = rad.detected && rad.measured_range_m !== null
            ? rad.measured_range_m : rad.range_m;
          var rangeLine = el("div", "align-val" + (rad.detected ? "" : " dim"));
          rangeLine.textContent = fmt(rv, 1) + (rad.detected ? "" : " · 未检出");
          rangeCell.appendChild(rangeLine);
          var vv = rad.detected && rad.measured_v_r !== null ? rad.measured_v_r : rad.v_r;
          var vrLine = el("div", "align-val" + (rad.detected ? "" : " dim"));
          vrLine.textContent = fmt(vv, 2);
          vrCell.appendChild(vrLine);
          snrCell.appendChild(el("div", "align-val" +
            (rad.detected && rad.snr_db !== null ? "" : " dim"),
            rad.detected && rad.snr_db !== null ? fmt(rad.snr_db, 1) : "—"));
          var tip = "距离真值 " + fmt(rad.range_m, 1) + " m · 径向速度真值 " +
            fmt(rad.v_r, 2) + " m/s";
          if (rad.az_deg !== null) { tip += " · 方位 " + fmt(rad.az_deg, 2) + "°"; }
          if (rad.el_deg !== null) { tip += " · 俯仰 " + fmt(rad.el_deg, 2) + "°"; }
          if (rad.rcs_dbsm !== null) { tip += " · RCS " + fmt(rad.rcs_dbsm, 1) + " dBsm"; }
          rangeCell.title = tip;
        }

        var matchCell = el("td", "");
        if (row.all3) {
          matchCell.appendChild(el("span", "align-ok", "✓ 三模态"));
        } else {
          var missing = [];
          var det = row.detected || {};
          if (!det.optical) { missing.push("光学"); }
          if (!det.ir) { missing.push("红外"); }
          if (!det.radar) { missing.push("雷达"); }
          var bad = el("span", "align-bad", "✗ 缺 " + missing.join("、"));
          bad.title = "光学" + (det.optical ? "✓" : "✗") +
            " 红外" + (det.ir ? "✓" : "✗") + " 雷达" + (det.radar ? "✓" : "✗");
          matchCell.appendChild(bad);
        }

        rtr.appendChild(nameCell);
        rtr.appendChild(optCell);
        rtr.appendChild(irCell);
        rtr.appendChild(rangeCell);
        rtr.appendChild(vrCell);
        rtr.appendChild(snrCell);
        rtr.appendChild(matchCell);
        tbody.appendChild(rtr);
      }
    }
    if (!shownRows) {
      var empty = el("tr", "");
      var etd = el("td", "align-empty", state.loading ? "正在读取…"
        : (only ? "当前页没有未对齐的目标" : "当前页没有可读的逐帧标签"));
      etd.colSpan = heads.length;
      empty.appendChild(etd);
      tbody.appendChild(empty);
    }
    table.appendChild(tbody);
    body.appendChild(table);
  }

  function renderChips(state, host) {
    var data = state.data || {};
    var t = data.totals || {};
    var chips = [];
    chips.push(["三模态 " + fmt(t.all3, 0) + " / " + fmt(t.rows, 0), (t.all3 || 0) === (t.rows || 0)]);
    chips.push(["光学 " + fmt(t.optical, 0), (t.optical || 0) === (t.rows || 0)]);
    chips.push(["红外 " + fmt(t.ir, 0), (t.ir || 0) === (t.rows || 0)]);
    chips.push(["雷达 " + fmt(t.radar, 0), (t.radar || 0) === (t.rows || 0)]);
    var a = data.annotations || {};
    if (a.boxes) { chips.push(["RD 标注 " + a.boxes + " 框", true]); }
    var mods = data.modalities || {};
    if (!mods.optical || !mods.optical.length) { chips.push(["未生成 光学", false]); }
    if (!mods.ir || !mods.ir.length) { chips.push(["未生成 红外", false]); }
    if (!mods.radar) { chips.push(["未生成 雷达", false]); }
    var wrap = el("div", "align-chips");
    for (var i = 0; i < chips.length; i++) {
      wrap.appendChild(el("span", "align-chip" + (chips[i][1] ? " ok" : " bad"), chips[i][0]));
    }
    host.appendChild(wrap);
  }

  /* ----------------------------------------------------------- 雷达双图 */

  function renderRadar(state, host) {
    var data = state.data || {};
    var pairs = data.radar_pairs || [];
    if (!pairs.length) {
      var none = el("p", "align-note",
        "这次运行没有同时写出 images/radar（标注）与 images/radar_clean（纯净），无法做同帧对照。");
      host.appendChild(none);
      return;
    }
    if (state.radarIdx >= pairs.length) { state.radarIdx = 0; }
    var pair = pairs[state.radarIdx];

    var tools = el("div", "radar-tools");
    var seg = el("div", "seg small");
    var modes = [["pair", "并排"], ["labeled", "只标注"], ["clean", "只纯净"]];
    for (var m = 0; m < modes.length; m++) {
      (function (mode, label) {
        var btn = el("button", state.radarMode === mode ? "on" : "", label);
        btn.type = "button";
        btn.onclick = function () { state.radarMode = mode; rerender(state); };
        seg.appendChild(btn);
      })(modes[m][0], modes[m][1]);
    }
    tools.appendChild(seg);

    var prev = el("button", "btn quiet small", "◀ 上一帧");
    prev.type = "button";
    prev.disabled = state.radarIdx <= 0;
    prev.onclick = function () { if (state.radarIdx > 0) { state.radarIdx--; rerender(state); } };
    var next = el("button", "btn quiet small", "下一帧 ▶");
    next.type = "button";
    next.disabled = state.radarIdx >= pairs.length - 1;
    next.onclick = function () {
      if (state.radarIdx < pairs.length - 1) { state.radarIdx++; rerender(state); }
    };
    var sel = el("select", "radar-frame");
    for (var i = 0; i < pairs.length; i++) {
      var opt = el("option", "", "帧 " + pairs[i].frame);
      opt.value = String(i);
      if (i === state.radarIdx) { opt.selected = true; }
      sel.appendChild(opt);
    }
    sel.onchange = function () {
      var idx = parseInt(sel.value, 10);
      if (isFinite(idx)) { state.radarIdx = Math.max(0, Math.min(idx, pairs.length - 1)); rerender(state); }
    };
    tools.appendChild(prev);
    tools.appendChild(sel);
    tools.appendChild(next);
    host.appendChild(tools);

    var wrap = el("div", "radar-pair" + (state.radarMode === "pair" ? "" : " single"));
    function figure(url, caption, note, dim) {
      var fig = el("figure", "radar-fig");
      if (dim) { fig.setAttribute("hidden", "hidden"); }
      var link = el("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener";
      var img = el("img");
      img.src = url + "?t=" + Date.now();
      img.alt = caption;
      img.loading = "lazy";
      link.appendChild(img);
      fig.appendChild(link);
      var cap = el("figcaption");
      cap.appendChild(el("b", "", caption));
      cap.appendChild(el("span", "align-cls", note));
      fig.appendChild(cap);
      return fig;
    }
    wrap.appendChild(figure(pair.labeled, "标注",
      "images/radar/" + pair.file + (pair.w ? " · " + pair.w + "×" + pair.h : ""),
      state.radarMode === "clean"));
    wrap.appendChild(figure(pair.clean, "纯净",
      "images/radar_clean/" + pair.file,
      state.radarMode === "labeled"));
    host.appendChild(wrap);
  }

  /* --------------------------------------------------------------- 组装 */

  function render(state) {
    var body = state.body;
    clear(body);
    var data = state.data;
    if (!data) {
      body.appendChild(el("p", "pf-idle", state.error ? state.error : "正在读取三模态标签…"));
      if (state.error) {
        var raw = el("a", "link", "打开原始 JSON");
        raw.href = state.api + "/align/" + encodeURIComponent(state.runId);
        raw.target = "_blank";
        raw.rel = "noopener";
        body.appendChild(raw);
      }
      return;
    }

    var meta = data.meta || {};
    var sub = [];
    if (data.frames_total) { sub.push(data.frames_total + " 帧"); }
    if (data.objects && data.objects.length) { sub.push(data.objects.length + " 目标"); }
    var views = meta.views || [];
    for (var i = 0; i < views.length && i < 2; i++) {
      if (views[i] && views[i].name) {
        sub.push(views[i].name + " " + fmt(views[i].fov_deg, 1) + "°");
      }
    }
    if (meta.ir_bands && meta.ir_bands.length) { sub.push("红外 " + meta.ir_bands.join("/")); }
    if (meta.radar_band) { sub.push("雷达 " + meta.radar_band); }
    if (state.data.shown) {
      sub.push("本页第 " + (data.start + 1) + "–" + data.end + " 帧");
    }
    var head = el("div", "align-head");
    head.appendChild(el("h3", "", "三模态对齐"));
    head.appendChild(el("span", "align-sub", sub.join(" · ")));
    body.appendChild(head);

    renderChips(state, body);

    var tools = el("div", "align-tools");
    var total = data.frames_total || 0;
    var prev = el("button", "btn quiet small", "◀ 上一页");
    prev.type = "button";
    prev.disabled = data.start <= 0;
    prev.onclick = function () { state.from = Math.max(0, data.start - PAGE); load(state); };
    var next = el("button", "btn quiet small", "下一页 ▶");
    next.type = "button";
    next.disabled = data.end >= total;
    next.onclick = function () { state.from = data.end; load(state); };
    tools.appendChild(prev);
    tools.appendChild(el("span", "align-range",
      total ? "第 " + (data.start + 1) + "–" + data.end + " / 共 " + total + " 帧" : "无帧"));
    tools.appendChild(next);

    var only = el("button", "chip" + (state.onlyMiss ? " on" : ""), "只看未对齐");
    only.type = "button";
    only.onclick = function () {
      state.onlyMiss = !state.onlyMiss;
      render(state);
    };
    tools.appendChild(only);
    tools.appendChild(el("span", "align-legend",
      "✓ = 该模态在本帧有检出；三模态 = 光学可见 且 红外可见 且 雷达 detected"));
    body.appendChild(tools);

    renderTable(state, body);

    var notes = data.notes || [];
    for (i = 0; i < notes.length; i++) { body.appendChild(el("p", "align-note", notes[i])); }
    body.appendChild(el("p", "align-note",
      "判定口径直接读运行目录：光学/红外取 labels/*/ 里的 boxes.visible，雷达取 targets[].detected；" +
      "未检出的雷达目标仍列出真值距离与径向速度（SNR 记 —）。"));

    var radar = el("div", "radar-block");
    var rhead = el("div", "align-head");
    rhead.appendChild(el("h3", "", "雷达距离-多普勒 · 标注 / 纯净"));
    rhead.appendChild(el("span", "align-sub", "同帧对照：images/radar 与 images/radar_clean"));
    radar.appendChild(rhead);
    var rbody = el("div", "radar-body");
    radar.appendChild(rbody);
    state.radarBody = rbody;
    renderRadar(state, rbody);
    body.appendChild(radar);
  }

  function rerender(state) {
    /* The radar block is redrawn on its own so switching 并排/标注/纯净 does
       not rebuild (and re-scroll) the alignment table above it. */
    var block = state.radarBody;
    if (!block) { render(state); return; }
    clear(block);
    renderRadar(state, block);
  }

  function load(state) {
    if (state.loading) { return; }
    state.loading = true;
    if (!state.data) { render(state); }
    var url = state.api + "/align/" + encodeURIComponent(state.runId) +
      "?from=" + state.from + "&frames=" + PAGE;
    ajax(url, function (err, data) {
      state.loading = false;
      if (err) {
        state.data = null;
        state.error = err.error || "无法读取三模态对齐数据";
        render(state);
        return;
      }
      state.data = data;
      state.error = "";
      state.radarIdx = 0;
      render(state);
    });
  }

  function mount(host, api, runId) {
    if (!host) { return; }
    var section = el("section", "align-block");
    host.appendChild(section);
    var state = {
      api: api, runId: runId, from: 0, onlyMiss: false,
      data: null, error: "", loading: false, radarMode: "pair", radarIdx: 0,
      body: section, radarBody: null
    };
    load(state);
    return section;
  }

  global.STUDIO_ALIGN = { mount: mount };
})(window);
