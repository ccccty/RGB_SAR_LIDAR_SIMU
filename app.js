/* simkit web app -- no build step, no dependencies, works offline. */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var el = function (tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt !== undefined) e.textContent = txt;
    return e;
  };
  var f1 = function (v) { return (Math.round(v * 10) / 10).toFixed(1); };
  var f2 = function (v) { return (Math.round(v * 100) / 100).toFixed(2); };
  var f0 = function (v) { return String(Math.round(v)); };

  var MOD_LABEL = {
    optical: '可见光', mwir: '中波红外', lwir: '长波红外',
    radar_ra: '雷达距离-角度', radar: '雷达距离-多普勒',
    lidar_range: '激光距离', lidar_intensity: '激光强度'
  };
  var MODS_ALL = ['optical', 'mwir', 'lwir', 'radar_ra', 'radar', 'lidar_range', 'lidar_intensity'];
  // Only offer modalities that are actually in the manifest: the range-Doppler
  // image is written by the kernel but is a 1600x64 strip with a 2 px target, so
  // export_web.py leaves it out of the site on purpose. An empty tab is worse
  // than no tab.
  var MODS = MODS_ALL.slice();
  var WNAME = { clear: '晴', cloudy: '阴', rain: '雨', fog: '雾', storm: '暴风雨' };

  var MAN = null;
  var S = { scene: 0, frame: 0, mod: 'optical' };

  // ---------- tabs ----------
  var TABS = ['gallery', 'compare', 'lidar', 'radar', 'explore', 'about'];
  function showTab(name) {
    TABS.forEach(function (t) {
      var sec = $(t);
      if (sec) sec.classList.toggle('on', t === name);
    });
    Array.prototype.forEach.call(document.querySelectorAll('#tabs button'), function (b) {
      b.classList.toggle('on', b.getAttribute('data-tab') === name);
    });
    if (name === 'lidar') setTimeout(resizePC, 30);
    if (name === 'radar') setTimeout(function () { drawRadar(); drawRdMark(curFrame()); }, 30);
    if (name === 'explore') setTimeout(function () { if (BANDS) exploreUpdate(); }, 40);
    if (name === 'compare') setTimeout(renderCompare, 30);
  }
  Array.prototype.forEach.call(document.querySelectorAll('#tabs button'), function (b) {
    b.addEventListener('click', function () { showTab(b.getAttribute('data-tab')); });
  });

  // ---------- helpers ----------
  function seg(host, items, cur, onPick) {
    host.innerHTML = '';
    items.forEach(function (it) {
      var b = el('button', it.id === cur ? 'on' : '', it.label);
      b.addEventListener('click', function () { onPick(it.id); });
      host.appendChild(b);
    });
  }
  function sceneList() {
    return MAN.scenes.map(function (s, i) { return { id: String(i), label: s.title }; });
  }
  function curScene() { return MAN.scenes[S.scene]; }
  function curFrame() {
    var sc = curScene();
    return sc.frames[Math.min(S.frame, sc.frames.length - 1)];
  }
  function weatherText(fr) {
    var w = fr.weather || {};
    var nm = WNAME[w.name] || WNAME[w.preset] || w.name || '—';
    return nm + ' · 云量 ' + f0((w.cloud_cover || 0) * 100) + '% · 雨 ' +
      f1(w.rain_mmh || 0) + ' mm/h · 能见度 ' + f1((w.visibility_m || 0) / 1000) + ' km';
  }
  function boxSvg(svg, boxes, w, h) {
    svg.innerHTML = '';
    if (!boxes || !boxes.length) return;
    var NS = 'http://www.w3.org/2000/svg';
    svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    boxes.forEach(function (b) {
      var r = document.createElementNS(NS, 'rect');
      r.setAttribute('x', b.x0); r.setAttribute('y', b.y0);
      r.setAttribute('width', Math.max(b.x1 - b.x0, 1));
      r.setAttribute('height', Math.max(b.y1 - b.y0, 1));
      r.setAttribute('fill', 'none');
      r.setAttribute('stroke', '#37c2ff');
      r.setAttribute('stroke-width', Math.max(w / 320, 1.2));
      svg.appendChild(r);
      var t = document.createElementNS(NS, 'text');
      t.setAttribute('x', b.x0 + 2);
      t.setAttribute('y', Math.max(b.y0 - 4, 10));
      t.setAttribute('fill', '#37c2ff');
      t.setAttribute('font-size', Math.max(w / 48, 9));
      t.setAttribute('font-family', 'monospace');
      t.textContent = b.label || 'target';
      svg.appendChild(t);
    });
  }

  // ---------- gallery ----------
  function galSync() {
    var sc = curScene(), fr = curFrame();
    seg($('galScene'), sceneList(), String(S.scene), function (id) {
      S.scene = parseInt(id, 10); S.frame = 0; galSync();
    });
    seg($('galMod'), MODS.map(function (m) {
      return { id: m, label: MOD_LABEL[m] };
    }), S.mod, function (m) { S.mod = m; galSync(); });

    var entry = fr.mods[S.mod];
    var img = $('galImg');
    if (entry) {
      img.style.display = 'block';
      img.src = 'data/' + entry.img;
      img.onload = function () { boxSvg($('galSvg'), entry.boxes, img.naturalWidth, img.naturalHeight); };
      boxSvg($('galSvg'), entry.boxes, entry.w, entry.h);
    } else {
      img.style.display = 'none';
      $('galSvg').innerHTML = '';
    }
    $('galTL').textContent = sc.title + ' · 帧 ' + (fr.i + 1) + '/' + sc.frames.length + '\n' +
      MOD_LABEL[S.mod];
    $('galBR').textContent = entry ? (entry.w + '×' + entry.h) : '无数据';
    $('galRange').max = String(sc.frames.length - 1);
    $('galRange').value = String(fr.i);
    $('galCount').textContent = (fr.i + 1) + ' / ' + sc.frames.length;

    var info = $('galInfo');
    info.innerHTML = '';
    [['天气', weatherText(fr)],
     ['标注框', entry && entry.boxes ? entry.boxes.length + ' 个' : '—'],
     ['雷达检出', fr.radar ? fr.radar.n + ' 点 / 杂波 ' + fr.radar.clutter : '—'],
     ['点云', fr.cloud ? fr.cloud.shown + ' / ' + fr.cloud.n + ' 点' : '—']
    ].forEach(function (kv) {
      var d = el('div', 'kv');
      d.appendChild(el('span', '', kv[0]));
      d.appendChild(el('span', '', String(kv[1])));
      info.appendChild(d);
    });

    var si = $('galSceneInfo');
    si.innerHTML = '';
    [['目标类别', sc.title.split('·')[1] ? sc.title.split('·')[1].trim() : sc.cls],
     ['雷达波段', sc.band + ' 频段'],
     ['帧数', sc.frames.length + ' 帧（晴/阴/雨循环）'],
     ['模态', sc.frames[0] && Object.keys(sc.frames[0].mods).length + ' 路']
    ].forEach(function (kv) {
      var d = el('div', 'kv');
      d.appendChild(el('span', '', kv[0]));
      d.appendChild(el('span', '', String(kv[1])));
      si.appendChild(d);
    });

    var th = $('galThumbs');
    th.innerHTML = '';
    sc.frames.forEach(function (f) {
      var e = f.mods[S.mod];
      if (!e) return;
      var box = el('div', 'card');
      box.style.padding = '6px';
      box.style.cursor = 'pointer';
      if (f.i === fr.i) box.style.borderColor = 'var(--accent)';
      var im = el('img');
      im.src = 'data/' + e.img;
      im.style.width = '100%';
      im.style.borderRadius = '8px';
      im.style.display = 'block';
      box.appendChild(im);
      var cap = el('div', 'mono muted', '帧 ' + (f.i + 1));
      cap.style.fontSize = '11px';
      cap.style.marginTop = '6px';
      box.appendChild(cap);
      box.addEventListener('click', function () { S.frame = f.i; galSync(); });
      th.appendChild(box);
    });
  }

  var playTimer = null;
  function galPlay() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; $('galPlay').textContent = '播放'; return; }
    $('galPlay').textContent = '暂停';
    playTimer = setInterval(function () {
      var n = curScene().frames.length;
      S.frame = (S.frame + 1) % n;
      galSync();
    }, 800);
  }
  function galStep(d) {
    var n = curScene().frames.length;
    S.frame = (S.frame + d + n) % n;
    galSync();
  }

  // ---------- compare ----------
  function cmpModList() {
    return MODS.filter(function (m) { return m !== 'optical'; }).map(function (m) {
      return { id: m, label: MOD_LABEL[m] };
    });
  }
  var CMP = { mod: 'mwir' };
  function renderCompare() {
    var sc = curScene(), fr = curFrame();
    seg($('cmpScene'), sceneList(), String(S.scene), function (id) {
      S.scene = parseInt(id, 10); S.frame = 0; renderCompare(); galSync();
    });
    seg($('cmpMod'), cmpModList(), CMP.mod, function (m) { CMP.mod = m; renderCompare(); });

    var a = fr.mods.optical, b = fr.mods[CMP.mod];
    var A2 = $('cmpA'), B2 = $('cmpB');
    if (a) { A2.src = 'data/' + a.img; A2.style.display = 'block'; }
    if (b) {
      B2.src = 'data/' + b.img;
      B2.style.display = 'block';
      B2.style.opacity = String(parseInt($('cmpMix').value, 10) / 100);
      if (a) { B2.style.width = A2.clientWidth + 'px'; B2.style.height = A2.clientHeight + 'px'; }
    } else { B2.style.display = 'none'; }
    $('cmpBLabel').textContent = MOD_LABEL[CMP.mod];
    $('cmpTL').textContent = sc.title + ' · 帧 ' + (fr.i + 1) + '\n' + weatherText(fr);
    $('cmpCount').textContent = (fr.i + 1) + ' / ' + sc.frames.length;
    var showBoxes = $('cmpBoxes').checked;
    var boxes = showBoxes && a ? a.boxes : [];
    boxSvg($('cmpSvg'), boxes, a ? a.w : 640, a ? a.h : 480);
  }
  function cmpStep(d) {
    var n = curScene().frames.length;
    S.frame = (S.frame + d + n) % n;
    renderCompare(); galSync();
  }

  // ---------- point cloud ----------
  var PC = { yaw: -0.6, pitch: 0.35, zoom: 1, color: 'intensity', drag: null, spin: true };
  var pcFrameIdx = 0;

  function pcCloud() {
    var sc = curScene();
    var f = sc.frames[Math.min(pcFrameIdx, sc.frames.length - 1)];
    return f && f.cloud ? f.cloud : null;
  }
  function resizePC() {
    var c = $('pcCanvas'), host = $('pcViewer');
    if (!c || !host) return;
    var r = host.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = Math.max(2, Math.round(r.width * dpr));
    c.height = Math.max(2, Math.round(r.height * dpr));
    drawPC();
  }
  function cloudCenter(cl) {
    var n = cl.x.length, sx = 0, sy = 0, sz = 0;
    for (var i = 0; i < n; i++) { sx += cl.x[i]; sy += cl.y[i]; sz += cl.z[i]; }
    return [sx / n, sy / n, sz / n];
  }
  function drawPC() {
    var c = $('pcCanvas');
    if (!c) return;
    var g = c.getContext('2d');
    var W = c.width, H = c.height;
    g.fillStyle = '#05070a'; g.fillRect(0, 0, W, H);
    var cl = pcCloud();
    if (!cl || !cl.x.length) { $('pcTL').textContent = '无点云数据'; return; }

    var ctr = cloudCenter(cl);
    var cy = Math.cos(PC.yaw), sy = Math.sin(PC.yaw);
    var cp = Math.cos(PC.pitch), sp = Math.sin(PC.pitch);
    var zsc = Math.min(W, H) * 0.42 * PC.zoom;

    var n = cl.x.length;
    var px = new Float32Array(n), py = new Float32Array(n), pz = new Float32Array(n);
    for (var i = 0; i < n; i++) {
      var x = cl.x[i] / 100 - ctr[0] / 100;
      var y = cl.y[i] / 100 - ctr[1] / 100;
      var z = cl.z[i] / 100 - ctr[2] / 100;
      var x1 = x * cy - y * sy;
      var y1 = x * sy + y * cy;
      var y2 = y1 * cp - z * sp;
      var z2 = y1 * sp + z * cp;
      px[i] = W / 2 + x1 * zsc * 0.06;
      py[i] = H / 2 - z2 * zsc * 0.06;
      pz[i] = y2;
    }
    var order = new Array(n);
    for (i = 0; i < n; i++) order[i] = i;
    order.sort(function (a, b) { return pz[a] - pz[b]; });

    var OCOL = ['#37c2ff', '#ffb454', '#43d98a', '#ff5d6c', '#c792ea', '#8fbcbb'];
    var rad = Math.max(c.width / 900, 1.1);
    for (var k = 0; k < n; k++) {
      var j = order[k];
      if (PC.color === 'intensity') {
        var v = 0.25 + 0.75 * (cl.i[j] / 255);
        g.fillStyle = 'rgba(' + Math.round(90 * v + 30) + ',' +
          Math.round(200 * v + 20) + ',' + Math.round(255 * v) + ',0.9)';
      } else if (PC.color === 'object') {
        g.fillStyle = OCOL[((cl.o[j] % 6) + 6) % 6];
      } else {
        var w2 = Math.max(0, Math.min(1, cl.i[j] / 255));
        g.fillStyle = 'rgba(' + Math.round(255 * w2) + ',' + Math.round(140 + 90 * w2) + ',' +
          Math.round(60 * (1 - w2)) + ',0.9)';
      }
      g.beginPath();
      g.arc(px[j], py[j], rad, 0, 6.2832);
      g.fill();
    }

    $('pcTL').textContent = '点云 ' + cl.n + ' 点（绘制 ' + cl.shown + '）\n命中率 ' +
      f1((cl.hit || 0) * 100) + '% · 量程 ' + f0(cl.maxr || 0) + ' m';
    $('pcBR').textContent = '拖动旋转 · 滚轮缩放';
  }
  function pcSync() {
    var sc = curScene();
    seg($('pcScene'), sceneList(), String(S.scene), function (id) {
      S.scene = parseInt(id, 10); S.frame = 0; pcFrameIdx = 0; pcSync(); galSync();
    });
    seg($('pcColor'), [
      { id: 'intensity', label: '按反射强度' },
      { id: 'object', label: '按目标' },
      { id: 'range', label: '按距离' }
    ], PC.color, function (m) { PC.color = m; pcSync(); });

    var fr = sc.frames[Math.min(pcFrameIdx, sc.frames.length - 1)];
    $('pcRange').max = String(sc.frames.length - 1);
    $('pcRange').value = String(pcFrameIdx);
    $('pcCount').textContent = (pcFrameIdx + 1) + ' / ' + sc.frames.length;

    var li = $('pcInfo');
    li.innerHTML = '';
    var mods = (fr && fr.mods) || {};
    [['点数', fr && fr.cloud ? fr.cloud.n : '—'],
     ['命中率', fr && fr.cloud ? f1((fr.cloud.hit || 0) * 100) + '%' : '—'],
     ['量程', fr && fr.cloud ? f0(fr.cloud.maxr || 0) + ' m' : '—'],
     ['帧序', (pcFrameIdx + 1) + ' / ' + sc.frames.length]
    ].forEach(function (kv) {
      var d = el('div', 'kv');
      d.appendChild(el('span', '', kv[0]));
      d.appendChild(el('span', '', String(kv[1])));
      li.appendChild(d);
    });

    var rimg = mods.lidar_range;
    $('pcRangeImg').src = rimg ? 'data/' + rimg.img : '';
    resizePC();
  }
  (function bindPC() {
    var c = $('pcCanvas');
    if (!c) return;
    c.addEventListener('mousedown', function (e) { PC.drag = { x: e.clientX, y: e.clientY }; });
    window.addEventListener('mouseup', function () { PC.drag = null; });
    window.addEventListener('mousemove', function (e) {
      if (!PC.drag) return;
      PC.yaw += (e.clientX - PC.drag.x) * 0.008;
      PC.pitch = Math.max(-1.4, Math.min(1.4, PC.pitch + (e.clientY - PC.drag.y) * 0.008));
      PC.drag = { x: e.clientX, y: e.clientY };
      drawPC();
    });
    c.addEventListener('wheel', function (e) {
      e.preventDefault();
      PC.zoom = Math.max(0.25, Math.min(6, PC.zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
      drawPC();
    }, { passive: false });
    setInterval(function () {
      if (!PC.spin || PC.drag) return;
      var sec = $('lidar');
      if (!sec || !sec.classList.contains('on')) return;
      PC.yaw += 0.006;
      drawPC();
    }, 50);
  })();

  // ---------- radar ----------
  // A range/angle map is 95% noise floor: the noise plateau spans under 1 dB, so
  // any linear window either crushes the target or blows out the picture. The
  // kernel therefore ships the pixel of the strongest echo with each frame, and
  // this overlay circles it so the viewer does not have to hunt for one dot.
  function drawRdMark(fr) {
    var cv = $('rdOverlay');
    if (!cv) return;
    var img = $('rdImg');
    var pk = fr && fr.radar && fr.radar.peak;
    var box = cv.parentNode.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(1, Math.round(box.width * dpr));
    cv.height = Math.max(1, Math.round(box.height * dpr));
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, box.width, box.height);
    if (!pk || !$('rdMark').checked || !img.naturalWidth) return;

    // the <img> is letterboxed inside the viewer; map image px -> css px
    var s = Math.min(box.width / img.naturalWidth, box.height / img.naturalHeight);
    var dw = img.naturalWidth * s, dh = img.naturalHeight * s;
    var ox = (box.width - dw) / 2, oy = (box.height - dh) / 2;
    var x = ox + pk.px * s, y = oy + pk.py * s;
    ctx.strokeStyle = '#ff5d6c';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, 14, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - 22, y); ctx.lineTo(x - 6, y);
    ctx.moveTo(x + 6, y); ctx.lineTo(x + 22, y);
    ctx.moveTo(x, y - 22); ctx.lineTo(x, y - 6);
    ctx.moveTo(x, y + 6); ctx.lineTo(x, y + 22);
    ctx.stroke();
    ctx.fillStyle = '#ff5d6c';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('最强回波 ' + f1(pk.db) + ' dB', x + 18, y - 16);
  }

  function rdSync() {
    var sc = curScene(), fr = curFrame();
    seg($('rdScene'), sceneList(), String(S.scene), function (id) {
      S.scene = parseInt(id, 10); S.frame = 0; rdSync(); galSync();
    });

    var e = fr.mods.radar_ra || fr.mods.radar;
    $('rdImg').src = e ? 'data/' + e.img : '';
    $('rdRange').max = String(sc.frames.length - 1);
    $('rdRange').value = String(fr.i);
    $('rdCount').textContent = (fr.i + 1) + ' / ' + sc.frames.length;
    $('rdTL').textContent = sc.title + '\n' + sc.band + ' 频段 · ' + weatherText(fr);
    drawRdMark(fr);

    var r = fr.radar;
    var info = $('rdInfo');
    info.innerHTML = '';
    var snrs = r && r.pts ? r.pts.map(function (p) { return p[3]; }) : [];
    [['检出点', r ? r.n : '—'],
     ['杂波点', r ? r.clutter : '—'],
     ['最强 SNR', snrs.length ? f1(Math.max.apply(null, snrs)) + ' dB' : '—']
    ].forEach(function (kv) {
      var d = el('div', 'kv');
      d.appendChild(el('span', '', kv[0]));
      d.appendChild(el('span', '', String(kv[1])));
      info.appendChild(d);
    });

    var bg = $('rdBudget');
    bg.innerHTML = '';
    var bud = sc.budget;
    if (!bud) {
      bg.appendChild(el('div', 'muted', '该场景未导出链路预算'));
    } else {
      var rows = [
        ['目标 / 距离', bud.target + ' · ' + f0(bud.range_m) + ' m'],
        ['波段 / 天线增益', bud.band + ' ' + f1(bud.fc_ghz) + ' GHz · ' + f1(bud.gain_dbi) + ' dBi'],
        ['RCS / 门限', f1(bud.rcs_db) + ' dBsm · ' + f0(bud.det_snr_db) + ' dB'],
        ['发射 / 积累', f0(bud.tx_power_dbm) + ' dBm · ' + bud.n_chirps + ' chirps']
      ];
      bud.by_weather.forEach(function (w) {
        rows.push([WNAME[w.preset] + ' 最大距离', f0(w.max_range_m) + ' m']);
        rows.push([WNAME[w.preset] + ' 余量',
          (w.margin_db >= 0 ? '+' : '') + f1(w.margin_db) + ' dB']);
      });
      rows.forEach(function (kv) {
        var d = el('div', 'kv');
        d.appendChild(el('span', '', kv[0]));
        var v = el('span', '', kv[1]);
        if (kv[0].indexOf('余量') >= 0) {
          v.style.color = kv[1].charAt(0) === '-' ? 'var(--bad)' : 'var(--ok)';
        }
        d.appendChild(v);
        bg.appendChild(d);
      });
      var neg = bud.by_weather.filter(function (w) { return w.margin_db < 0; });
      if (neg.length) {
        bg.appendChild(el('div', 'note',
          '负余量意味着该天气下目标已超出作用距离，检出点少或为 0 是物理结果。'));
      }
    }

    var bx = $('rdBoxes');
    bx.innerHTML = '';
    (r && r.boxes ? r.boxes : []).forEach(function (b) {
      var d = el('div');
      d.style.marginBottom = '6px';
      d.textContent = b.name + ' [' + b.cls + ']  中心 ' +
        b.center.map(function (v) { return f1(v); }).join(', ') +
        '  尺寸 ' + b.size.map(function (v) { return f1(v); }).join('×') +
        '  朝向 ' + f0(b.yaw) + '°';
      bx.appendChild(d);
    });
    setTimeout(drawRadar, 20);
  }
  function drawRadar() {
    var cv = $('rdCanvas');
    if (!cv) return;
    var host = $('rdPtViewer');
    if (!host) return;
    var rc = host.getBoundingClientRect();
    if (rc.width < 10) return;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.max(2, Math.round(rc.width * dpr));
    cv.height = Math.max(2, Math.round(rc.height * dpr));
    var W = cv.width, H = cv.height;
    var g = cv.getContext('2d');
    g.fillStyle = '#05070a';
    g.fillRect(0, 0, W, H);

    var fr = curFrame();
    var rd = fr && fr.radar;
    var img = $('rdImg');
    if (img && img.naturalWidth) {
      $('rdBR').textContent = img.naturalWidth + ' × ' + img.naturalHeight + ' px';
    }
    if (!rd || !rd.pts || !rd.pts.length) {
      $('rdPtBR').textContent = '该帧无检出点';
      g.fillStyle = 'rgba(150,180,210,0.5)';
      g.font = Math.round(13 * dpr) + 'px monospace';
      g.fillText('无检出点 —— 目标超出作用距离或 MTI 门限', 16 * dpr, H / 2);
      return;
    }

    // 3D point cloud gives the geometry; a plan view is what reads best for radar.
    var ctr = [0, 0, 0], nn = rd.pts.length;
    rd.pts.forEach(function (p) {
      var az = p[1] * Math.PI / 180;
      ctr[0] += p[0] * Math.sin(az); ctr[1] += p[0] * Math.cos(az); ctr[2] += p[2];
    });
    ctr = [ctr[0] / nn, ctr[1] / nn, ctr[2] / nn];

    var span = 1;
    rd.pts.forEach(function (p) {
      var az = p[1] * Math.PI / 180;
      var dx = p[0] * Math.sin(az) - ctr[0], dy = p[0] * Math.cos(az) - ctr[1];
      span = Math.max(span, Math.abs(dx), Math.abs(dy));
    });
    var scale = (Math.min(W, H) * 0.40) / span;
    var cx = W / 2, cy = H / 2;

    // grid
    g.strokeStyle = 'rgba(70,100,130,0.30)';
    g.lineWidth = 1;
    g.font = Math.round(9 * dpr) + 'px monospace';
    g.fillStyle = 'rgba(150,180,210,0.5)';
    [0.25, 0.5, 0.75, 1.0].forEach(function (f) {
      g.beginPath();
      g.arc(cx, cy, span * f * scale, 0, 6.2832);
      g.stroke();
    });
    g.beginPath();
    g.moveTo(cx - span * scale, cy); g.lineTo(cx + span * scale, cy);
    g.moveTo(cx, cy - span * scale); g.lineTo(cx, cy + span * scale);
    g.stroke();
    g.fillText(f0(span) + ' m', cx + span * scale * 0.5 + 4 * dpr, cy - 4 * dpr);
    g.fillText('+X', cx + span * scale + 2 * dpr, cy + 10 * dpr);
    g.fillText('+Y', cx + 4 * dpr, cy - span * scale - 4 * dpr);

    var snrMin = Infinity, snrMax = -Infinity;
    rd.pts.forEach(function (p) {
      if (p[3] < snrMin) snrMin = p[3];
      if (p[3] > snrMax) snrMax = p[3];
    });
    var rngS = Math.max(snrMax - snrMin, 1);
    rd.pts.forEach(function (p) {
      var az = p[1] * Math.PI / 180;
      var x = cx + (p[0] * Math.sin(az) - ctr[0]) * scale;
      var y = cy - (p[0] * Math.cos(az) - ctr[1]) * scale;
      var t = (p[3] - snrMin) / rngS;
      g.fillStyle = 'rgba(' + Math.round(90 + 165 * t) + ',' +
        Math.round(215 - 115 * t) + ',255,0.95)';
      g.beginPath();
      g.arc(x, y, (1.8 + 3.2 * t) * dpr, 0, 6.2832);
      g.fill();
    });

    // sensor at the origin of the plan view
    var ox = cx + (0 - ctr[0]) * scale, oy = cy - (0 - ctr[1]) * scale;
    g.fillStyle = '#ff5d6c';
    g.beginPath();
    g.moveTo(ox, oy - 8 * dpr); g.lineTo(ox - 5 * dpr, oy + 3 * dpr);
    g.lineTo(ox + 5 * dpr, oy + 3 * dpr); g.closePath(); g.fill();

    $('rdPtBR').textContent = '俯视 · ' + nn + ' 点 · SNR ' + f1(snrMin) + '–' +
      f1(snrMax) + ' dB · 跨度 ' + f0(span) + ' m';
    $('rdPtTL').textContent = (fr && fr.weather ? (WNAME[fr.weather.preset] || '') : '') +
      '  检出 ' + rd.n + ' 点' + (rd.clutter ? ' / 杂波 ' + rd.clutter : '');
  }

  // ---------- explore ----------
  // Band constants come from web/data/radar_bands.json, which tools/export_web.py
  // writes straight out of simkit/radar.py -- so the browser cannot drift from
  // the kernel. Nothing here is a hand-copied number.
  var BANDS = null;
  var PGR_DEFAULT = 15.0;

  function radarRange(band, rcsDbsm, rain, ptxDbm, nch) {
    var b = BANDS[band];
    var att = b.gas_db_km + b.rain_k * Math.pow(Math.max(rain, 0), b.rain_a);
    var ptx = Math.pow(10, (ptxDbm - 30) / 10);
    var g = Math.pow(10, b.gain_db / 10);
    var sig = Math.pow(10, rcsDbsm / 10);
    var pn = Math.pow(10, b.pn_dbw / 10);
    var loss = Math.pow(10, b.loss_db / 10);
    var snr0 = 10 * Math.log10(ptx * g * g * b.lam_m * b.lam_m * sig /
      (Math.pow(4 * Math.PI, 3) * loss * pn)) + 10 * Math.log10(Math.max(nch, 1));
    var target = snr0 - b.det_snr_db;
    // r^4 * 10^(2*att*r/1000 / 10) = 10^(target/10); monotone in r -> bisect.
    var lo = 1, hi = 500000;
    var f = function (r) { return 10 * Math.log10(Math.pow(r, 4)) + 2 * att * r / 1000; };
    if (f(lo) >= target) return 0;
    if (f(hi) < target) return hi;
    for (var i = 0; i < 60; i++) {
      var mid = 0.5 * (lo + hi);
      if (f(mid) < target) lo = mid; else hi = mid;
    }
    return lo;
  }
  function exploreUpdate() {
    var rcs = parseFloat($('exRcs').value);
    var rain = parseFloat($('exRain').value);
    var ptx = parseFloat($('exPtx').value);
    var nch = parseFloat($('exNch').value);
    $('exRcsV').textContent = f1(rcs) + ' dBsm';
    $('exRainV').textContent = f0(rain) + ' mm/h';
    $('exPtxV').textContent = f0(ptx) + ' dBm';
    $('exNchV').textContent = f0(nch);

    var names = Object.keys(BANDS).sort(function (a, b) {
      return BANDS[a].fc_ghz - BANDS[b].fc_ghz;
    });
    var res = {};
    names.forEach(function (b) { res[b] = radarRange(b, rcs, rain, ptx, nch); });

    $('exRadarNote').textContent = '雨衰减按 ITU-R P.838-3 的 γ = k·R^α 计算，' +
      '两次穿过雨区累计为 2·γ·r。阈值取 SNR = 11 dB。X 波段雨衰小、天线增益低；' +
      'W 波段天线小、雨衰大——这就是远距离必然选 X/Ka 的原因。';

    chart($('exRadarCanvas'), names, [{
      label: 'RCS ' + f1(rcs) + ' dBsm · 雨 ' + f0(rain) + ' mm/h',
      vals: names.map(function (b) { return res[b]; }),
      color: '#37c2ff'
    }]);

    var vis = parseFloat($('exVis').value);
    var rain2 = parseFloat($('exRain2').value);
    $('exVisV').textContent = f1(vis) + ' km';
    $('exRain2V').textContent = f0(rain2) + ' mm/h';
    var dists = [0.1, 0.3, 1, 2, 3, 5, 10, 20];
    var series = [
      { label: '905 nm 激光（双程）', color: '#43d98a',
        vals: dists.map(function (d) {
          return 100 * Math.exp(-2 * (0.8 * 3.912 / Math.max(vis, 0.05) +
            0.072 * Math.pow(rain2, 0.72)) * d);
        }) },
      { label: '可见光（单程）', color: '#37c2ff',
        vals: dists.map(function (d) { return 100 * Math.exp(-(3.912 / Math.max(vis, 0.05)) * d); }) },
      { label: '红外 8–12 µm（单程）', color: '#ffb454',
        vals: dists.map(function (d) {
          return 100 * Math.exp(-((0.06 * Math.pow(20 / Math.max(vis, 0.05), 0.3) +
            0.02 * Math.pow(rain2, 0.8))) * d);
        }) }
    ];
    $('exOptNote').textContent = '纵轴是透过率（%）。激光是双程衰减，' +
      '能见度差时掉得最快——这正是激光雷达在雨雾天最先失效的原因；' +
      '长波红外最抗雾，因为散射跟 λ 成反比。';
    chart($('exOptCanvas'), dists.map(function (d) { return d + 'km'; }), series, { pct: true });

    var t = $('exTable');
    var rows = ['<tr><th>波段</th><th>频率</th><th>天线增益</th><th>雨衰 dB/km</th>' +
      '<th class="n">最大距离</th></tr>'];
    names.forEach(function (b) {
      var bb = BANDS[b];
      var att = bb.gas_db_km + bb.rain_k * Math.pow(Math.max(rain, 0), bb.rain_a);
      rows.push('<tr><td>' + b + '</td><td class="n">' + f1(bb.fc_ghz) + ' GHz</td>' +
        '<td class="n">' + f1(bb.gain_db) + ' dBi</td>' +
        '<td class="n">' + f2(att) + '</td><td class="n">' +
        (res[b] >= 1000 ? f1(res[b] / 1000) + ' km' : f0(res[b]) + ' m') + '</td></tr>');
    });
    t.innerHTML = rows.join('');
  }

  function chart(cv, labels, series, opt) {
    if (!cv) return;
    opt = opt || {};
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var rc = cv.getBoundingClientRect();
    if (rc.width < 10) return;
    cv.width = Math.max(2, Math.round(rc.width * dpr));
    cv.height = Math.max(2, Math.round(rc.height * dpr));
    var g = cv.getContext('2d');
    var W = cv.width, H = cv.height;
    g.clearRect(0, 0, W, H);
    var padL = 58 * dpr, padR = 14 * dpr, padT = 16 * dpr, padB = 34 * dpr;
    var pw = W - padL - padR, ph = H - padT - padB;

    var vmax = 0;
    series.forEach(function (s) { s.vals.forEach(function (v) { if (v > vmax) vmax = v; }); });
    if (!vmax) vmax = 1;

    g.strokeStyle = 'rgba(80,110,140,0.3)';
    g.lineWidth = 1;
    g.fillStyle = 'rgba(150,180,210,0.7)';
    g.font = (10 * dpr) + 'px monospace';
    var i;
    for (i = 0; i <= 4; i++) {
      var y = padT + ph * i / 4;
      g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + pw, y); g.stroke();
      var val = vmax * (1 - i / 4);
      g.fillText(opt.pct ? f0(val) + '%'
        : (val >= 1000 ? f1(val / 1000) + 'k' : f0(val)), 6 * dpr, y + 3 * dpr);
    }
    var n = labels.length;
    for (var j = 0; j < n; j++) {
      var x = padL + pw * (n === 1 ? 0.5 : j / (n - 1));
      g.fillText(labels[j], x - 8 * dpr, H - 12 * dpr);
    }
    series.forEach(function (s) {
      g.strokeStyle = s.color;
      g.lineWidth = 2 * dpr;
      g.beginPath();
      s.vals.forEach(function (v, k) {
        var xx = padL + pw * (n === 1 ? 0.5 : k / (n - 1));
        var yy = padT + ph * (1 - v / vmax);
        if (k === 0) g.moveTo(xx, yy); else g.lineTo(xx, yy);
      });
      g.stroke();
      s.vals.forEach(function (v, k) {
        var xx = padL + pw * (n === 1 ? 0.5 : k / (n - 1));
        var yy = padT + ph * (1 - v / vmax);
        g.fillStyle = s.color;
        g.beginPath(); g.arc(xx, yy, 2.6 * dpr, 0, 6.2832); g.fill();
      });
    });
    var lx = padL + 6 * dpr, ly = padT + 10 * dpr;
    series.forEach(function (s) {
      g.fillStyle = s.color;
      g.fillRect(lx, ly - 5 * dpr, 10 * dpr, 3 * dpr);
      g.fillStyle = 'rgba(200,220,240,0.85)';
      g.fillText(s.label, lx + 14 * dpr, ly);
      ly += 14 * dpr;
    });
  }

  // ---------- boot ----------
  function boot() {
    var have = {};
    MAN.scenes.forEach(function (s) {
      s.frames.forEach(function (f) {
        Object.keys(f.mods || {}).forEach(function (k) { have[k] = 1; });
      });
    });
    MODS = MODS_ALL.filter(function (m) { return have[m]; });
    if (MODS.indexOf(S.mod) < 0) S.mod = MODS[0];

    $('genInfo').textContent = MAN.scenes.length + ' 场景 · ' +
      MAN.scenes.reduce(function (a, s) { return a + s.frames.length; }, 0) + ' 帧 · ' +
      MODS.length + ' 种模态';
    galSync(); pcSync(); rdSync(); renderCompare();
    if (BANDS) exploreUpdate();
  }

  document.addEventListener('DOMContentLoaded', function () {
    $('galPrev').addEventListener('click', function () { galStep(-1); });
    $('galNext').addEventListener('click', function () { galStep(1); });
    $('galPlay').addEventListener('click', galPlay);
    $('galRange').addEventListener('input', function () {
      S.frame = parseInt(this.value, 10); galSync();
    });
    $('cmpMix').addEventListener('input', function () {
      $('cmpB').style.opacity = String(parseInt(this.value, 10) / 100);
    });
    $('cmpBoxes').addEventListener('change', renderCompare);
    $('cmpPrev').addEventListener('click', function () { cmpStep(-1); });
    $('cmpNext').addEventListener('click', function () { cmpStep(1); });
    $('pcPrev').addEventListener('click', function () {
      var n = curScene().frames.length;
      pcFrameIdx = (pcFrameIdx - 1 + n) % n; pcSync();
    });
    $('pcNext').addEventListener('click', function () {
      var n = curScene().frames.length;
      pcFrameIdx = (pcFrameIdx + 1) % n; pcSync();
    });
    $('pcRange').addEventListener('input', function () {
      pcFrameIdx = parseInt(this.value, 10); pcSync();
    });
    $('pcReset').addEventListener('click', function () {
      PC.yaw = -0.6; PC.pitch = 0.35; PC.zoom = 1; drawPC();
    });
    $('pcSpin').addEventListener('change', function () { PC.spin = this.checked; });
    $('rdPrev').addEventListener('click', function () {
      var n = curScene().frames.length;
      S.frame = (S.frame - 1 + n) % n; rdSync(); galSync();
    });
    $('rdNext').addEventListener('click', function () {
      var n = curScene().frames.length;
      S.frame = (S.frame + 1) % n; rdSync(); galSync();
    });
    $('rdRange').addEventListener('input', function () {
      S.frame = parseInt(this.value, 10); rdSync(); galSync();
    });
    ['exRcs', 'exRain', 'exPtx', 'exNch', 'exVis', 'exRain2'].forEach(function (id) {
      $(id).addEventListener('input', exploreUpdate);
    });
    $('rdMark').addEventListener('change', function () { drawRdMark(curFrame()); });
    window.addEventListener('resize', function () {
      resizePC(); drawRadar(); drawRdMark(curFrame()); exploreUpdate();
    });
    $('rdImg').addEventListener('load', function () { drawRdMark(curFrame()); });

    // data/boot.js is a pre-baked copy of the same two JSON files. It is what
    // makes the page work from file:// (no server, no fetch), which is the
    // air-gapped case this project actually targets. When it is present we use
    // it; otherwise we fall back to fetching the JSON, which is what a normal
    // static host serves.
    function fromBoot() {
      var d = window.SIMKIT_DATA;
      if (!d || !d.manifest || !d.bands) return null;
      return Promise.resolve([d.manifest, d.bands]);
    }
    var loadData = fromBoot() || Promise.all([
      fetch('data/manifest.json', { cache: 'no-cache' })
        .then(function (r) { if (!r.ok) throw new Error('manifest ' + r.status); return r.json(); }),
      fetch('data/radar_bands.json', { cache: 'no-cache' })
        .then(function (r) { if (!r.ok) throw new Error('bands ' + r.status); return r.json(); })
    ]);
    loadData.then(function (res) { MAN = res[0]; BANDS = res[1]; boot(); exploreUpdate(); })
      .catch(function (e) {
        document.body.insertAdjacentHTML('beforeend',
          '<div style="position:fixed;bottom:16px;left:16px;background:#ff5d6c;color:#000;' +
          'padding:10px 16px;border-radius:8px;font:13px monospace;z-index:99">' +
          '数据未加载：请先运行 tools/export_web.py 生成 data/boot.js<br>' +
          String(e) + '</div>');
      });
  });
})();