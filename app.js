/* simkit web app -- no build step, no dependencies, works offline.
 *
 * Layout of this file:
 *   1  helpers / modality tables
 *   2  tabs
 *   3  shared drawing (boxes, locator, zoom, charts)
 *   4  gallery / compare
 *   5  point cloud
 *   6  radar
 *   7  SAR
 *   8  seeker (dual field of view)
 *   9  explore (live physics)
 *  10  boot
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  // 目标类别译名：manifest 里的 cls 是引擎的英文标识
  var CLS_LABEL = {
    fighter: '战斗机', uav: '无人机', uav_fixed: '固定翼无人机',
    uav_rotor: '旋翼无人机', helicopter: '直升机', airliner: '民航飞机',
    vehicle: '地面车辆', tank: '坦克', ship: '舰船', boat: '船只',
    artillery: '自行火炮', launcher: '发射车', person: '人员',
    building: '建筑', scenery: '场景物'
  };

  var el = function (tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt !== undefined) e.textContent = txt;
    return e;
  };
  var NS = 'http://www.w3.org/2000/svg';
  var f1 = function (v) { return (Math.round(v * 10) / 10).toFixed(1); };
  var f2 = function (v) { return (Math.round(v * 100) / 100).toFixed(2); };
  var f0 = function (v) { return String(Math.round(v)); };
  var isNum = function (v) { return typeof v === 'number' && isFinite(v); };
  var num = function (v) { return isNum(v) ? v : null; };
  function fmtM(v) {
    if (!isNum(v)) return '—';
    return Math.abs(v) >= 1000 ? f2(v / 1000) + ' km' : f0(v) + ' m';
  }
  function fmtAxis(v) {
    if (!isNum(v)) return '';
    var a = Math.abs(v);
    if (a >= 1000) return f1(v / 1000) + 'k';
    if (a >= 10) return f0(v);
    return f1(v);
  }
  function svgEl(tag, attrs) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) {
      e.setAttribute(k, attrs[k]);
    }
    return e;
  }

  // ---------- modalities ----------
  var MOD_LABEL = {
    optical: '可见光', mwir: '中波红外', lwir: '长波红外',
    optical_wide: '可见光 · 宽视场', mwir_wide: '中波红外 · 宽视场',
    lwir_wide: '长波红外 · 宽视场',
    radar_ra: '雷达距离-角度', radar: '雷达距离-多普勒',
    sar: 'SAR 合成孔径',
    lidar_range: '激光距离', lidar_intensity: '激光强度'
  };
  var MODS_ALL = [
    'optical', 'optical_wide', 'mwir', 'mwir_wide', 'lwir', 'lwir_wide',
    'radar_ra', 'sar', 'lidar_range', 'lidar_intensity'
  ];
  // Only offer modalities that are actually in the manifest: the range-Doppler
  // image is written by the kernel but is a 1600x64 strip with a 2 px target, so
  // export_web.py leaves it out of the site on purpose. An empty tab is worse
  // than no tab. Same rule covers sar / *_wide when a dataset has none.
  var MODS = MODS_ALL.slice();
  var WNAME = { clear: '晴', cloudy: '阴', rain: '雨', fog: '雾', storm: '暴风雨' };

  var MAN = null;
  var S = { scene: 0, frame: 0, mod: 'optical' };

  // Dev-only override: ?manifest=_tmp/webmock/manifest.json
  var Q = {};
  (function () {
    var q = location.search.replace(/^\?/, '');
    if (!q) return;
    q.split('&').forEach(function (kv) {
      var i = kv.indexOf('=');
      var k = i < 0 ? kv : kv.slice(0, i);
      var v = i < 0 ? '' : kv.slice(i + 1);
      try { Q[decodeURIComponent(k)] = decodeURIComponent(v); } catch (e) { Q[k] = v; }
    });
  })();
  var MAN_URL = Q.manifest || '';
  var FELL_BACK = false;   // ?manifest= died and we used data/boot.js instead

  // Footer label for manifest.generated. Today the exporter writes a bare tag
  // ("simkit"); if it ever writes a date we say so instead of printing a number
  // nobody can interpret. No value is hard coded here.
  function genLabel(g) {
    if (!g) return '';
    var s = String(g);
    if (/^\d{4}-\d{2}-\d{2}/.test(s)) {
      return '生成于 ' + s.replace('T', ' ').replace(/\.[0-9]+.*$/, '');
    }
    return '数据来源 ' + s;
  }

  // One quiet line of text, for conditions the user should be told about but
  // which should not look like an error (a dev-only URL parameter going stale).
  function notice(txt) {
    var el = document.getElementById('loadNotice');
    if (!el) {
      el = document.createElement('div');
      el.id = 'loadNotice';
      el.className = 'notice';
      document.body.appendChild(el);
    }
    el.textContent = txt;
  }

  function imgURL(e) {
    if (!e || !e.img) return '';
    return ((MAN && MAN.base) || 'data/') + e.img;
  }
  // Filmstrip copy: half the pixels, a fifth of the bytes. Built by
  // tools/opt_images.py from the same source frame, so a thumb can never show
  // a different frame than the one it opens.
  function thumbURL(e) {
    if (!e || !e.img) return '';
    var base = (MAN && MAN.base) || 'data/';
    if (base !== 'data/') return imgURL(e);
    return 'data/_t/' + e.img.replace(/\.(webp|png|jpg)$/i, '') + '.webp';
  }

  // Label for a modality key. The curated table keeps the tab bar compact for the
  // keys we ship; anything the exporter invents later (a new band, a new view) is
  // labelled from scene.mods[].title rather than showing a bare key.
  var MOD_TITLE = {};
  function modLabel(m) {
    return MOD_LABEL[m] || MOD_TITLE[m] || m;
  }

  // Per-scene modality list. scene.mods is the exporter's own list
  // ([{key,title,boxes}]) and carries the Chinese title, so the button text is
  // not duplicated here. A key is only offered when at least one frame really
  // has pixels for it -- the "no empty button" rule the site always had, now
  // applied per scene and per whatever the exporter discovered on disk.
  function sceneMods(sc) {
    var out = [], seen = {};
    (sc.mods || []).forEach(function (m) {
      if (!m || !m.key || seen[m.key]) return;
      var has = (sc.frames || []).some(function (f) { return f.mods && f.mods[m.key]; });
      if (!has) return;
      seen[m.key] = 1;
      out.push({ id: m.key, label: m.title || modLabel(m.key) });
    });
    (sc.frames || []).forEach(function (f) {
      Object.keys(f.mods || {}).forEach(function (k) {
        if (seen[k]) return;
        seen[k] = 1;
        out.push({ id: k, label: modLabel(k) });
      });
    });
    return out;
  }

  // ---------- tabs + routing ----------
  // Every tab is a real URL (#/sar), so a link can point at the view that
  // proves the point, and the back button works. pushState rather than
  // location.hash= so a click does not add two history entries.
  var TABS = ['home', 'gallery', 'compare', 'sar', 'seeker', 'lidar', 'radar',
    'figures', 'explore', 'pipeline', 'about'];
  var CURRENT = 'home';
  function hashTab() {
    var m = /^#\/?([a-z0-9_]+)/i.exec(location.hash || '');
    var t = m ? m[1] : 'home';
    return TABS.indexOf(t) < 0 ? 'home' : t;
  }
  function showTab(name, opts) {
    opts = opts || {};
    if (TABS.indexOf(name) < 0) name = 'home';
    CURRENT = name;
    TABS.forEach(function (t) {
      var sec = $(t);
      if (sec) sec.classList.toggle('on', t === name);
    });
    Array.prototype.forEach.call(document.querySelectorAll('#tabs button'), function (b) {
      var on = b.getAttribute('data-tab') === name;
      b.classList.toggle('on', on);
      if (on) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
    });
    if (opts.syncHash !== false && location.hash !== '#/' + name) {
      try { history.pushState({ tab: name }, '', '#/' + name); } catch (e) { location.hash = '#/' + name; }
    }
    if (!opts.keepScroll) window.scrollTo({ top: 0, behavior: opts.instant ? 'auto' : 'smooth' });
    if (!MAN) return;
    if (name !== 'seeker') skStop();
    if (name !== 'sar') sarStop();
    if (name !== 'gallery' && playTimer) galPlay();
    if (name === 'home') setTimeout(homeSync, 20);
    if (name === 'gallery') setTimeout(galSync, 20);
    if (name === 'lidar') setTimeout(function () { pcSync(); lidarPick(); resizePC(); }, 30);
    if (name === 'radar') setTimeout(function () { rdSync(); drawRadar(); drawRdMark(curFrame()); }, 30);
    if (name === 'explore') setTimeout(function () { if (BANDS) exploreUpdate(); }, 40);
    if (name === 'compare') setTimeout(renderCompare, 30);
    if (name === 'sar') setTimeout(sarSync, 30);
    if (name === 'seeker') setTimeout(skSync, 30);
  }
  Array.prototype.forEach.call(document.querySelectorAll('#tabs button'), function (b) {
    b.addEventListener('click', function () { showTab(b.getAttribute('data-tab')); });
  });
  window.addEventListener('popstate', function () {
    showTab(hashTab(), { syncHash: false, instant: true });
  });

  // ---------- shared helpers ----------
  function seg(host, items, cur, onPick) {
    if (!host) return;
    host.innerHTML = '';
    items.forEach(function (it) {
      var b = el('button', it.id === cur ? 'on' : '', it.label);
      b.addEventListener('click', function () { onPick(it.id); });
      host.appendChild(b);
    });
  }
  function kvInto(host, rows) {
    if (!host) return;
    host.innerHTML = '';
    rows.forEach(function (r) {
      var d = el('div', 'kv');
      d.appendChild(el('span', '', r[0]));
      var v = el('span', '', (r[1] === null || r[1] === undefined || r[1] === '') ? '—' : String(r[1]));
      if (r[2]) v.style.color = r[2];
      d.appendChild(v);
      host.appendChild(d);
    });
  }
  function sceneList() {
    return MAN.scenes.map(function (s, i) { return { id: String(i), label: s.title }; });
  }
  function curScene() { return MAN.scenes[Math.min(S.scene, MAN.scenes.length - 1)]; }
  function curFrame() {
    var sc = curScene();
    if (!sc || !sc.frames.length) return null;
    return sc.frames[Math.max(0, Math.min(S.frame, sc.frames.length - 1))];
  }
  function weatherText(fr) {
    var w = (fr && fr.weather) || {};
    var nm = WNAME[w.name] || WNAME[w.preset] || w.name || '—';
    return nm + ' · 云量 ' + f0((w.cloud_cover || 0) * 100) + '% · 雨 ' +
      f1(w.rain_mmh || 0) + ' mm/h · 能见度 ' + f1((w.visibility_m || 0) / 1000) + ' km';
  }
  function aspectStyle(host, w, h) {
    if (!host || !w || !h) return;
    var ar = w / h;
    if (ar > 3.2) ar = 3.2; else if (ar < 0.55) ar = 0.55;
    host.style.aspectRatio = String(ar);
  }
  // A dataset can carry SAR in one scene only. Rather than show an empty plot,
  // jump to the first scene that actually has the data (the user can switch back
  // with the scene pills). Returns true when it moved.
  function ensureScene(pred) {
    if (!MAN || pred(curScene())) return false;
    for (var i = 0; i < MAN.scenes.length; i++) {
      if (pred(MAN.scenes[i])) { S.scene = i; S.frame = 0; return true; }
    }
    return false;
  }

  // Draw boxes in image-pixel coordinates. The overlay <svg> carries the image's
  // viewBox with preserveAspectRatio=meet, so it stays glued to a letterboxed
  // <img> without any code measuring the rendered rect.
  //
  // opt.locator adds corner brackets + faint leader lines around the first box:
  // a 3 px target is otherwise unfindable even when the box is drawn correctly.
  function drawBoxes(svg, boxes, w, h, opt) {
    if (!svg) return;
    opt = opt || {};
    svg.innerHTML = '';
    if (!w || !h) return;
    svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    if (!boxes || !boxes.length) return;

    var col = opt.color || '#37c2ff';
    var sw = Math.max(w / 320, 1.2);
    var fs = Math.max(w / 52, 9);

    boxes.forEach(function (b) {
      var bw = Math.max(b.x1 - b.x0, 1), bh = Math.max(b.y1 - b.y0, 1);
      svg.appendChild(svgEl('rect', {
        x: b.x0, y: b.y0, width: bw, height: bh, fill: 'none',
        stroke: col, 'stroke-width': sw, 'vector-effect': 'non-scaling-stroke'
      }));
      if (b.label) {
        svg.appendChild(svgEl('text', {
          x: b.x0 + 2, y: Math.max(b.y0 - 4, fs + 1), fill: col,
          'font-size': fs, 'font-family': 'ui-monospace, monospace'
        })).textContent = b.label;
      }
    });

    if (opt.locator) {
      var b0 = boxes[0];
      var cx = (b0.x0 + b0.x1) / 2, cy = (b0.y0 + b0.y1) / 2;
      var L = Math.max(w, h) * 0.115, hh = L / 2, arm = L * 0.3;
      var lc = opt.locColor || '#ffffff';
      [[0, cy, cx - hh, cy], [cx + hh, cy, w, cy],
       [cx, 0, cx, cy - hh], [cx, cy + hh, cx, h]].forEach(function (s) {
        svg.appendChild(svgEl('line', {
          x1: s[0], y1: s[1], x2: s[2], y2: s[3], stroke: lc,
          'stroke-width': Math.max(sw * 0.6, 0.8), 'stroke-opacity': 0.22,
          'stroke-dasharray': '5 7', 'vector-effect': 'non-scaling-stroke'
        }));
      });
      [[cx - hh, cy - hh + arm, cx - hh, cy - hh, cx - hh + arm, cy - hh],
       [cx + hh - arm, cy - hh, cx + hh, cy - hh, cx + hh, cy - hh + arm],
       [cx - hh, cy + hh - arm, cx - hh, cy + hh, cx - hh + arm, cy + hh],
       [cx + hh - arm, cy + hh, cx + hh, cy + hh, cx + hh, cy + hh - arm]
      ].forEach(function (p) {
        svg.appendChild(svgEl('path', {
          d: 'M' + p[0] + ' ' + p[1] + 'L' + p[2] + ' ' + p[3] + 'L' + p[4] + ' ' + p[5],
          fill: 'none', stroke: lc, 'stroke-width': Math.max(sw, 1.1),
          'stroke-opacity': 0.9, 'stroke-linecap': 'square', 'vector-effect': 'non-scaling-stroke'
        }));
      });
      if (opt.locLabel) {
        var t = svgEl('text', {
          x: cx - hh, y: Math.max(cy - hh - 5, fs), fill: lc, 'fill-opacity': 0.9,
          'font-size': fs, 'font-family': 'ui-monospace, monospace'
        });
        t.textContent = opt.locLabel;
        svg.appendChild(t);
      }
    }
  }

  // ---------- zoom (click to magnify, wheel, drag) ----------
  function makeZoom(host, stage, opts) {
    opts = opts || {};
    var Z = { s: 1, x: 0, y: 0, down: null, moved: false };

    function apply() {
      if (Z.s <= 1.001) { Z.s = 1; Z.x = 0; Z.y = 0; }
      stage.style.transform = 'translate(' + Z.x.toFixed(2) + 'px,' + Z.y.toFixed(2) +
        'px) scale(' + Z.s.toFixed(3) + ')';
      host.classList.toggle('zoomed', Z.s > 1.001);
      if (opts.onChange) opts.onChange(Z);
    }
    function setOrigin() {
      var o = (opts.origin && opts.origin()) || [0.5, 0.5];
      var px = Math.max(0, Math.min(1, o[0])) * 100;
      var py = Math.max(0, Math.min(1, o[1])) * 100;
      stage.style.transformOrigin = px.toFixed(2) + '% ' + py.toFixed(2) + '%';
    }
    function clampPan() {
      var r = host.getBoundingClientRect();
      var mx = Math.max(0, r.width * (Z.s - 1) * 0.5);
      var my = Math.max(0, r.height * (Z.s - 1) * 0.5);
      Z.x = Math.max(-mx, Math.min(mx, Z.x));
      Z.y = Math.max(-my, Math.min(my, Z.y));
    }

    Z.reset = function () { Z.s = 1; Z.x = 0; Z.y = 0; apply(); };
    Z.apply = apply;

    host.addEventListener('dblclick', function () { Z.reset(); });
    host.addEventListener('click', function () {
      if (Z.moved) { Z.moved = false; return; }
      if (Z.s > 1.001) { Z.reset(); return; }
      setOrigin();
      Z.s = opts.clickScale || 4;
      clampPan();
      apply();
    });
    host.addEventListener('wheel', function (e) {
      if (Z.s <= 1.001) return;   // at fit-to-width the page keeps the wheel
      e.preventDefault();
      Z.s = Math.max(1, Math.min(opts.max || 24, Z.s * (e.deltaY > 0 ? 0.86 : 1.16)));
      clampPan();
      apply();
    }, { passive: false });
    host.addEventListener('mousedown', function (e) {
      if (Z.s <= 1.001) return;
      Z.down = { x: e.clientX, y: e.clientY, bx: Z.x, by: Z.y };
      host.classList.add('dragging');
      e.preventDefault();
    });
    window.addEventListener('mousemove', function (e) {
      if (!Z.down) return;
      var dx = e.clientX - Z.down.x, dy = e.clientY - Z.down.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) Z.moved = true;
      Z.x = Z.down.bx + dx;
      Z.y = Z.down.by + dy;
      clampPan();
      apply();
    });
    window.addEventListener('mouseup', function () {
      if (!Z.down) return;
      Z.down = null;
      host.classList.remove('dragging');
    });
    return Z;
  }

  // ---------- gallery ----------
  function galSync() {
    var sc = curScene(), fr = curFrame();
    if (!sc || !fr) return;
    seg($('galScene'), sceneList(), String(S.scene), function (id) {
      S.scene = parseInt(id, 10); S.frame = 0; galSync();
    });
    // Buttons come from this scene's own modality list, so a scene that has no
    // SAR (or no wide view) simply does not offer the button. No fixed state
    // table: adding a band later only changes the exporter.
    var list = sceneMods(sc);
    var ok = false;
    list.forEach(function (it) { if (it.id === S.mod) ok = true; });
    if (!ok) S.mod = list.length ? list[0].id : '';
    seg($('galMod'), list, S.mod, function (m) { S.mod = m; galSync(); });

    var entry = fr.mods ? fr.mods[S.mod] : null;
    var img = $('galImg');
    if (entry) {
      aspectStyle($('galViewer'), entry.w, entry.h);
      img.style.display = 'block';
      img.src = imgURL(entry);
      var small = entry.boxes && entry.boxes.length &&
        Math.max(entry.boxes[0].x1 - entry.boxes[0].x0,
                 entry.boxes[0].y1 - entry.boxes[0].y0) < Math.min(entry.w, entry.h) * 0.05;
      img.onload = function () {
        drawBoxes($('galSvg'), entry.boxes, img.naturalWidth, img.naturalHeight,
          { locator: small, locLabel: small ? '目标' : '' });
      };
      drawBoxes($('galSvg'), entry.boxes, entry.w, entry.h,
        { locator: small, locLabel: small ? '目标' : '' });
    } else {
      img.style.display = 'none';
      $('galSvg').innerHTML = '';
    }
    $('galTL').textContent = sc.title + ' · 帧 ' + (fr.i + 1) + '/' + sc.frames.length + '\n' +
      modLabel(S.mod);
    $('galBR').textContent = entry ? (entry.w + '×' + entry.h) : '无数据';
    $('galRange').max = String(sc.frames.length - 1);
    $('galRange').value = String(fr.i);
    $('galCount').textContent = (fr.i + 1) + ' / ' + sc.frames.length;

    kvInto($('galInfo'), [
      ['天气', weatherText(fr)],
      ['标注框', entry && entry.boxes ? entry.boxes.length + ' 个' : '—'],
      ['雷达检出', fr.radar ? fr.radar.n + ' 点 / 杂波 ' + fr.radar.clutter : '—'],
      ['点云', fr.cloud ? fr.cloud.shown + ' / ' + fr.cloud.n + ' 点'
        : (CLOUDIDX && CLOUDIDX[sc.id] ? '按需加载（' + CLOUDIDX[sc.id].points + ' 点）' : '—')],
      ['SAR', fr.sar ? (isNum(fr.sar.snr_db) ? f1(fr.sar.snr_db) + ' dB SNR' : '有') : '—']
    ]);

    var nMods = 0;
    (sc.frames || []).forEach(function (f) { nMods = Math.max(nMods, Object.keys(f.mods || {}).length); });
    kvInto($('galSceneInfo'), [
      // 类别取 sc.cls 的固定译名。此前是从标题里截「·」后半段，标题一换成
      // 「空中 · 迎头」这种带场景名的写法，类别就显示成「迎头」。
      ['目标类别', CLS_LABEL[sc.cls] || sc.cls || '—'],
      ['雷达波段', sc.band ? sc.band + ' 频段' : '—'],
      ['帧数', sc.frames.length + ' 帧（晴/阴/雨循环）'],
      ['模态', nMods + ' 路']
    ]);

    var th = $('galThumbs');
    th.innerHTML = '';
    var lazy = !$('galLazy') || $('galLazy').checked;
    sc.frames.forEach(function (f) {
      var e = f.mods ? f.mods[S.mod] : null;
      if (!e) return;
      var box = el('div', 'card' + (f.i === fr.i ? ' on-th' : ''));
      var im = el('img');
      im.src = lazy ? thumbURL(e) : imgURL(e);
      im.alt = sc.title + ' 帧 ' + (f.i + 1);
      im.loading = 'lazy';
      im.decoding = 'async';
      box.appendChild(im);
      var cap = el('div', 'mono muted');
      cap.style.cssText = 'font-size:11px;margin-top:6px;display:flex;justify-content:space-between';
      cap.appendChild(el('span', '', '帧 ' + (f.i + 1)));
      if (f.weather) cap.appendChild(el('span', '', WNAME[f.weather.name] || ''));
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
  function cmpModList(sc) {
    // SAR has a different imaging geometry (range/azimuth grid), and a wide
    // view has a different FOV, so fading either over an optical frame compares
    // nothing. Keep both out of this slider; they have their own tabs.
    return sceneMods(sc).filter(function (it) {
      return it.id !== 'optical' && it.id !== 'sar' && !/_wide$/.test(it.id);
    });
  }
  var CMP = { mod: 'mwir' };
  function renderCompare() {
    var sc = curScene(), fr = curFrame();
    if (!sc || !fr) return;
    var list = cmpModList(sc);
    if (!list.length) {
      $('cmpBLabel').textContent = '—';
      $('cmpB').style.display = 'none';
      $('cmpTL').textContent = '这份数据集只有可见光一路，没有可对照的模态';
      return;
    }
    var ok = false;
    list.forEach(function (it) { if (it.id === CMP.mod) ok = true; });
    if (!ok) CMP.mod = list[0].id;

    seg($('cmpScene'), sceneList(), String(S.scene), function (id) {
      S.scene = parseInt(id, 10); S.frame = 0; renderCompare(); galSync();
    });
    seg($('cmpMod'), list, CMP.mod, function (m) { CMP.mod = m; renderCompare(); });

    var a = fr.mods ? fr.mods.optical : null, b = fr.mods ? fr.mods[CMP.mod] : null;
    var A2 = $('cmpA'), B2 = $('cmpB');
    if (a) {
      aspectStyle($('cmpViewer'), a.w, a.h);
      A2.src = imgURL(a);
      A2.style.display = 'block';
    }
    if (b) {
      B2.src = imgURL(b);
      B2.style.display = 'block';
      B2.style.opacity = String(parseInt($('cmpMix').value, 10) / 100);
      if (a) { B2.style.width = A2.clientWidth + 'px'; B2.style.height = A2.clientHeight + 'px'; }
    } else { B2.style.display = 'none'; }
    $('cmpBLabel').textContent = modLabel(CMP.mod);
    $('cmpTL').textContent = sc.title + ' · 帧 ' + (fr.i + 1) + '\n' + weatherText(fr);
    $('cmpCount').textContent = (fr.i + 1) + ' / ' + sc.frames.length;
    var showBoxes = $('cmpBoxes').checked;
    var boxes = showBoxes && a ? a.boxes : [];
    drawBoxes($('cmpSvg'), boxes, a ? a.w : 640, a ? a.h : 480);
  }
  function cmpStep(d) {
    var n = curScene().frames.length;
    S.frame = (S.frame + d + n) % n;
    renderCompare(); galSync();
  }

  // ---------- point cloud ----------
  var PC = { yaw: -0.6, pitch: 0.35, zoom: 1, color: 'intensity', drag: null, spin: true };
  var pcFrameIdx = 0;

  // Point clouds are the only payload big enough to hurt: they were 69% of the
  // old 646 KB manifest for a tab most visitors never open. They now live in
  // data/cloud/<scene>.json and are fetched the first time a scene's cloud is
  // actually needed. The offline bundle (file://) still carries them inline, so
  // the frame already has f.cloud there and nothing is fetched.
  var CLOUD = {}, CLOUD_PEND = {};
  // Per-scene point counts (data/cloud/_index.json). Half the clips only carry a
  // few dozen lidar returns, so the first visit to the point-cloud tab opens the
  // densest scene instead of an empty field -- with a notice saying it moved.
  var CLOUDIDX = null;
  function loadCloudIndex() {
    fetch('data/cloud/_index.json')
      .then(function (r) { if (!r.ok) throw 0; return r.json(); })
      .then(function (d) { CLOUDIDX = d; })
      .catch(function () { CLOUDIDX = null; });
  }
  function lidarPick() {
    if (!CLOUDIDX || !MAN || PC.picked) return;
    var cur = CLOUDIDX[curScene().id];
    if (cur && cur.points >= 800) return;
    var best = -1, bestN = 0;
    MAN.scenes.forEach(function (s, i) {
      var e = CLOUDIDX[s.id];
      if (e && e.points > bestN) { bestN = e.points; best = i; }
    });
    if (best >= 0 && best !== S.scene) {
      S.scene = best; pcFrameIdx = 0; pcSync();
      notice('已切到点云最密的场景：' + MAN.scenes[best].title + '（' + bestN + ' 点）');
    }
  }
  function ensureCloud(sc) {
    var id = sc.id;
    if (CLOUD[id] || CLOUD_PEND[id]) return;
    CLOUD_PEND[id] = true;
    var st = $('pcState');
    if (st) { st.textContent = '点云加载中…'; st.className = 'tag warn'; }
    fetch('data/cloud/' + encodeURIComponent(id) + '.json')
      .then(function (r) { if (!r.ok) throw new Error('cloud ' + r.status); return r.json(); })
      .then(function (d) {
        CLOUD[id] = d; CLOUD_PEND[id] = false;
        var s2 = $('pcState');
        if (s2) { s2.textContent = '点云已载入'; s2.className = 'tag ok'; }
        if (CURRENT === 'lidar') { pcSync(); resizePC(); }
      })
      .catch(function () {
        CLOUD_PEND[id] = false;
        var s3 = $('pcState');
        if (s3) { s3.textContent = '点云不可用'; s3.className = 'tag bad'; }
        drawPC();
      });
  }
  function pcCloud() {
    var sc = curScene();
    var f = sc.frames[Math.min(pcFrameIdx, sc.frames.length - 1)];
    if (!f) return null;
    if (f.cloud) return f.cloud;
    var c = CLOUD[sc.id];
    if (c && c[String(f.i)] !== undefined) return c[String(f.i)];
    // Do not fetch 50-150 KB of point cloud just because boot() paints every
    // tab once; wait until the visitor is actually looking at this one.
    if (!c && CURRENT === 'lidar') ensureCloud(sc);
    return null;
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
    var dp = Math.min(window.devicePixelRatio || 1, 2);
    var cl = pcCloud();
    var pend = CLOUD_PEND[curScene() && curScene().id];
    if (!cl || !cl.x.length) {
      $('pcTL').textContent = pend ? '正在载入点云…' : '无点云数据';
      g.fillStyle = 'rgba(150,180,210,0.55)';
      g.font = Math.round(13 * dp) + 'px monospace';
      g.fillText(pend ? '点云按场景按需加载，稍等片刻'
                      : '该场景未导出点云（PLY），只导出了图像模态', 18 * dp, H / 2);
      return;
    }

    // Orthographic camera that fits itself to the cloud. The old version
    // divided by a fixed centimetre constant, so a 40 m rover and a 900 m UAV
    // did not both land inside the canvas -- the 3D view was blank for most
    // scenes. Everything below is in metres, centred on the cloud centroid.
    var ctr = cloudCenter(cl);
    var cy = Math.cos(PC.yaw), sy = Math.sin(PC.yaw);
    var cp = Math.cos(PC.pitch), sp = Math.sin(PC.pitch);
    var n = cl.x.length;
    var xs = new Float32Array(n), ys = new Float32Array(n), ds = new Float32Array(n);
    var radius = 1e-3, i;
    for (i = 0; i < n; i++) {
      var x = (cl.x[i] - ctr[0]) / 100;
      var y = (cl.y[i] - ctr[1]) / 100;
      var z = (cl.z[i] - ctr[2]) / 100;
      radius = Math.max(radius, Math.sqrt(x * x + y * y + z * z));
      var x1 = x * cy - y * sy;
      var y1 = x * sy + y * cy;
      xs[i] = x1;
      ys[i] = y1 * sp + z * cp;
      ds[i] = y1 * cp - z * sp;
    }
    // Fit to the 85th percentile radius, not the maximum: one stray far-field
    // return (a fence at 2.6 km next to a 3 m rover) would otherwise shrink the
    // target to a handful of pixels in the middle of a huge empty box.
    var dists = new Float32Array(n);
    for (i = 0; i < n; i++) dists[i] = Math.sqrt(xs[i] * xs[i] + ys[i] * ys[i] + ds[i] * ds[i]);
    var sorted = Array.prototype.slice.call(dists).sort(function (a, b) { return a - b; });
    var r85 = sorted[Math.min(n - 1, Math.floor(n * 0.85))] || radius;
    var scale = Math.min(W, H) * 0.42 * PC.zoom / Math.max(r85, radius * 0.18, 1e-3);
    var px = new Float32Array(n), py = new Float32Array(n);
    for (i = 0; i < n; i++) {
      px[i] = W / 2 + xs[i] * scale;
      py[i] = H / 2 - ys[i] * scale;
    }

    // Bounding box in world space -> the same transform. Gives the eye a
    // depth cue that a bare point cloud does not have.
    var mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    for (i = 0; i < n; i++) {
      var wx = (cl.x[i] - ctr[0]) / 100, wy = (cl.y[i] - ctr[1]) / 100, wz = (cl.z[i] - ctr[2]) / 100;
      mn[0] = Math.min(mn[0], wx); mx[0] = Math.max(mx[0], wx);
      mn[1] = Math.min(mn[1], wy); mx[1] = Math.max(mx[1], wy);
      mn[2] = Math.min(mn[2], wz); mx[2] = Math.max(mx[2], wz);
    }
    function proj(wx, wy, wz) {
      var a1 = wx * cy - wy * sy;
      var b1 = wx * sy + wy * cy;
      return [W / 2 + a1 * scale, H / 2 - (b1 * sp + wz * cp) * scale];
    }
    var corners = [];
    for (var ci = 0; ci < 8; ci++) {
      corners.push(proj(ci & 1 ? mx[0] : mn[0], ci & 2 ? mx[1] : mn[1], ci & 4 ? mx[2] : mn[2]));
    }
    g.strokeStyle = 'rgba(56,189,248,0.18)';
    g.lineWidth = Math.max(1, dp * 0.8);
    [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]].forEach(function (e) {
      g.beginPath();
      g.moveTo(corners[e[0]][0], corners[e[0]][1]);
      g.lineTo(corners[e[1]][0], corners[e[1]][1]);
      g.stroke();
    });

    var order = new Array(n);
    for (i = 0; i < n; i++) order[i] = i;
    order.sort(function (a, b) { return ds[a] - ds[b]; });

    var OCOL = ['#38bdf8', '#fbbf24', '#4ade80', '#fb7185', '#a78bfa', '#67e8f9'];
    var rad = Math.max(Math.min(W, H) / 620, 1.2) * dp;
    for (var k = 0; k < n; k++) {
      var j = order[k];
      var depth = 0.55 + 0.45 * Math.max(0, Math.min(1, (ds[j] - ds[order[0]]) / (Math.abs(ds[order[n - 1]] - ds[order[0]]) || 1)));
      if (PC.color === 'intensity') {
        var v = 0.25 + 0.75 * (cl.i[j] / 255);
        g.fillStyle = 'rgba(' + Math.round((90 * v + 30) * depth) + ',' +
          Math.round((200 * v + 20) * depth) + ',' + Math.round(255 * v * depth) + ',0.92)';
      } else if (PC.color === 'object') {
        g.fillStyle = OCOL[((cl.o[j] % 6) + 6) % 6];
      } else {
        var w2 = Math.max(0, Math.min(1, cl.i[j] / 255));
        g.fillStyle = 'rgba(' + Math.round(255 * w2) + ',' + Math.round(140 + 90 * w2) + ',' +
          Math.round(60 * (1 - w2)) + ',0.92)';
      }
      g.beginPath();
      g.arc(px[j], py[j], rad, 0, 6.2832);
      g.fill();
    }

    var span = Math.round(Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]));
    $('pcTL').textContent = '点云 ' + cl.n + ' 点（绘制 ' + cl.shown + '）\n点云跨度 ' +
      span + ' m · 回波率 ' + f1((cl.hit || 0) * 100) + '% · 量程 ' + f0(cl.maxr || 0) + ' m';
    $('pcBR').textContent = '拖动旋转 · 滚轮缩放 · 双击复位';
  }
  function pcSync() {
    var sc = curScene();
    seg($('pcScene'), sceneList(), String(S.scene), function (id) {
      PC.picked = true;
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

    // Read through pcCloud(), not fr.cloud: since the clouds moved to
    // data/cloud/, the frame object no longer carries them and this panel used
    // to sit at "—" while the canvas was already drawing points.
    var cl = pcCloud();
    kvInto($('pcInfo'), [
      ['点数', cl ? cl.n : (CLOUD_PEND[sc.id] ? '载入中…' : '—')],
      ['回波率', cl ? f1((cl.hit || 0) * 100) + '%' : '—'],
      ['量程', cl ? f0(cl.maxr || 0) + ' m' : '—'],
      ['帧序', (pcFrameIdx + 1) + ' / ' + sc.frames.length]
    ]);

    var mods = (fr && fr.mods) || {};
    var rimg = mods.lidar_range;
    if (rimg) { $('pcRangeImg').src = imgURL(rimg); $('pcRangeImg').style.display = 'block'; }
    else { $('pcRangeImg').removeAttribute('src'); $('pcRangeImg').style.display = 'none'; }
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

  // The range/angle map is 95% noise floor: raw amplitude sits in 0..103 with a
  // mean of 4, so at fit-to-width it is indistinguishable from a black square.
  // Percentile stretch is display-only -- the exported pixels never change, and
  // the numeric dB values still come from the manifest, not from this canvas.
  var RDC = { canvas: null };
  // Scope-ish ramp: the low end stays near black so noise does not turn the
  // panel into TV static, and only the top decades carry warm colour.
  var RDSTOPS = [[0.00, 5, 7, 10], [0.16, 11, 42, 91], [0.34, 14, 116, 144],
                 [0.52, 22, 163, 74], [0.70, 234, 179, 8], [0.86, 249, 115, 22],
                 [1.00, 255, 255, 255]];
  function RDLUT(lo, hi) {
    var lut = new Uint8Array(256 * 3), k, c;
    for (k = 0; k < 256; k++) {
      var t = Math.max(0, Math.min(1, (k - lo) / (hi - lo)));
      var j = 0;
      while (j < RDSTOPS.length - 2 && t > RDSTOPS[j + 1][0]) j++;
      var a = RDSTOPS[j], b = RDSTOPS[j + 1];
      var f = (t - a[0]) / Math.max(1e-6, b[0] - a[0]);
      for (c = 1; c <= 3; c++) lut[k * 3 + c - 1] = Math.round(a[c] + (b[c] - a[c]) * f);
    }
    return function (v) { return [lut[v * 3], lut[v * 3 + 1], lut[v * 3 + 2]]; };
  }
  function stretchInto(img) {
    var off = RDC.canvas || (RDC.canvas = document.createElement('canvas'));
    var w = img.naturalWidth, h = img.naturalHeight;
    off.width = w; off.height = h;
    var c = off.getContext('2d');
    c.drawImage(img, 0, 0, w, h);
    var d = c.getImageData(0, 0, w, h);          // throws on file:// (tainted)
    var px = d.data, hist = new Uint32Array(256), i;
    for (i = 0; i < px.length; i += 4) hist[px[i]]++;
    var total = w * h, acc = 0, lo = 0, hi = 255;
    // The window is the operator's choice, not the data's: the noise plateau
    // sits under p90, so anchoring the low end there keeps the floor dark while
    // everything above p99.9 clips to white. Without this the map is either a
    // black rectangle or a field of TV static.
    for (i = 0; i < 256; i++) { acc += hist[i]; if (acc >= total * 0.90) { lo = i; break; } }
    acc = 0;
    for (i = 255; i >= 0; i--) { acc += hist[i]; if (acc >= total * 0.001) { hi = i; break; } }
    if (hi - lo < 6) hi = Math.min(255, lo + 6);
    var lut = RDLUT(lo, hi);
    for (i = 0; i < px.length; i += 4) {
      var s = lut(px[i]);
      px[i] = s[0]; px[i + 1] = s[1]; px[i + 2] = s[2];
    }
    c.putImageData(d, 0, 0);
    return off;
  }
  function drawRdStretch() {
    var cv = $('rdStretchCv'), img = $('rdImg');
    if (!cv || !img) return;
    var on = !$('rdStretch') || $('rdStretch').checked;
    if (!on || !img.naturalWidth || !img.src) {
      cv.style.display = 'none';
      img.style.opacity = '';
      return;
    }
    var box = cv.parentNode.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.max(1, Math.round(box.width * dpr));
    cv.height = Math.max(1, Math.round(box.height * dpr));
    var off;
    try { off = stretchInto(img); } catch (e) { cv.style.display = 'none'; img.style.opacity = ''; return; }
    var g = cv.getContext('2d');
    g.clearRect(0, 0, cv.width, cv.height);
    var s = Math.min(cv.width / img.naturalWidth, cv.height / img.naturalHeight);
    var dw = img.naturalWidth * s, dh = img.naturalHeight * s;
    g.drawImage(off, (cv.width - dw) / 2, (cv.height - dh) / 2, dw, dh);
    cv.style.display = 'block';
    img.style.opacity = '0';
  }

  function rdSync() {
    var sc = curScene(), fr = curFrame();
    if (!sc || !fr) return;
    seg($('rdScene'), sceneList(), String(S.scene), function (id) {
      S.scene = parseInt(id, 10); S.frame = 0; rdSync(); galSync();
    });

    var e = (fr.mods && (fr.mods.radar_ra || fr.mods.radar)) || null;
    if (e) {
      $('rdImg').src = imgURL(e);
      aspectStyle($('rdViewer'), e.w, e.h);
      drawRdStretch();
    } else {
      $('rdImg').removeAttribute('src');
      $('rdStretchCv').style.display = 'none';
    }
    $('rdRange').max = String(sc.frames.length - 1);
    $('rdRange').value = String(fr.i);
    $('rdCount').textContent = (fr.i + 1) + ' / ' + sc.frames.length;
    $('rdTL').textContent = sc.title + '\n' + (sc.band ? sc.band + ' 频段 · ' : '') + weatherText(fr);
    drawRdMark(fr);

    var r = fr.radar;
    var snrs = r && r.pts ? r.pts.map(function (p) { return p[3]; }) : [];
    kvInto($('rdInfo'), [
      ['检出点', r ? r.n : '—'],
      ['杂波点', r ? r.clutter : '—'],
      ['最强 SNR', snrs.length ? f1(Math.max.apply(null, snrs)) + ' dB' : '—']
    ]);

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
      (bud.by_weather || []).forEach(function (w) {
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
      var neg = (bud.by_weather || []).filter(function (w) { return w.margin_db < 0; });
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
    if (!(r && r.boxes && r.boxes.length)) {
      bx.appendChild(el('div', 'muted', '该帧没有导出的 3D 包围盒'));
    }
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
    } else {
      $('rdBR').textContent = '';
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
    $('rdPtTL').textContent = (fr && fr.weather ? (WNAME[fr.weather.preset] || WNAME[fr.weather.name] || '') : '') +
      '  检出 ' + rd.n + ' 点' + (rd.clutter ? ' / 杂波 ' + rd.clutter : '');
  }

  // ---------- SAR ----------
  // The shipped PNG is already an 8-bit amplitude image that the exporter
  // stretched once. The 30/45/60 dB buttons therefore re-map that amplitude with
  // a gamma curve: a narrow window lifts weak scatterers, a wide window pulls the
  // bright ones back. No dB is re-derived in the browser and nothing is claimed
  // to be a calibration -- the caption under the plot says exactly this.
  var SAR = { dr: 45, src: '', simple: false, zoom: null };
  var SAR_DR = [30, 45, 60];
  var sarTimer = null;

  function sceneHasSar(sc) {
    if (!sc) return false;
    if (sc.sar_meta) return true;
    return (sc.frames || []).some(function (f) { return !!(f.mods && f.mods.sar); });
  }
  function sarMeta(fr, sc) { return (fr && fr.sar) || (sc && sc.sar_meta) || null; }

  function sarSync() {
    var sc = curScene(), fr = curFrame();
    if (!sc || !fr) return;
    if (ensureScene(sceneHasSar)) { sarSync(); return; }
    seg($('sarScene'), sceneList(), String(S.scene), function (id) {
      S.scene = parseInt(id, 10); S.frame = 0; sarSync();
    });
    seg($('sarDr'), SAR_DR.map(function (d) {
      return { id: String(d), label: d + ' dB' };
    }), String(SAR.dr), function (d) { SAR.dr = parseInt(d, 10); sarSync(); });

    var has = sceneHasSar(sc);
    $('sarBody').hidden = !has;
    $('sarNone').hidden = has;
    $('sarDr').hidden = !has;
    $('sarPill').textContent = has
      ? sc.title + ' · 帧 ' + (fr.i + 1) + '/' + sc.frames.length
      : '无 SAR 数据';
    $('sarCbLo').textContent = '暗 · 噪底';
    $('sarCbHi').textContent = '亮 · 饱和';
    $('sarCbCur').textContent = SAR.dr + ' dB · γ=' + f2(SAR.dr / 45);
    if (!has) return;

    var entry = fr.mods ? fr.mods.sar : null;
    var meta = sarMeta(fr, sc) || {};

    sarSetImage(entry);
    if (!entry) {
      $('sarTL').textContent = sc.title + ' · 帧 ' + (fr.i + 1) + '\n该帧没有 SAR 图';
      $('sarBR').textContent = '—';
      $('sarSvg').innerHTML = '';
      $('sarTicks').innerHTML = '';
    } else {
      $('sarTL').textContent = sc.title + ' · 帧 ' + (fr.i + 1) + '/' + sc.frames.length +
        '\n行 = 方位向 · 列 = 地距向 · ' + entry.w + '×' + entry.h;
      $('sarBR').textContent = '拉伸 ' + SAR.dr + ' dB · γ=' + f2(SAR.dr / 45) +
        (SAR.simple ? '（CSS 近似）' : '');
      var small = entry.boxes && entry.boxes.length &&
        Math.max(entry.boxes[0].x1 - entry.boxes[0].x0, entry.boxes[0].y1 - entry.boxes[0].y0) <
        Math.min(entry.w, entry.h) * 0.05;
      drawBoxes($('sarSvg'), entry.boxes, entry.w, entry.h,
        { locator: small, locLabel: small ? '目标' : '', color: '#ffb454' });
    }

    $('sarRange').max = String(sc.frames.length - 1);
    $('sarRange').value = String(fr.i);
    $('sarCount').textContent = (fr.i + 1) + ' / ' + sc.frames.length;

    var res = (meta.res_range_m !== undefined) ? meta.res_range_m : meta.res_range;
    var resA = (meta.res_az_m !== undefined) ? meta.res_az_m : meta.res_az;
    var geo = [
      ['距离分辨率', isNum(res) ? f1(res) + ' m' : '—'],
      ['方位分辨率', isNum(resA) ? f1(resA) + ' m' : '—'],
      ['测绘带 近 / 远', (isNum(meta.near_m) && isNum(meta.far_m))
        ? fmtM(meta.near_m) + ' / ' + fmtM(meta.far_m) : '—'],
      ['测绘带宽', isNum(meta.width_m) ? fmtM(meta.width_m) : '—']
    ];
    var band = [meta.band, meta.polarization].filter(Boolean).join(' ');
    if (band) geo.push(['波段 / 极化', band]);
    kvInto($('sarGeo'), geo);

    kvInto($('sarStats'), [
      ['峰值 / 均值', (isNum(meta.peak_db) ? f1(meta.peak_db) : '—') + ' / ' +
        (isNum(meta.mean_db) ? f1(meta.mean_db) : '—') + ' dB'],
      ['噪声底', isNum(meta.noise_db) ? f1(meta.noise_db) + ' dB' : '—'],
      ['SNR', isNum(meta.snr_db) ? f1(meta.snr_db) + ' dB' : '—'],
      ['动态范围', isNum(meta.dynamic_range_db) ? f1(meta.dynamic_range_db) + ' dB' : '—'],
      ['目标峰值', isNum(meta.target_peak_db) ? f1(meta.target_peak_db) + ' dB' : '—'],
      ['目标对比度', isNum(meta.target_contrast_db) ? f1(meta.target_contrast_db) + ' dB' : '—'],
      ['散射中心', isNum(meta.n_scatterers) ? f0(meta.n_scatterers) : '—']
    ]);

    var p = meta.platform || {};
    kvInto($('sarPlat'), [
      ['平台高度', isNum(p.altitude_m) ? f0(p.altitude_m) + ' m' : '—'],
      ['平台速度', isNum(p.speed_mps) ? f1(p.speed_mps) + ' m/s' : '—'],
      ['航向角', isNum(p.heading_deg) ? f1(p.heading_deg) + '°' : '—'],
      ['斜视角', isNum(p.squint_deg) ? f1(p.squint_deg) + '°' : '—']
    ]);

    $('sarNote').innerHTML = '拉伸说明：导出的 PNG 已经是<b>导出时做过一次 DB 拉伸</b>的 8-bit 灰度，' +
      '这里切换 30 / 45 / 60 dB 只是对这份幅度做 <b>gamma 重映射</b>' +
      '（' + SAR.dr + ' dB 对应 γ=' + f2(SAR.dr / 45) + '），' +
      '窄窗口把弱散射提亮、宽窗口压回高亮，目的是让只有几个像素的目标可判读，<b>不改动原始数据</b>。' +
      (SAR.simple ? '当前是 <code>file://</code> 打开，画布像素不可读，' +
        '灰度映射退化为 CSS 滤镜近似（结论一致，只有亮度曲线略有差异）。' : '');

    sarLayout();
  }

  function sarStop() {
    if (!sarTimer) return;
    clearInterval(sarTimer);
    sarTimer = null;
    if ($('sarPlay')) $('sarPlay').textContent = '播放';
  }

  function sarSetImage(entry) {
    var img = $('sarImg'), cv = $('sarCanvas');
    if (!entry) {
      img.removeAttribute('src');
      img.style.display = 'none';
      cv.style.display = 'none';
      SAR.src = '';
      return;
    }
    var src = imgURL(entry);
    var paint = function () {
      var w = img.naturalWidth || entry.w || 4;
      var h = img.naturalHeight || entry.h || 3;
      sarPaint(img, cv, w, h);
      sarLayout();
    };
    if (SAR.src === src && img.complete && img.naturalWidth) { paint(); return; }
    SAR.src = src;
    img.onload = paint;
    img.onerror = function () {
      img.style.display = 'none';
      cv.style.display = 'none';
      $('sarTL').textContent = 'SAR 图加载失败：' + src;
    };
    img.src = src;
    if (img.complete && img.naturalWidth) paint();
  }

  function sarPaint(img, cv, w, h) {
    var gam = SAR.dr / 45;
    cv.width = w;
    cv.height = h;
    var g = cv.getContext('2d');
    g.clearRect(0, 0, w, h);
    g.drawImage(img, 0, 0, w, h);
    try {
      var d = g.getImageData(0, 0, w, h);
      var p = d.data, lut = new Uint8Array(256), i;
      for (i = 0; i < 256; i++) lut[i] = Math.round(255 * Math.pow(i / 255, gam));
      for (i = 0; i < p.length; i += 4) {
        var v = lut[p[i]];
        p[i] = v; p[i + 1] = v; p[i + 2] = v;
      }
      g.putImageData(d, 0, 0);
      SAR.simple = false;
    } catch (e) {
      // file:// gives every document an opaque origin, so the canvas is tainted
      // and getImageData throws. Keep the page usable: switch to a CSS filter.
      SAR.simple = true;
    }
    cv.style.display = SAR.simple ? 'none' : 'block';
    img.style.display = SAR.simple ? 'block' : 'none';
    img.style.filter = SAR.simple
      ? 'grayscale(1) brightness(' + (1 / gam).toFixed(2) + ') contrast(' +
        (0.9 + 0.2 * gam).toFixed(2) + ')'
      : 'none';
  }

  function sarLayout() {
    var host = $('sarViewer'), plot = $('sarPlot'), ticks = $('sarTicks');
    if (!host || !plot || !ticks) return;
    var img = $('sarImg'), cv = $('sarCanvas');
    var entry = curFrame() && curFrame().mods ? curFrame().mods.sar : null;
    if (!entry) { ticks.innerHTML = ''; return; }
    var w = (img && img.naturalWidth) || cv.width || entry.w;
    var h = (img && img.naturalHeight) || cv.height || entry.h;
    if (!w || !h) return;
    host.style.aspectRatio = w + ' / ' + h;

    var vr = host.getBoundingClientRect();
    var pr = plot.getBoundingClientRect();
    if (vr.width < 20 || !pr.width) return;
    var s = Math.min(vr.width / w, vr.height / h);
    var dw = w * s, dh = h * s;
    var ox = (vr.left - pr.left) + (vr.width - dw) / 2;
    var oy = (vr.top - pr.top) + (vr.height - dh) / 2;

    var meta = sarMeta(curFrame(), curScene()) || {};
    ticks.innerHTML = '';
    var near = num(meta.near_m), far = num(meta.far_m), wid = num(meta.width_m);
    var i, t, f;

    if (near !== null && far !== null && far > near) {
      for (i = 0; i <= 4; i++) {
        f = i / 4;
        t = el('div', 'tick x', fmtAxis(near + (far - near) * f));
        t.style.left = (ox + dw * f).toFixed(1) + 'px';
        t.style.top = (oy + dh + 6).toFixed(1) + 'px';
        ticks.appendChild(t);
      }
      var xn = el('div', 'axis-name', '地距 (m) · 近 → 远');
      xn.style.left = (ox + dw / 2).toFixed(1) + 'px';
      xn.style.top = (oy + dh + 27).toFixed(1) + 'px';
      xn.style.transform = 'translateX(-50%)';
      ticks.appendChild(xn);
    }
    if (wid !== null) {
      for (i = 0; i <= 4; i++) {
        f = i / 4;
        var val = (f - 0.5) * wid;
        t = el('div', 'tick y', f === 0.5 ? '0' : (val > 0 ? '+' : '') + fmtAxis(val));
        t.style.right = (Math.max(0, pr.width - ox + 6)).toFixed(1) + 'px';
        t.style.top = (oy + dh * f).toFixed(1) + 'px';
        ticks.appendChild(t);
      }
      var yn = el('div', 'axis-name', '方位 (m) · 相对中心');
      yn.style.left = ox.toFixed(1) + 'px';
      yn.style.top = '1px';
      ticks.appendChild(yn);
    }
  }

  (function bindSarZoom() {
    var host = $('sarViewer');
    if (!host) return;
    SAR.zoom = makeZoom(host, $('sarStage'), {
      clickScale: 3.5,
      origin: function () {
        var e = curFrame() && curFrame().mods ? curFrame().mods.sar : null;
        if (!e || !e.boxes || !e.boxes.length) return [0.5, 0.5];
        var b = e.boxes[0];
        return [((b.x0 + b.x1) / 2) / (e.w || 1), ((b.y0 + b.y1) / 2) / (e.h || 1)];
      }
    });
  })();

  // ---------- seeker: narrow vs wide field of view ----------
  var SK = { timer: null, seekZoom: null, wideZoom: null };

  function sceneHasViews(sc) {
    if (!sc) return false;
    return (sc.frames || []).some(function (f) {
      return !!(f.views && Object.keys(f.views).length) ||
        !!(f.mods && (f.mods.optical_wide || f.mods.mwir_wide || f.mods.lwir_wide));
    });
  }

  // Which image belongs to a view? The exporter writes the narrow view under the
  // untagged modality key and the wide one under "<modality>_wide".
  function panePick(fr, which) {
    var mods = (fr && fr.mods) || {};
    var order = which === 'wide'
      ? ['optical_wide', 'mwir_wide', 'lwir_wide']
      : ['optical', 'mwir', 'lwir'];
    var i;
    for (i = 0; i < order.length; i++) if (mods[order[i]]) return { key: order[i], e: mods[order[i]] };
    var keys = Object.keys(mods);
    for (i = 0; i < keys.length; i++) {
      var k = keys[i];
      if (k === 'sar' || k === 'radar_ra' || k.indexOf('lidar') === 0) continue;
      if ((k.indexOf('_wide') > 0) === (which === 'wide')) return { key: k, e: mods[k] };
    }
    return null;
  }

  // views[].name comes from the scene config, so it is not guaranteed to be
  // "seeker"/"wide". Fall back to the narrowest / widest fov when the names differ.
  function viewOf(fr, which) {
    var vs = (fr && fr.views) || {};
    var ks = Object.keys(vs);
    if (!ks.length) return null;
    if (vs[which]) return vs[which];
    var sorted = ks.slice().sort(function (a, b) {
      return (vs[a].fov_deg || 0) - (vs[b].fov_deg || 0);
    });
    return vs[which === 'wide' ? sorted[sorted.length - 1] : sorted[0]];
  }
  function targetCenter(e, w, h) {
    if (e && e.boxes && e.boxes.length) {
      var b = e.boxes[0];
      return [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];
    }
    return [w / 2, h / 2];
  }
  function boxSpanPx(e) {
    if (!e || !e.boxes || !e.boxes.length) return 0;
    var b = e.boxes[0];
    return Math.max(b.x1 - b.x0, b.y1 - b.y0);
  }

  function skPane(which) {
    var fr = curFrame() || {};
    var v = viewOf(fr, which);
    var pick = panePick(fr, which);
    var e = pick ? pick.e : null;
    var w = (v && v.w) || (e && e.w) || 640;
    var h = (v && v.h) || (e && e.h) || 480;
    // drawing space: the overlay must use the pixel grid the *labels* are written
    // in, which is the render resolution recorded in views[].w/h. The shipped PNG
    // may be a downscale of that (export_web.py caps images at 900 px), and a
    // viewBox of the original grid still lands on the same pixels after "meet".
    var dw = (v && v.w) || (e && e.w) || 640, dh = (v && v.h) || (e && e.h) || 480;
    var span = (v && isNum(v.target_px)) ? v.target_px
      : (isNum(fr.target_px) ? fr.target_px : boxSpanPx(e));
    return {
      fr: fr, v: v, e: e, w: w, h: h, dw: dw, dh: dh,
      span: span,
      ratio: w ? span / w * 100 : 0,
      range: num(fr.range_m)
    };
  }

  function skFillStats(host, P, missing) {
    if (!host) return;
    host.innerHTML = '';
    var cells = [
      { k: '目标像素跨度', v: (!missing && P.span) ? f1(P.span) : '—', u: 'px', c: '' },
      { k: '画面占比（线性）', v: (!missing && P.span) ? f2(P.ratio) : '—', u: '%', c: 'accent' },
      { k: '视线距离', v: (!missing && isNum(P.range)) ? fmtM(P.range).replace(/ (m|km)$/, '') : '—',
        u: (!missing && isNum(P.range)) ? (P.range >= 1000 ? 'km' : 'm') : '', c: '' }
    ];
    cells.forEach(function (c) {
      var d = el('div', 'stat' + (c.c ? ' ' + c.c : ''));
      d.appendChild(el('div', 'k', c.k));
      var v = el('div', 'v', c.v);
      if (c.u && c.v !== '—') v.appendChild(el('em', '', c.u));
      d.appendChild(v);
      host.appendChild(d);
    });
  }

  function skRender(which) {
    var wide = which === 'wide';
    var ids = {
      view: wide ? 'skWideView' : 'skSeekView',
      img: wide ? 'skWideImg' : 'skSeekImg',
      svg: wide ? 'skWideSvg' : 'skSeekSvg',
      tl: wide ? 'skWideTL' : 'skSeekTL',
      stats: wide ? 'skWideStats' : 'skSeekStats',
      fov: wide ? 'skWideFov' : 'skSeekFov',
      cap: wide ? 'skWideMagCap' : 'skSeekMagCap'
    };
    var P = skPane(which);
    var host = $(ids.view), img = $(ids.img), svg = $(ids.svg);
    if (!host || !img) return;
    aspectStyle(host, P.dw, P.dh);

    if (!P.e) {
      img.removeAttribute('src');
      img.style.visibility = 'hidden';
      svg.innerHTML = '';
      $(ids.tl).textContent = wide ? '宽视场图缺失' : '窄视场图缺失';
      $(ids.fov).textContent = P.v ? f1(P.v.fov_deg) + '° FOV' : '视场未知';
      $(ids.cap).textContent = '无图';
      skFillStats($(ids.stats), P, true);
      return;
    }
    img.style.visibility = 'visible';
    var src = imgURL(P.e);
    if (img.getAttribute('src') !== src) {
      img.onload = function () { skMag(which); };
      img.setAttribute('src', src);
    }
    drawBoxes(svg, P.e.boxes, P.dw, P.dh,
      { locator: true, locLabel: '目标', color: '#37c2ff' });

    $(ids.fov).textContent = P.v
      ? f1(P.v.fov_deg) + '° FOV' + (isNum(P.v.roll_deg) && Math.abs(P.v.roll_deg) > 0.05
        ? ' · 滚转 ' + f1(P.v.roll_deg) + '°' : '')
      : '视场未知';
    $(ids.tl).textContent = (curScene().title || '') + ' · 帧 ' + ((P.fr.i || 0) + 1) + '\n' +
      (wide ? '宽视场（搜索）' : '窄视场（机载导引）') +
      (P.v ? ' · ' + P.dw + '×' + P.dh : '');
    skFillStats($(ids.stats), P, false);
    skMag(which);
  }

  // The magnifier is the whole point of this tab: at fit-to-width a 3 px target
  // is a speck. A nearest-neighbour crop centred on the box centre (4x the box
  // span, clamped) makes it readable without touching the source pixels.
  function skMag(which) {
    var wide = which === 'wide';
    var cv = $(wide ? 'skWideMag' : 'skSeekMag');
    var cap = $(wide ? 'skWideMagCap' : 'skSeekMagCap');
    var img = $(wide ? 'skWideImg' : 'skSeekImg');
    if (!cv || !img) return;
    var P = skPane(which);
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var SZ = cv.clientWidth || 112;
    cv.width = Math.round(SZ * dpr);
    cv.height = Math.round(SZ * dpr);
    var g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#05070a';
    g.fillRect(0, 0, SZ, SZ);
    if (!P.e || !img.naturalWidth) { if (cap) cap.textContent = '无图'; return; }

    var k = img.naturalWidth / (P.dw || img.naturalWidth);
    var iw = img.naturalWidth, ih = img.naturalHeight;
    var c = targetCenter(P.e, P.dw, P.dh);
    var span = Math.max(P.span || boxSpanPx(P.e) || 4, 2) * k;
    var crop = Math.max(span * 4, 10);
    crop = Math.min(crop, Math.min(iw, ih) - 1);
    var sx = Math.max(0, Math.min(iw - crop, c[0] * k - crop / 2));
    var sy = Math.max(0, Math.min(ih - crop, c[1] * k - crop / 2));
    var sq = Math.min(crop, ih - sy, iw - sx);

    g.imageSmoothingEnabled = false;
    g.drawImage(img, sx, sy, sq, sq, 0, 0, SZ, SZ);

    var sc = SZ / sq;
    var b = P.e.boxes && P.e.boxes[0];
    if (b) {
      g.strokeStyle = '#37c2ff';
      g.lineWidth = 1;
      g.strokeRect((b.x0 * k - sx) * sc, (b.y0 * k - sy) * sc,
        Math.max((b.x1 - b.x0) * k * sc, 3), Math.max((b.y1 - b.y0) * k * sc, 3));
    }
    g.strokeStyle = 'rgba(255,255,255,.5)';
    g.beginPath();
    g.moveTo(SZ / 2 - 7, SZ / 2); g.lineTo(SZ / 2 - 2, SZ / 2);
    g.moveTo(SZ / 2 + 2, SZ / 2); g.lineTo(SZ / 2 + 7, SZ / 2);
    g.moveTo(SZ / 2, SZ / 2 - 7); g.lineTo(SZ / 2, SZ / 2 - 2);
    g.moveTo(SZ / 2, SZ / 2 + 2); g.lineTo(SZ / 2, SZ / 2 + 7);
    g.stroke();
    if (cap) {
      // sq = crop width in source pixels; SZ/sq is the pixel-level zoom applied
      cap.textContent = Math.round(sq) + 'px · ' + f1(SZ / sq) + '×';
      cap.title = '取景 ' + Math.round(sq) + ' 源像素，按 ' + f1(SZ / sq) +
        ' 倍像素放大显示（目标只有几像素时用来判读）';
    }
  }

  function skChart(svg, n, series, opt) {
    if (!svg) return;
    opt = opt || {};
    var legend = series.filter(function (s) { return s.label; }).length > 0;
    var W = 620, H = legend ? 190 : 176, padL = 52, padR = 12;
    var padT = legend ? 36 : 12, padB = 26;
    var pw = W - padL - padR, ph = H - padT - padB;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.innerHTML = '';
    if (!n) return;

    var minV = Infinity, maxV = -Infinity, i, j, s;
    for (i = 0; i < series.length; i++) {
      for (j = 0; j < series[i].vals.length; j++) {
        var v = series[i].vals[j];
        if (isNum(v)) { if (v < minV) minV = v; if (v > maxV) maxV = v; }
      }
    }
    if (!isFinite(minV)) {
      svg.appendChild(svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle', 'font-size': 12 }))
        .textContent = '该场景未导出这条曲线';
      return;
    }
    if (opt.zero) minV = 0;
    if (maxV - minV < 1e-9) { maxV += 1; minV -= 1; }
    var padv = (maxV - minV) * 0.08;
    maxV += padv;
    if (!opt.zero) minV -= padv;

    // X axis is real time when the exporter wrote frame.t, frame index
    // otherwise. Two clips recorded at different frame rates then line up
    // instead of being stretched to the same width.
    var ts = (opt.t && opt.t.length === n && opt.t.every(isNum)) ? opt.t : null;
    if (ts) {
      for (i = 1; i < n; i++) { if (ts[i] < ts[i - 1]) { ts = null; break; } }
    }
    var t0 = ts ? ts[0] : 0, t1 = ts ? ts[n - 1] : n - 1, tspan = t1 - t0;
    var X = function (idx) {
      if (!ts) return padL + (n <= 1 ? pw / 2 : pw * idx / (n - 1));
      if (!(tspan > 0)) return padL + pw / 2;
      return padL + pw * (ts[idx] - t0) / tspan;
    };
    var Y = function (val) { return padT + ph * (1 - (val - minV) / (maxV - minV)); };

    for (i = 0; i <= 3; i++) {
      var yy = padT + ph * i / 3;
      svg.appendChild(svgEl('line', {
        x1: padL, y1: yy, x2: padL + pw, y2: yy,
        stroke: '#1d2634', 'stroke-width': 1
      }));
      var lab = svgEl('text', {
        x: padL - 8, y: yy + 3.5, 'text-anchor': 'end', 'font-size': 10
      });
      lab.textContent = fmtAxis(maxV - (maxV - minV) * i / 3);
      svg.appendChild(lab);
    }
    [0, 0.5, 1].forEach(function (fr_, k) {
      var idx = k === 0 ? 0 : (k === 2 ? n - 1 : Math.round((n - 1) / 2));
      var px = ts ? (padL + pw * fr_) : X(idx);
      var t = svgEl('text', {
        x: px, y: H - 8, 'text-anchor': k === 0 ? 'start' : (k === 2 ? 'end' : 'middle'),
        'font-size': 10
      });
      t.textContent = ts ? fmtAxis(t0 + tspan * fr_) + ' s' : '帧 ' + (idx + 1);
      svg.appendChild(t);
    });
    var axName = svgEl('text', {
      x: padL + pw, y: padT - 4, 'text-anchor': 'end', 'font-size': 10, opacity: 0.7
    });
    axName.textContent = ts ? '横轴 = 时间' : '横轴 = 帧序号（无 t 字段）';
    svg.appendChild(axName);

    series.forEach(function (sr) {
      var d = '', open = false;
      for (var idx = 0; idx < sr.vals.length; idx++) {
        var val = sr.vals[idx];
        if (!isNum(val)) { open = false; continue; }
        d += (open ? 'L' : 'M') + X(idx).toFixed(1) + ' ' + Y(val).toFixed(1);
        open = true;
      }
      if (!d) return;
      svg.appendChild(svgEl('path', {
        d: d, fill: 'none', stroke: sr.color, 'stroke-width': 1.8,
        'stroke-linejoin': 'round', 'stroke-linecap': 'round'
      }));
      for (idx = 0; idx < sr.vals.length; idx++) {
        if (!isNum(sr.vals[idx])) continue;
        svg.appendChild(svgEl('circle', {
          cx: X(idx), cy: Y(sr.vals[idx]), r: 1.9, fill: sr.color
        }));
      }
    });

    var cur = Math.min(S.frame, n - 1);
    svg.appendChild(svgEl('line', {
      x1: X(cur), y1: padT, x2: X(cur), y2: padT + ph,
      stroke: '#37c2ff', 'stroke-width': 1, 'stroke-opacity': 0.55,
      'stroke-dasharray': '3 3'
    }));
    series.forEach(function (sr) {
      var val = sr.vals[cur];
      if (!isNum(val)) return;
      svg.appendChild(svgEl('circle', {
        cx: X(cur), cy: Y(val), r: 3.6, fill: '#0a0d12',
        stroke: sr.color, 'stroke-width': 2
      }));
    });

    // click a column to jump to that frame; column edges sit on the midpoints
    // between samples so uneven frame spacing still gets a full-width target
    for (i = 0; i < n; i++) {
      var lx = i === 0 ? padL : (X(i - 1) + X(i)) / 2;
      var rx = i === n - 1 ? padL + pw : (X(i) + X(i + 1)) / 2;
      var hit = svgEl('rect', {
        x: lx, y: padT, width: Math.max(2, rx - lx), height: ph,
        fill: 'transparent', style: 'cursor:pointer'
      });
      (function (idx) {
        hit.addEventListener('click', function () { S.frame = idx; skSync(); });
      })(i);
      svg.appendChild(hit);
    }

    var lg = padL + 2, lgy = 12;
    series.forEach(function (sr) {
      if (!sr.label) return;
      svg.appendChild(svgEl('rect', {
        x: lg, y: lgy - 5, width: 9, height: 3, fill: sr.color
      }));
      var t = svgEl('text', { x: lg + 14, y: lgy, 'font-size': 10 });
      t.textContent = sr.label;
      svg.appendChild(t);
      lgy += 13;
    });
  }

  function skCharts(sc) {
    var n = sc.frames.length;
    var ranges = sc.frames.map(function (f) { return isNum(f.range_m) ? f.range_m : null; });
    var seekPct = sc.frames.map(function (f) {
      var v = viewOf(f, 'seeker');
      if (!v || !isNum(v.target_px) || !v.w) return null;
      return v.target_px / v.w * 100;
    });
    var widePct = sc.frames.map(function (f) {
      var v = viewOf(f, 'wide');
      if (!v || !isNum(v.target_px) || !v.w) return null;
      return v.target_px / v.w * 100;
    });

    var tAxis = sc.frames.map(function (f) { return isNum(f.t) ? f.t : null; });
    if (tAxis.some(function (v) { return v === null; })) tAxis = null;

    skChart($('skRangeChart'), n, [{ vals: ranges, color: '#37c2ff', label: '视线距离 (m)' }],
      { zero: true, t: tAxis });
    skChart($('skPxChart'), n, [
      { vals: seekPct, color: '#37c2ff', label: '窄视场 2.2°' },
      { vals: widePct, color: '#ffb454', label: '宽视场 18°' }
    ], { zero: true, t: tAxis });

    var fr = curFrame() || {};
    $('skRangeNow').textContent = isNum(fr.range_m) ? '当前 ' + fmtM(fr.range_m) : '未导出 range_m';
    var sp = skPane('seeker');
    $('skPxNow').textContent = sp.span ? '当前 ' + f2(sp.ratio) + ' %' : '未导出 target_px';
  }

  function skSync() {
    var sc = curScene(), fr = curFrame();
    if (!sc || !fr) return;
    if (ensureScene(sceneHasViews)) { skSync(); return; }
    var has = sceneHasViews(sc);
    $('skBody').hidden = !has;
    $('skNone').hidden = has;
    seg($('skScene'), sceneList(), String(S.scene), function (id) {
      S.scene = parseInt(id, 10); S.frame = 0; skSync();
    });
    $('skPill').textContent = has
      ? sc.title + ' · 帧 ' + (fr.i + 1) + '/' + sc.frames.length
      : '无双视场数据';
    if (!has) return;

    $('skRange').max = String(sc.frames.length - 1);
    $('skRange').value = String(fr.i);
    $('skCount').textContent = (fr.i + 1) + ' / ' + sc.frames.length;

    skRender('seeker');
    skRender('wide');
    skCharts(sc);
  }

  function skStop() {
    if (!SK.timer) return;
    clearInterval(SK.timer);
    SK.timer = null;
    if ($('skPlay')) $('skPlay').textContent = '播放';
  }
  function skPlay() {
    if (SK.timer) { skStop(); return; }
    $('skPlay').textContent = '暂停';
    SK.timer = setInterval(function () {
      var n = curScene().frames.length;
      S.frame = (S.frame + 1) % n;
      skSync();
    }, 900);
  }
  function skStep(d) {
    var n = curScene().frames.length;
    S.frame = (S.frame + d + n) % n;
    skSync();
  }

  (function bindSkZoom() {
    if (!$('skSeekView')) return;
    var mk = function (which) {
      return function () {
        var P = skPane(which);
        if (!P.e) return [0.5, 0.5];
        var c = targetCenter(P.e, P.dw, P.dh);
        return [c[0] / (P.dw || 1), c[1] / (P.dh || 1)];
      };
    };
    SK.seekZoom = makeZoom($('skSeekView'), $('skSeekStage'), { clickScale: 6, origin: mk('seeker') });
    SK.wideZoom = makeZoom($('skWideView'), $('skWideStage'), { clickScale: 8, origin: mk('wide') });
  })();

  // ---------- explore ----------
  // Band constants come from data/radar_bands.json, which tools/export_web.py
  // writes straight out of simkit/radar.py -- so the browser cannot drift from
  // the kernel. Nothing here is a hand-copied number.
  var BANDS = null;

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
    if (!BANDS) return;
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

  // ---------- home ----------
  // One showcase clip drives both the hero crossfade and the modality wall: the
  // scene that ships the most modalities per frame (today the low-altitude
  // air-to-ground clip: optical + both IR + both wide views + radar + lidar + SAR).
  var HOME = { scene: 0, frame: 0, mod: 0, wallFrame: 0, timer: null };
  var WALL_MODS = ['optical', 'mwir', 'lwir', 'radar_ra', 'sar', 'lidar_range', 'lidar_intensity'];

  function showcaseScene() {
    var best = 0, n = -1;
    MAN.scenes.forEach(function (s, i) {
      var m = 0;
      s.frames.forEach(function (fr) { m = Math.max(m, Object.keys(fr.mods || {}).length); });
      if (m > n) { n = m; best = i; }
    });
    return best;
  }
  function wallList(fr) {
    var has = {};
    Object.keys((fr && fr.mods) || {}).forEach(function (k) { has[k] = 1; });
    var l = WALL_MODS.filter(function (k) { return has[k]; });
    // never leave a hole: fall back to whatever this frame actually has
    if (!l.length) l = Object.keys((fr && fr.mods) || {});
    return l.slice(0, 6);
  }
  function modNote(sc, fr, k) {
    var e = (fr.mods || {})[k];
    if (!e) return '';
    if (k === 'radar_ra') return (sc.band || 'mmW') + ' 频段';
    if (k === 'sar') return fr.sar ? f2(fr.sar.res_range_m) + ' × ' + f2(fr.sar.res_az_m) + ' m' : 'RD 聚焦';
    if (k === 'lidar_range' || k === 'lidar_intensity') return '905 nm';
    return e.w + '×' + e.h;
  }
  function buildHeroStats() {
    var host = $('heroStats');
    if (!host) return;
    var nFrames = 0, nImgs = 0, mods = {};
    MAN.scenes.forEach(function (s) {
      s.frames.forEach(function (fr) {
        nFrames++;
        Object.keys(fr.mods || {}).forEach(function (k) { mods[k] = 1; nImgs++; });
      });
    });
    var rows = [
      [MAN.scenes.length, '场景', ''],
      [nFrames, '帧', ''],
      [Object.keys(mods).length, '模态', ''],
      [nImgs, '成像', '']
    ];
    host.innerHTML = '';
    rows.forEach(function (r) {
      var d = el('div');
      d.appendChild(el('dt', '', r[1]));
      var dd = el('dd', '', String(r[0]));
      d.appendChild(dd);
      host.appendChild(d);
    });
  }
  // High-resolution hero pack rendered at 1440x1080 by the same kernel
  // (scenes/hero_*.json, out/hero/). Falls back to the manifest frames the site
  // already ships if assets/hero/ is absent -- the page degrades, never breaks.
  var HERO_SHOTS = [];
  function initHeroShots() {
    fetch('assets/hero/index.json', { cache: 'no-cache' })
      .then(function (r) { if (!r.ok) throw new Error('hero ' + r.status); return r.json(); })
      .then(function (list) {
        HERO_SHOTS = (list || []).filter(function (s) { return s && s.file && s.group !== 'wx'; });
        if (!HERO_SHOTS.length) { HERO_SHOTS = []; return; }
        renderWeatherBand(list.filter(function (s) { return s && s.group === 'wx'; }));
        // The manifest may not have arrived yet; heroRender() needs it for the
        // scene title, so only redraw once MAN exists. boot() re-renders anyway.
        if (MAN) { HOME.mod = 0; heroRender(); }
      })
      .catch(function () { HERO_SHOTS = []; });
  }
  function renderWeatherBand(list) {
    var host = $('wxGrid');
    if (!host) return;
    var band = $('wxBand');
    if (!list.length) { if (band) band.hidden = true; return; }
    if (band) band.hidden = false;
    host.innerHTML = '';
    list.forEach(function (s) {
      var card = el('div', 'figcard');
      card.style.cursor = 'default';
      var im = el('img');
      im.src = 'assets/hero/' + s.file;
      im.alt = s.label || '';
      im.loading = 'lazy';
      card.appendChild(im);
      var c = el('div', 'fc');
      c.appendChild(el('b', '', s.label || s.file));
      c.appendChild(el('span', '', s.w + '×' + s.h + ' · 同一场景，换光照与天气参数'));
      card.appendChild(c);
      host.appendChild(card);
    });
  }
  function heroRender() {
    var sc = MAN.scenes[HOME.scene];
    if (!sc) return;
    var img = $('heroImg');
    // --- case 1: the high-resolution pack ---
    if (HERO_SHOTS.length) {
      HOME.mod = ((HOME.mod % HERO_SHOTS.length) + HERO_SHOTS.length) % HERO_SHOTS.length;
      var shot = HERO_SHOTS[HOME.mod];
      img.src = 'assets/hero/' + shot.file;
      img.style.opacity = '';
      aspectStyle($('heroViewer'), shot.w, shot.h);
      $('heroSvg').innerHTML = '';
      $('heroTL').textContent = shot.label || '';
      $('heroBadge').textContent = shot.group === 'ir' ? 'LWIR 8-bit' : 'RGB 8-bit';
      $('heroCap').textContent = shot.w + '×' + shot.h +
        ' · 顶层真实背景照片 + 同一套内核渲染（out/demo_*）';
      var chips1 = $('heroMods');
      chips1.innerHTML = '';
      HERO_SHOTS.forEach(function (s, i) {
        var b = el('button', i === HOME.mod ? 'on' : '');
        b.type = 'button';
        b.appendChild(el('span', 'dotmod'));
        b.appendChild(document.createTextNode((s.label || '').replace(' · ', ' ')));
        b.addEventListener('click', function () { HOME.mod = i; heroRender(); heroStop(); });
        chips1.appendChild(b);
      });
      return;
    }
    // --- case 2: fall back to the manifest frames ---
    var fr = sc.frames[Math.min(HOME.frame, sc.frames.length - 1)];
    var list = WALL_MODS.filter(function (k) { return (fr.mods || {})[k]; });
    if (!list.length) return;
    HOME.mod = ((HOME.mod % list.length) + list.length) % list.length;
    var k = list[HOME.mod], e = fr.mods[k];
    img.src = imgURL(e);
    aspectStyle($('heroViewer'), e.w, e.h);
    drawBoxes($('heroSvg'), e.boxes, e.w, e.h, {});
    $('heroTL').textContent = sc.title;
    $('heroBadge').textContent = modLabel(k);
    $('heroCap').textContent = '帧 ' + (fr.i + 1) + ' / ' + sc.frames.length + ' · ' +
      weatherText(fr) + ' · ' + (e.w + '×' + e.h);
    var chips = $('heroMods');
    chips.innerHTML = '';
    list.forEach(function (key, i) {
      var b = el('button', i === HOME.mod ? 'on' : '');
      b.type = 'button';
      b.appendChild(el('span', 'dotmod'));
      b.appendChild(document.createTextNode(modLabel(key)));
      b.addEventListener('click', function () { HOME.mod = i; heroRender(); heroStop(); });
      chips.appendChild(b);
    });
  }
  function heroPlay() {
    heroStop();
    HOME.timer = setInterval(function () {
      if (document.hidden || CURRENT !== 'home') return;
      HOME.mod++;
      heroRender();
    }, 5200);
  }
  function heroStop() { if (HOME.timer) { clearInterval(HOME.timer); HOME.timer = null; } }

  // Data panels (radar, SAR, lidar strips) are statistical images with a huge
  // dynamic range; at their native scale they read as black rectangles next to
  // a photographic RGB panel. Draw them through the same display stretch the
  // radar tab uses, into a canvas laid exactly where object-fit would put them.
  var STRETCH_KEYS = { radar_ra: 1, sar: 1, lidar_range: 1, lidar_intensity: 1, radar: 1 };
  function paintStretched(box, img) {
    var cv = document.createElement('canvas');
    cv.className = 'ovl-canvas';
    box.appendChild(cv);
    img.style.visibility = 'hidden';
    var draw = function () {
      if (!img.naturalWidth) return;
      var r = box.getBoundingClientRect();
      if (!r.width) return;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      cv.width = Math.round(r.width * dpr);
      cv.height = Math.round(r.height * dpr);
      var off;
      try { off = stretchInto(img); } catch (e) { img.style.visibility = ''; cv.remove(); return; }
      var g = cv.getContext('2d');
      g.clearRect(0, 0, cv.width, cv.height);
      var s = Math.min(cv.width / img.naturalWidth, cv.height / img.naturalHeight);
      g.drawImage(off, (cv.width - img.naturalWidth * s) / 2,
        (cv.height - img.naturalHeight * s) / 2,
        img.naturalWidth * s, img.naturalHeight * s);
    };
    if (img.complete && img.naturalWidth) draw(); else img.addEventListener('load', draw);
  }

  function wallRender() {
    var host = $('mwGrid');
    if (!host) return;
    var sc = MAN.scenes[HOME.scene];
    var fr = sc.frames[Math.min(HOME.wallFrame, sc.frames.length - 1)];
    var list = wallList(fr);
    host.innerHTML = '';
    list.forEach(function (k) {
      var e = fr.mods[k];
      var card = el('div', 'wcard' + (e.w / e.h > 1.6 ? ' wide' : ''));
      var box = el('div', 'wimg');
      var im = el('img');
      im.src = thumbURL(e);
      im.alt = sc.title + ' ' + modLabel(k);
      im.loading = 'lazy';
      im.decoding = 'async';
      box.appendChild(im);
      if (STRETCH_KEYS[k]) paintStretched(box, im);
      card.appendChild(box);
      var meta = el('div', 'wmeta');
      meta.appendChild(el('b', '', modLabel(k)));
      meta.appendChild(el('span', '', modNote(sc, fr, k)));
      card.appendChild(meta);
      card.addEventListener('click', function () {
        S.scene = HOME.scene; S.frame = fr.i;
        if (fr.mods[k]) S.mod = k;
        showTab('gallery');
        setTimeout(galSync, 40);
      });
      host.appendChild(card);
    });
    $('mwCount').textContent = '帧 ' + (fr.i + 1) + ' / ' + sc.frames.length +
      ' · ' + (fr.weather ? (WNAME[fr.weather.name] || '') : '');
  }

  function homeSync() {
    if (!MAN) return;
    if (!HOME.init) {
      HOME.init = true;
      HOME.scene = showcaseScene();
      HOME.frame = 0;
      buildHeroStats();
      $('mwPrev').addEventListener('click', function () {
        var n = MAN.scenes[HOME.scene].frames.length;
        HOME.wallFrame = (HOME.wallFrame - 1 + n) % n;
        HOME.frame = HOME.wallFrame;
        wallRender(); heroRender();
      });
      $('mwNext').addEventListener('click', function () {
        var n = MAN.scenes[HOME.scene].frames.length;
        HOME.wallFrame = (HOME.wallFrame + 1) % n;
        HOME.frame = HOME.wallFrame;
        wallRender(); heroRender();
      });
      var hv = $('heroViewer');
      if (hv) {
        hv.addEventListener('pointerenter', heroStop);
        hv.addEventListener('pointerleave', function () { if (CURRENT === 'home') heroPlay(); });
      }
    }
    // pick the frame with the most modalities, then the middle of the clip
    var sc = MAN.scenes[HOME.scene];
    var bestI = 0, bestN = -1;
    sc.frames.forEach(function (fr, i) {
      var n = Object.keys(fr.mods || {}).length;
      if (n > bestN) { bestN = n; bestI = i; }
    });
    if (!HOME.started) {
      HOME.started = true;
      HOME.frame = Math.min(bestI, sc.frames.length - 1);
      HOME.wallFrame = HOME.frame;
      HOME.mod = 0;
      heroRender();
      wallRender();
      heroPlay();
    }
  }

  // ---------- figures ----------
  var FIGS = [], FIG_TAG = '全部', FIG_AT = 0;

  function figURL(f) { return 'assets/figures/' + f.file; }
  // Cards are ~420 px wide, the lightbox is not: ship both sizes.
  function figThumb(f) { return 'assets/figures/' + (f.thumb || f.file); }

  function renderFigPreview() {
    var host = $('figPreview');
    if (!host) return;
    host.innerHTML = '';
    FIGS.slice(0, 3).forEach(function (f) {
      var a = el('a', 'figcard');
      a.href = '#/figures';
      var im = el('img');
      im.src = figThumb(f);
      im.alt = f.title || '';
      im.loading = 'lazy';
      im.decoding = 'async';
      a.appendChild(im);
      var c = el('div', 'fc');
      c.appendChild(el('b', '', f.title || f.file));
      c.appendChild(el('span', '', f.subtitle || ''));
      a.appendChild(c);
      a.addEventListener('click', function (ev) {
        ev.preventDefault();
        showTab('figures');
        setTimeout(function () { figOpen(FIGS.indexOf(f)); }, 60);
      });
      host.appendChild(a);
    });
  }

  function renderFigs() {
    var grid = $('figGrid');
    if (!grid) return;
    var tags = ['全部'];
    FIGS.forEach(function (f) {
      if (f.tag && tags.indexOf(f.tag) < 0) tags.push(f.tag);
    });
    seg($('figTags'), tags.map(function (t, i) { return { id: String(i), label: t }; }),
      String(tags.indexOf(FIG_TAG)),
      function (id) { FIG_TAG = tags[parseInt(id, 10)] || '全部'; renderFigs(); });

    var list = FIGS.filter(function (f) { return FIG_TAG === '全部' || f.tag === FIG_TAG; });
    $('figCount').textContent = list.length + ' / ' + FIGS.length + ' 张';
    grid.innerHTML = '';
    list.forEach(function (f) {
      var item = el('div', 'fgitem');
      item.tabIndex = 0;
      var box = el('div', 'fgi');
      var im = el('img');
      im.src = figThumb(f);
      im.alt = f.title || '';
      im.loading = 'lazy';
      im.decoding = 'async';
      box.appendChild(im);
      box.appendChild(el('div', 'fgn', '图 ' + (f.id || '')));
      item.appendChild(box);
      var t = el('div', 'fgt');
      t.appendChild(el('b', '', f.title || f.file));
      t.appendChild(el('span', '', f.desc || f.subtitle || ''));
      item.appendChild(t);
      item.addEventListener('click', function () { figOpen(FIGS.indexOf(f)); });
      item.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') figOpen(FIGS.indexOf(f));
      });
      grid.appendChild(item);
    });
  }

  function figOpen(i) {
    if (!FIGS.length) return;
    FIG_AT = (i + FIGS.length) % FIGS.length;
    var f = FIGS[FIG_AT];
    $('figBig').src = figURL(f);
    $('figBig').alt = f.title || '';
    $('figTitle').textContent = '图 ' + (f.id || '') + ' · ' + (f.title || f.file);
    $('figSub').textContent = f.subtitle || '';
    $('figDesc').textContent = f.desc || '';
    $('figSource').textContent = f.source ? '数据来源 ' + f.source : '';
    $('figBox').hidden = false;
    document.body.classList.add('locked');
  }
  function figClose() {
    $('figBox').hidden = true;
    document.body.classList.remove('locked');
  }
  function figStep(d) { figOpen(FIG_AT + d); }

  function initFigures() {
    ['figClose'].forEach(function (id) {
      var b = $(id);
      if (b) b.addEventListener('click', figClose);
    });
    $('figPrev').addEventListener('click', function () { figStep(-1); });
    $('figNext').addEventListener('click', function () { figStep(1); });
    $('figBox').addEventListener('click', function (ev) {
      if (ev.target === $('figBox')) figClose();
    });
    document.addEventListener('keydown', function (ev) {
      if ($('figBox').hidden) return;
      if (ev.key === 'Escape') figClose();
      if (ev.key === 'ArrowLeft') figStep(-1);
      if (ev.key === 'ArrowRight') figStep(1);
    });
    fetch('assets/figures/index.json', { cache: 'no-cache' })
      .then(function (r) { if (!r.ok) throw new Error('figures ' + r.status); return r.json(); })
      .then(function (list) {
        FIGS = (list || []).filter(function (f) { return f && f.file; });
        if (!FIGS.length) throw new Error('empty');
        renderFigPreview();
        renderFigs();
      })
      .catch(function () {
        var g = $('figGrid');
        if (g) g.innerHTML = '<div class="card empty-card"><h3>图鉴还没有导出</h3>' +
          '<p>先在仓库里运行 <code>env\\Scripts\\python.exe tools\\make_figures.py</code> ' +
          '生成 assets/figures/；用 file:// 直接打开页面时浏览器也不允许读取 index.json。</p></div>';
        var p = $('figPreview');
        if (p) p.parentNode.style.display = 'none';
      });
  }

  // ---------- boot ----------
  function boot() {
    var have = {};
    MAN.scenes.forEach(function (s) {
      (s.mods || []).forEach(function (m) {
        if (m && m.key && m.title) MOD_TITLE[m.key] = m.title;
      });
      s.frames.forEach(function (f) {
        Object.keys(f.mods || {}).forEach(function (k) { have[k] = 1; });
      });
    });
    MODS = MODS_ALL.filter(function (m) { return have[m]; });
    if (MODS.indexOf(S.mod) < 0) S.mod = MODS[0];

    // Footer is assembled from the manifest itself, never from constants: a
    // dataset that only exports three scenes or two bands should say so.
    var nFrame = MAN.scenes.reduce(function (a, s) { return a + s.frames.length; }, 0);
    var g = genLabel(MAN.generated);
    $('genInfo').textContent =
      (g ? g + ' · ' : '') + MAN.scenes.length + ' 场景 · ' + nFrame + ' 帧 · ' +
      MODS.length + ' 种模态';

    if (MAN_URL) {
      var b = $('devBadge');
      b.hidden = false;
      if (FELL_BACK) {
        b.textContent = '回落 · 内置数据';
        b.title = '按 URL 参数加载失败，已回落到 data/boot.js';
      } else {
        b.textContent = 'MOCK · ' + MAN_URL.split('/').pop();
        b.title = '开发预览：manifest 来自 ' + MAN_URL;
      }
    }

    // Only the home tab is painted at boot. Every other tab syncs on first
    // show (see showTab), which keeps the first paint from downloading a
    // gallery frame, a radar map and two seeker views nobody asked for.
    homeSync();
    if (BANDS) exploreUpdate();
  }

  document.addEventListener('DOMContentLoaded', function () {
    // open whatever the URL asked for before any data arrives, so a deep link
    // never flashes the home tab first
    showTab(hashTab(), { syncHash: false, instant: true });
    initHeroShots();
    initFigures();
    document.addEventListener('keydown', function (ev) {
      var tag = (ev.target && ev.target.tagName) || '';
      if (/INPUT|TEXTAREA|SELECT/.test(tag)) return;
      if (CURRENT === 'gallery') {
        if (ev.key === 'ArrowLeft') galStep(-1);
        if (ev.key === 'ArrowRight') galStep(1);
      } else if (CURRENT === 'seeker') {
        if (ev.key === 'ArrowLeft') skStep(-1);
        if (ev.key === 'ArrowRight') skStep(1);
      }
    });
    if ($('galLazy')) $('galLazy').addEventListener('change', function () { if (MAN) galSync(); });
    if ($('galPrev')) $('galPrev').addEventListener('click', function () { galStep(-1); });
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

    // SAR controls
    $('sarPrev').addEventListener('click', function () {
      var n = curScene().frames.length;
      S.frame = (S.frame - 1 + n) % n; sarSync();
    });
    $('sarNext').addEventListener('click', function () {
      var n = curScene().frames.length;
      S.frame = (S.frame + 1) % n; sarSync();
    });
    $('sarRange').addEventListener('input', function () {
      S.frame = parseInt(this.value, 10); sarSync();
    });
    $('sarPlay').addEventListener('click', function () {
      if (sarTimer) { sarStop(); return; }
      $('sarPlay').textContent = '暂停';
      sarTimer = setInterval(function () {
        var n = curScene().frames.length;
        S.frame = (S.frame + 1) % n; sarSync();
      }, 800);
    });

    // seeker controls
    $('skPrev').addEventListener('click', function () { skStep(-1); });
    $('skNext').addEventListener('click', function () { skStep(1); });
    $('skRange').addEventListener('input', function () {
      S.frame = parseInt(this.value, 10); skSync();
    });
    $('skPlay').addEventListener('click', skPlay);

    window.addEventListener('resize', function () {
      resizePC();
      drawRadar();
      drawRdMark(curFrame());
      if (BANDS) exploreUpdate();
      sarLayout();
      skMag('seeker');
      skMag('wide');
    });
    $('rdImg').addEventListener('load', function () { drawRdMark(curFrame()); drawRdStretch(); });
    if ($('rdStretch')) $('rdStretch').addEventListener('change', drawRdStretch);
    var sarI = $('sarImg');
    if (sarI) sarI.addEventListener('load', function () { sarLayout(); });

    // data/boot.js is a pre-baked copy of the same two JSON files. It is what
    // makes the page work from file:// (no server, no fetch), which is the
    // air-gapped case this project actually targets. When it is present we use
    // it; otherwise we fall back to fetching the JSON, which is what a normal
    // static host serves. ?manifest= forces the fetch path (dev only).
    function fromBoot() {
      if (MAN_URL) return null;
      var d = window.SIMKIT_DATA;
      if (!d || !d.manifest || !d.bands) return null;
      return Promise.resolve([d.manifest, d.bands]);
    }
    var loadData = fromBoot() || Promise.all([
      fetch(MAN_URL || 'data/manifest.json', { cache: 'no-cache' })
        .then(function (r) { if (!r.ok) throw new Error('manifest ' + r.status); return r.json(); }),
      fetch('data/radar_bands.json', { cache: 'no-cache' })
        .then(function (r) { if (!r.ok) throw new Error('bands ' + r.status); return r.json(); })
    ]);
    loadData.then(function (res) {
      MAN = res[0]; BANDS = res[1];
      boot();
      exploreUpdate();
      // The URL decides which tab is open; boot() already warmed every view, so
      // this only re-runs the side effects that depend on layout (canvas sizes).
      loadCloudIndex();
      showTab(hashTab(), { syncHash: false, instant: true, keepScroll: true });
    })
      .catch(function (e) {
        // A bad ?manifest= (typo, file moved, server down) used to leave a blank
        // page even though data/boot.js was sitting right there. Rescue it.
        var d = window.SIMKIT_DATA;
        if (MAN_URL && d && d.manifest && d.bands) {
          FELL_BACK = true;
          MAN = d.manifest; BANDS = d.bands; boot(); exploreUpdate();
          notice('按 URL 参数加载失败，已回落到内置数据');
          return;
        }
        document.body.insertAdjacentHTML('beforeend',
          '<div style="position:fixed;bottom:16px;left:16px;background:#ff5d6c;color:#000;' +
          'padding:10px 16px;border-radius:8px;font:13px monospace;z-index:99;max-width:70vw">' +
          '数据未加载：请先运行 tools/export_web.py 生成 data/boot.js<br>' +
          String(e) + '</div>');
      });
  });

})();
