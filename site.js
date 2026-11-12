/* site.js — 2026-11-12 showcase.
 *
 * Data: data/showcase.js (window.SIMKIT_SHOWCASE) with data/showcase.json as
 * the fetch fallback, both written by tools/export_showcase.py. Every number on
 * the page is read from that file; nothing here is typed in by hand.
 *
 * Missing data degrades to the static markup in index.html: the hero frame,
 * the RD panel and the captions already ship in HTML.
 */
(function () {
  'use strict';

  var S = null;
  var st = { frame: 12, chan: 'optical', playing: false, temp: 300 };
  var timer = null;
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var SVGNS = 'http://www.w3.org/2000/svg';

  function $(id) { return document.getElementById(id); }
  function pad6(n) { var s = String(n); while (s.length < 6) { s = '0' + s; } return s; }
  function n1(v) { return (Math.round(Number(v) * 10) / 10).toFixed(1); }
  function n2(v) { return (Math.round(Number(v) * 100) / 100).toFixed(2); }
  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function el(tag, attrs, text) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) { for (var k in attrs) { if (attrs.hasOwnProperty(k)) { n.setAttribute(k, attrs[k]); } } }
    if (text != null) { n.textContent = text; }
    return n;
  }
  function clear(node) { while (node && node.firstChild) { node.removeChild(node.firstChild); } }

  /* ------------------------------------------------------------- frame data */

  function seriesOf(which) {
    return which === 'jam' ? S.jam.series : S.run.series;
  }
  function frameAt(which, i) {
    var list = seriesOf(which);
    return list[clamp(i, 0, list.length - 1)];
  }
  function closingAt(i) {
    var list = S.run.series;
    var a = list[clamp(i - 3, 0, list.length - 1)];
    var b = list[clamp(i + 3, 0, list.length - 1)];
    var dt = b.t - a.t;
    if (!dt) { return 0; }
    return Math.abs((b.range_m - a.range_m) / dt);
  }
  function imgSrc(runId, mod, i) {
    return 'data/run/' + runId + '/' + mod + '/' + pad6(i) + '.webp';
  }

  /* ------------------------------------------------------------------ render */

  function render(i) {
    if (!S) { return; }
    var fr = frameAt('run', i);
    st.frame = fr.i;

    var opt = $('optImg');
    var rd = $('rdImg');
    if (opt) { opt.src = imgSrc(S.run.id, st.chan, fr.i); }
    if (rd) { rd.src = imgSrc(S.run.id, 'rd', fr.i); }

    var clock = $('optClock');
    if (clock) { clock.textContent = '帧 ' + fr.i + ' / ' + S.run.frames; }

    var t = $('tFrame'); if (t) { t.textContent = fr.i; }
    t = $('tTime'); if (t) { t.textContent = n2(fr.t) + ' s'; }
    t = $('tRange'); if (t) { t.textContent = n1(fr.range_m); }
    t = $('tClose'); if (t) { t.textContent = n1(closingAt(fr.i)); }
    t = $('tPx'); if (t) { t.textContent = n1(fr.px); }
    t = $('tSnr'); if (t) { t.textContent = n1(fr.snr); }
    t = $('tNpts'); if (t) { t.textContent = fr.npts; }

    var truth = (fr.targets && fr.targets[0]) || null;
    if (truth) {
      t = $('tTrueR'); if (t) { t.textContent = n1(truth.r) + ' m'; }
      t = $('tTrueV'); if (t) { t.textContent = (truth.v >= 0 ? '+' : '') + n1(truth.v) + ' m/s'; }
    }
    if (fr.peak) {
      t = $('tPeakR'); if (t) { t.textContent = n1(fr.peak.r) + ' m'; }
      t = $('tPeakV'); if (t) { t.textContent = (fr.peak.v >= 0 ? '+' : '') + n1(fr.peak.v) + ' m/s'; }
      if (truth && $('dNote')) {
        var dr = Math.abs(fr.peak.r - truth.r);
        $('dNote').textContent = Math.round(dr) + ' m';
      }
      if ($('peakNote')) { $('peakNote').textContent = peakNote(fr); }
    }

    var count = $('count'); if (count) { count.textContent = (fr.i + 1) + ' / ' + S.run.frames; }
    var scrub = $('scrub'); if (scrub && Number(scrub.value) !== fr.i) { scrub.value = String(fr.i); }

    drawOptFrame(fr);
    drawRadar($('rdOv'), fr, S.run.rd, true);
    markStrip(fr.i);
  }

  // The brightest cell is often a decoy, not the hull. Say which entity sits on
  // it instead of calling the offset a calibration error.
  function peakNote(fr) {
    if (!fr.peak || !S || !S.run.ecm) { return ''; }
    var best = null, bestD = 1e9;
    (S.run.ecm.entities || []).forEach(function (e) {
      var at = e.at || {};
      if (at.r == null) { return; }
      var d = Math.abs(at.r - fr.peak.r);
      if (d < bestD && d < 40) { bestD = d; best = e; }
    });
    if (!best) { return '峰值落在噪声里，没有实体与之对应。'; }
    var kind = KIND[best.kind] || best.kind;
    var rcs = best.at && best.at.rcs ? '（本帧 σ ' + n1(best.at.rcs) + ' m²）' : '';
    return '最强单元落在 ' + Math.round(best.at.r) + ' m 的' + kind + rcs + '上，比 28 m 船体亮；';
  }

  var marked = -1;
  function markStrip(i) {
    if (marked === i) { return; }
    var list = $('strip');
    if (!list) { return; }
    var btns = list.querySelectorAll('button');
    for (var k = 0; k < btns.length; k++) { btns[k].setAttribute('aria-current', 'false'); }
    if (btns[i]) {
      btns[i].setAttribute('aria-current', 'true');
      var strip = list, left = btns[i].offsetLeft - strip.clientWidth / 2 + btns[i].clientWidth / 2;
      if (strip.scrollTo) { strip.scrollTo({ left: left, behavior: reduced ? 'auto' : 'smooth' }); }
    }
    marked = i;
  }

  /* Optical overlay: corner brackets around the COCO box, plus a scale tick. */
  function drawOptFrame(fr) {
    var svg = $('optOv');
    if (!svg || !fr.box) { return; }
    clear(svg);
    var W = 768, H = 576;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var x = fr.box[0], y = fr.box[1], w = fr.box[2], h = fr.box[3];
    var pad = 7, c = Math.max(6, Math.min(12, w * 0.45));
    var x0 = x - pad, y0 = y - pad, x1 = x + w + pad, y1 = y + h + pad;
    var pts = [
      [[x0, y0 + c], [x0, y0], [x0 + c, y0]],
      [[x1 - c, y0], [x1, y0], [x1, y0 + c]],
      [[x1, y1 - c], [x1, y1], [x1 - c, y1]],
      [[x0 + c, y1], [x0, y1], [x0, y1 - c]]
    ];
    for (var i = 0; i < pts.length; i++) {
      var d = 'M' + pts[i][0][0] + ',' + pts[i][0][1] +
        'L' + pts[i][1][0] + ',' + pts[i][1][1] +
        'L' + pts[i][2][0] + ',' + pts[i][2][1];
      svg.appendChild(el('path', { d: d, fill: 'none', stroke: '#4cd7ff', 'stroke-width': '1.4', opacity: '.95' }));
    }
    var label = el('text', {
      x: x0, y: y0 - 7, fill: '#d6f2ff', 'font-size': '11',
      'font-family': 'ui-monospace, Consolas, monospace', 'letter-spacing': '.06em'
    }, 'ship_01 · ' + Math.round(w) + '×' + Math.round(h) + ' px');
    label.setAttribute('paint-order', 'stroke');
    label.setAttribute('stroke', '#04070b');
    label.setAttribute('stroke-width', '3');
    svg.appendChild(label);
  }

  /* Radar overlay: axes, zero-velocity line, truth diamond, peak cross. */
  function drawRadar(svg, fr, dims, full) {
    if (!svg || !dims) { return; }
    clear(svg);
    var W = dims.w, H = dims.h;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var rmax = dims.r_max_m, vnyq = dims.v_nyq_mps;
    var xOf = function (r) { return r / rmax * W; };
    var yOf = function (v) { return (1 - (v + vnyq) / (2 * vnyq)) * H; };
    var mono = 'ui-monospace, Consolas, monospace';

    // zero radial velocity: the static-clutter ridge
    var yz = yOf(0);
    svg.appendChild(el('line', {
      x1: 0, y1: yz, x2: W, y2: yz, stroke: '#8fa6b8',
      'stroke-width': '1', 'stroke-dasharray': '2 6', opacity: '.35'
    }));

    var compact = window.matchMedia && window.matchMedia('(max-width: 720px)').matches;
    if (full && !compact) {
      var rt = [0, 1000, 2000, 3000, 4000, 5000, 6000];
      for (var i = 0; i < rt.length; i++) {
        var x = xOf(rt[i]);
        // keep clear of the colour bar's labels in the right-hand corner
        if (x > W - 70) { continue; }
        svg.appendChild(el('line', { x1: x, y1: H - 7, x2: x, y2: H - 1, stroke: '#7d93a6', 'stroke-width': '1', opacity: '.55' }));
        var xl = el('text', {
          x: x + 3, y: H - 10, fill: '#c3d6e4', 'font-size': '10', 'font-family': mono
        }, rt[i] === 0 ? '0' : String(rt[i] / 1000) + ' km');
        xl.setAttribute('paint-order', 'stroke');
        xl.setAttribute('stroke', '#04070b');
        xl.setAttribute('stroke-width', '3');
        svg.appendChild(xl);
      }
      var vt = [-150, -100, -50, 50, 100, 150];
      for (var j = 0; j < vt.length; j++) {
        var y = yOf(vt[j]);
        if (y < 10 || y > H - 10) { continue; }
        svg.appendChild(el('line', { x1: 0, y1: y, x2: 6, y2: y, stroke: '#7d93a6', 'stroke-width': '1', opacity: '.55' }));
        var yl = el('text', {
          x: 9, y: y + 3.5, fill: '#c3d6e4', 'font-size': '10', 'font-family': mono
        }, (vt[j] > 0 ? '+' : '') + vt[j]);
        yl.setAttribute('paint-order', 'stroke');
        yl.setAttribute('stroke', '#04070b');
        yl.setAttribute('stroke-width', '3');
        svg.appendChild(yl);
      }
      if (!compact) {
        svg.appendChild(el('text', {
          x: W - 35, y: yz - 7, fill: '#9fb2c2', 'font-size': '9.5', 'font-family': mono,
          'letter-spacing': '.08em'
        }, '零速'));
      }
    }

    if (full && fr.db_hi != null) {
      var defs = el('defs');
      var grad = el('linearGradient', { id: 'rdRamp', x1: '0', y1: '1', x2: '0', y2: '0' });
      [['0', '#2c1250'], ['8', '#4145ab'], ['17', '#4675ed'], ['25', '#3e9bfe'],
       ['33', '#21b9f2'], ['42', '#11d0d0'], ['50', '#22e19f'], ['58', '#52eb69'],
       ['67', '#94ed3d'], ['75', '#d6df2a'], ['83', '#fbb62f'], ['92', '#f8771f'],
       ['100', '#b40426']].forEach(function (s) {
        grad.appendChild(el('stop', { offset: s[0] + '%', 'stop-color': s[1] }));
      });
      defs.appendChild(grad);
      svg.appendChild(defs);
      var cw = 9, cx = W - 16, cy0 = 44, cy1 = H - 16;
      svg.appendChild(el('rect', {
        x: cx, y: cy0, width: cw, height: cy1 - cy0, fill: 'url(#rdRamp)',
        stroke: 'rgba(0,0,0,.55)', 'stroke-width': '1'
      }));
      svg.appendChild(el('line', {
        x1: cx - 60, y1: yz, x2: cx, y2: yz, stroke: 'rgba(255,255,255,.10)', 'stroke-width': '1'
      }));
      function dbLabel(y, txt, anchor) {
        var t = el('text', {
          x: cx - 6, y: y + 3, fill: '#b9c6d4', 'font-size': '9.5', 'font-family': mono,
          'text-anchor': 'end'
        }, txt);
        t.setAttribute('paint-order', 'stroke');
        t.setAttribute('stroke', '#04070b');
        t.setAttribute('stroke-width', '3');
        svg.appendChild(t);
      }
      dbLabel(cy0 + 4, Math.round(fr.db_hi) + ' dBW');
      dbLabel(cy1 - 2, String(Math.round(fr.db_lo)));
    }

    var truth = (fr.targets && fr.targets[0]) || null;
    if (truth) {
      var tx = xOf(truth.r), ty = yOf(truth.v);
      if (tx > 0 && tx < W && ty > 0 && ty < H) {
        svg.appendChild(el('path', {
          d: 'M' + tx + ',' + (ty - 7) + 'L' + (tx + 7) + ',' + ty + 'L' + tx + ',' + (ty + 7) + 'L' + (tx - 7) + ',' + ty + 'Z',
          fill: 'none', stroke: '#4cd7ff', 'stroke-width': '1.6'
        }));
        if (full) {
          var lt = el('text', {
            x: tx + 11, y: ty - 8, fill: '#bff0ff', 'font-size': '10',
            'font-family': mono, 'letter-spacing': '.06em'
          }, '真值');
          lt.setAttribute('paint-order', 'stroke');
          lt.setAttribute('stroke', '#04070b');
          lt.setAttribute('stroke-width', '3');
          svg.appendChild(lt);
        }
      }
    }
    if (full && fr.peak) {
      var px = xOf(fr.peak.r), py = yOf(fr.peak.v);
      if (px > 0 && px < W && py > 0 && py < H) {
        svg.appendChild(el('path', {
          d: 'M' + (px - 6) + ',' + py + 'h12M' + px + ',' + (py - 6) + 'v12',
          stroke: '#ff9a4d', 'stroke-width': '1.4', fill: 'none'
        }));
        var lp = el('text', {
          x: px + 9, y: py + 15, fill: '#ffd2ab', 'font-size': '10',
          'font-family': mono, 'letter-spacing': '.06em'
        }, '峰值');
        lp.setAttribute('paint-order', 'stroke');
        lp.setAttribute('stroke', '#04070b');
        lp.setAttribute('stroke-width', '3');
        svg.appendChild(lp);
      }
    }
  }

  /* --------------------------------------------------------------- timeline */

  function buildStrip() {
    var ul = $('strip');
    if (!ul || !S) { return; }
    var frag = document.createDocumentFragment();
    for (var i = 0; i < S.run.frames; i++) {
      var li = document.createElement('li');
      var b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('data-i', String(i));
      b.setAttribute('aria-label', '第 ' + i + ' 帧');
      var im = document.createElement('img');
      im.src = 'data/run/' + S.run.id + '/strip/' + pad6(i) + '.webp';
      im.width = 70; im.height = 52; im.loading = 'lazy';
      im.alt = '第 ' + i + ' 帧缩略图';
      b.appendChild(im);
      li.appendChild(b);
      frag.appendChild(li);
    }
    ul.appendChild(frag);
    ul.addEventListener('click', function (ev) {
      var b = ev.target.closest('button');
      if (!b) { return; }
      stop();
      render(parseInt(b.getAttribute('data-i'), 10));
    });
  }

  function play() {
    if (st.playing) { return; }
    st.playing = true;
    var btn = $('play');
    if (btn) {
      btn.setAttribute('data-on', '1');
      btn.setAttribute('aria-label', '暂停');
      btn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" fill="currentColor"/></svg>';
    }
    timer = window.setInterval(function () {
      render((st.frame + 1) % S.run.frames);
    }, 100);
  }
  function stop() {
    if (!st.playing) { return; }
    st.playing = false;
    window.clearInterval(timer);
    timer = null;
    var btn = $('play');
    if (btn) {
      btn.setAttribute('data-on', '0');
      btn.setAttribute('aria-label', '播放');
      btn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5l11 7-11 7z" fill="currentColor"/></svg>';
    }
  }

  /* ---------------------------------------------------------------- planck */

  var C2 = 14388; // hc/k in um*K

  function planck(lambdaUm, T) {
    var x = C2 / (lambdaUm * T);
    if (x > 700) { return 0; }
    return 1 / (Math.pow(lambdaUm, 5) * (Math.exp(x) - 1));
  }
  function curve(T, n) {
    var out = [], max = 0, i, l, v;
    for (i = 0; i < n; i++) {
      l = 2 + (16 - 2) * i / (n - 1);
      v = planck(l, T);
      if (v > max) { max = v; }
      out.push([l, v]);
    }
    for (i = 0; i < out.length; i++) { out[i][1] = out[i][1] / max; }
    return out;
  }
  function integral(a, b, T, n) {
    var s = 0, i, l, h = (b - a) / n;
    for (i = 0; i < n; i++) {
      l = a + h * (i + 0.5);
      s += planck(l, T);
    }
    return s * h;
  }

  var PL = { w: 640, h: 300, x0: 54, x1: 604, y0: 24, y1: 250, lo: 2, hi: 16 };

  function buildPlanck() {
    var svg = $('planckSvg');
    if (!svg) { return; }
    clear(svg);
    var mono = 'ui-monospace, Consolas, monospace';
    var g = el('g');
    svg.appendChild(g);

    // atmospheric windows
    function win(a, b, fill, label, color) {
      var xa = PL.x0 + (a - PL.lo) / (PL.hi - PL.lo) * (PL.x1 - PL.x0);
      var xb = PL.x0 + (b - PL.lo) / (PL.hi - PL.lo) * (PL.x1 - PL.x0);
      g.appendChild(el('rect', { x: xa, y: PL.y0, width: xb - xa, height: PL.y1 - PL.y0, fill: fill }));
      g.appendChild(el('text', {
        x: (xa + xb) / 2, y: PL.y1 - 8, fill: color, 'font-size': '10.5',
        'font-family': mono, 'text-anchor': 'middle', 'letter-spacing': '.04em'
      }, label));
    }
    win(3, 5, 'rgba(255,154,77,.07)', '3–5 µm · MWIR', '#e0a675');
    win(8, 12, 'rgba(76,215,255,.06)', '8–12 µm · LWIR', '#7fc6e0');

    // frame
    g.appendChild(el('path', {
      d: 'M' + PL.x0 + ',' + PL.y0 + 'V' + PL.y1 + 'H' + PL.x1,
      fill: 'none', stroke: '#2a3846', 'stroke-width': '1'
    }));

    // x ticks
    for (var l = 2; l <= 16; l += 2) {
      var x = PL.x0 + (l - PL.lo) / (PL.hi - PL.lo) * (PL.x1 - PL.x0);
      g.appendChild(el('line', { x1: x, y1: PL.y1, x2: x, y2: PL.y1 + 5, stroke: '#2a3846' }));
      g.appendChild(el('text', {
        x: x, y: PL.y1 + 18, fill: '#8494a6', 'font-size': '10', 'font-family': mono, 'text-anchor': 'middle'
      }, String(l)));
    }
    g.appendChild(el('text', {
      x: PL.x1, y: PL.y1 + 34, fill: '#5d6b7c', 'font-size': '10', 'font-family': mono, 'text-anchor': 'end'
    }, '波长 / µm'));
    g.appendChild(el('text', {
      x: PL.x0, y: PL.y0 - 8, fill: '#5d6b7c', 'font-size': '10', 'font-family': mono
    }, '归一化辐亮度'));

    return { g: g, mono: mono };
  }

  function renderPlanck(T) {
    st.temp = T;
    var svg = $('planckSvg');
    if (!svg) { return; }
    var mono = 'ui-monospace, Consolas, monospace';
    var old = svg.querySelector('#pkCurves');
    if (old) { svg.removeChild(old); }
    var g = el('g', { id: 'pkCurves' });
    svg.appendChild(g);

    function path(list, stroke, width, dash, opacity) {
      var d = '', i;
      for (i = 0; i < list.length; i++) {
        var x = PL.x0 + (list[i][0] - PL.lo) / (PL.hi - PL.lo) * (PL.x1 - PL.x0);
        var y = PL.y1 - list[i][1] * (PL.y1 - PL.y0);
        d += (i ? 'L' : 'M') + x.toFixed(2) + ',' + y.toFixed(2);
      }
      var p = el('path', { d: d, fill: 'none', stroke: stroke, 'stroke-width': width, opacity: opacity || 1 });
      if (dash) { p.setAttribute('stroke-dasharray', dash); }
      g.appendChild(p);
    }

    path(curve(800, 240), '#ff9a4d', 1.2, '0', '.55');
    path(curve(300, 240), '#4cd7ff', 1.2, '0', '.5');
    path(curve(T, 240), '#e8eff7', 2, '0', '1');

    // Wien peak of the live curve
    var lmax = 2898 / T;
    if (lmax > PL.lo && lmax < PL.hi) {
      var xm = PL.x0 + (lmax - PL.lo) / (PL.hi - PL.lo) * (PL.x1 - PL.x0);
      g.appendChild(el('line', {
        x1: xm, y1: PL.y0, x2: xm, y2: PL.y1, stroke: '#ff9a4d',
        'stroke-width': '1', 'stroke-dasharray': '3 4', opacity: '.65'
      }));
      g.appendChild(el('text', {
        x: xm + 5, y: PL.y0 + 30, fill: '#ffc79a', 'font-size': '10', 'font-family': mono
      }, 'λmax ' + n2(lmax) + ' µm'));
    }

    var tot = integral(1, 40, T, 1600);
    var fm = integral(3, 5, T, 800) / tot;
    var fl = integral(8, 12, T, 1200) / tot;

    var out = $('tempOut'); if (out) { out.textContent = T + ' K'; }
    out = $('pkLambda'); if (out) { out.textContent = n2(2898 / T) + ' µm'; }
    out = $('pkMwir'); if (out) { out.textContent = (fm * 100).toFixed(fm < 0.01 ? 2 : 1) + '%'; }
    out = $('pkLwir'); if (out) { out.textContent = (fl * 100).toFixed(1) + '%'; }
    out = $('pkRatio'); if (out) { out.textContent = '×' + Math.round(planck(4, 800) / planck(4, 300)).toLocaleString('en-US'); }
  }

  /* ------------------------------------------------------------- snr chart */

  function buildSnr() {
    var svg = $('snrSvg');
    if (!svg || !S) { return; }
    clear(svg);
    var mono = 'ui-monospace, Consolas, monospace';
    var W = 900, H = 260, x0 = 54, x1 = 862, y0 = 26, y1 = 214;
    var n = S.run.series.length;

    var yOf = function (db) { return y1 - clamp(db, 0, 80) / 80 * (y1 - y0); };
    var xOf = function (i) { return x0 + i / (n - 1) * (x1 - x0); };

    [0, 20, 40, 60, 80].forEach(function (db) {
      var y = yOf(db);
      svg.appendChild(el('line', { x1: x0, y1: y, x2: x1, y2: y, stroke: '#1b2531', 'stroke-width': '1' }));
      svg.appendChild(el('text', {
        x: x0 - 10, y: y + 3.5, fill: '#5d6b7c', 'font-size': '10', 'font-family': mono, 'text-anchor': 'end'
      }, db === 0 ? '0 dB' : String(db)));
    });
    [0, 10, 20, 29].forEach(function (i) {
      if (i >= n) { return; }
      svg.appendChild(el('text', {
        x: xOf(i), y: y1 + 18, fill: '#5d6b7c', 'font-size': '10', 'font-family': mono, 'text-anchor': 'middle'
      }, '帧 ' + i));
    });

    function line(list, stroke, width) {
      var d = '';
      for (var i = 0; i < list.length; i++) {
        d += (i ? 'L' : 'M') + xOf(i).toFixed(1) + ',' + yOf(list[i].snr).toFixed(1);
      }
      svg.appendChild(el('path', { d: d, fill: 'none', stroke: stroke, 'stroke-width': width, 'stroke-linejoin': 'round' }));
    }

    // the SNR the jammer removes, as a filled band between the two runs
    var band = 'M0,0';
    var d = '';
    for (var i = 0; i < n; i++) {
      d += (i ? 'L' : 'M') + xOf(i).toFixed(1) + ',' + yOf(S.run.series[i].snr).toFixed(1);
    }
    for (var k = n - 1; k >= 0; k--) {
      d += 'L' + xOf(k).toFixed(1) + ',' + yOf(S.jam.series[k].snr).toFixed(1);
    }
    svg.appendChild(el('path', { d: d + 'Z', fill: 'rgba(255,154,77,.10)', stroke: 'none' }));

    line(S.jam.series, '#ff9a4d', 1.6);
    line(S.run.series, '#4cd7ff', 2);

    // legend
    var ly = y0 + 6;
    svg.appendChild(el('line', { x1: x1 - 196, y1: ly, x2: x1 - 180, y2: ly, stroke: '#4cd7ff', 'stroke-width': '2' }));
    svg.appendChild(el('text', { x: x1 - 174, y: ly + 3.5, fill: '#a9bccc', 'font-size': '10.5', 'font-family': mono }, '无源'));
    svg.appendChild(el('line', { x1: x1 - 96, y1: ly, x2: x1 - 80, y2: ly, stroke: '#ff9a4d', 'stroke-width': '2' }));
    svg.appendChild(el('text', { x: x1 - 74, y: ly + 3.5, fill: '#a9bccc', 'font-size': '10.5', 'font-family': mono }, '有源压制'));
  }

  /* ------------------------------------------------------------------- ecm */

  function buildEcm() {
    if (!S) { return; }
    var a = S.run, b = S.jam;
    var fa = frameAt('run', 12), fb = frameAt('jam', 12);
    if ($('snrA')) { $('snrA').textContent = n1(fa.snr) + ' dB'; }
    if ($('ptsA')) { $('ptsA').textContent = fa.npts + ' 点'; }
    if ($('snrB')) { $('snrB').textContent = n1(fb.snr) + ' dB'; }
    if ($('ptsB')) { $('ptsB').textContent = fb.npts + ' 点'; }
    drawRadar($('ecmAov'), fa, a.rd, false);
    drawRadar($('ecmBov'), fb, b.rd, false);
    if ($('winA') && fa.db_hi != null) {
      $('winA').textContent = n1(fa.db_lo) + ' ⋯ ' + n1(fa.db_hi) + ' dBW';
    }
    if ($('winB') && fb.db_hi != null) {
      $('winB').textContent = (fb.db_lo > 0 ? '+' : '') + n1(fb.db_lo) + ' ⋯ +' + n1(fb.db_hi) + ' dBW';
    }
    if ($('winGap') && fa.db_hi != null && fb.db_hi != null) {
      $('winGap').textContent = n1(fb.db_hi - fa.db_hi) + ' dB';
    }

    var js = b.ecm && b.ecm.js_db;
    if ($('statJs') && js) { $('statJs').textContent = n1(js[0]) + '–' + n1(js[1]) + ' dB'; }
    var rise = b.ecm && b.ecm.noise_rise_db;
    if ($('statRise') && rise) { $('statRise').textContent = n1(rise[0]) + '–' + n1(rise[1]) + ' dB'; }
    var drop = b.ecm && b.ecm.drop_db ? b.ecm.drop_db[12] : null;
    if ($('statDrop') && drop) { $('statDrop').textContent = n1(drop.snr_drop_db) + ' dB'; }

    var KIND = {
      chaff_cloud: '箔条云', corner_reflector: '角反射器', decoy: '拖曳诱饵',
      infrared_flare: '红外诱饵弹', noise_jammer: '宽带噪声压制', false_target: 'DRFM 假目标',
      velocity_gate_pull_off: '速度波门拖引'
    };
    function params(e) {
      var c = e.cfg || {}, bits = [];
      switch (e.kind) {
        case 'chaff_cloud':
          if (c.rcs_m2) { bits.push('σ ' + Math.round(c.rcs_m2) + ' m²'); }
          if (c.radius_m) { bits.push('半径 ' + c.radius_m + ' m'); }
          if (c.growth_mps) { bits.push(c.growth_mps + ' m/s 膨胀'); }
          if (c.decay_s) { bits.push(c.decay_s + ' s 衰减'); }
          break;
        case 'corner_reflector':
          if (c.size_m) { bits.push('边长 ' + c.size_m + ' m'); }
          break;
        case 'decoy':
          if (c.tether_m) { bits.push('拖缆 ' + c.tether_m + ' m'); }
          if (c.rcs_m2) { bits.push('σ ' + c.rcs_m2 + ' m²'); }
          break;
        case 'infrared_flare':
          if (c.temp_c) { bits.push(Math.round(c.temp_c) + ' °C'); }
          if (c.decay_s) { bits.push(c.decay_s + ' s 燃烧'); }
          break;
        case 'noise_jammer':
          if (c.erp_w) { bits.push(Math.round(c.erp_w) + ' W ERP'); }
          if (c.jam_bandwidth_mhz) { bits.push(c.jam_bandwidth_mhz + ' MHz'); }
          if (c.mode) { bits.push(c.mode); }
          if (c.sidelobe_db) { bits.push('旁瓣 ' + c.sidelobe_db + ' dB'); }
          break;
        case 'false_target':
          if (c.count) { bits.push(c.count + ' 个复制品'); }
          if (c.range_offset_m) { bits.push('+' + Math.round(c.range_offset_m) + ' m 起'); }
          if (c.range_step_m) { bits.push('步进 ' + Math.round(c.range_step_m) + ' m'); }
          if (c.power_ratio) { bits.push('功率 ×' + c.power_ratio); }
          break;
        case 'velocity_gate_pull_off':
          if (c.rate_mps) { bits.push(c.rate_mps + ' m/s 拖引'); }
          if (c.max_offset_mps) { bits.push('上限 ' + Math.round(c.max_offset_mps) + ' m/s'); }
          if (c.power_ratio) { bits.push('功率 ×' + c.power_ratio); }
          break;
      }
      var at = e.at || {};
      if (at.rcs) { bits.push('本帧 σ ' + n1(at.rcs) + ' m²'); }
      return bits.join(' · ') || '—';
    }
    function rows(entities, target) {
      var out = '';
      (entities || []).forEach(function (e) {
        var kind = KIND[e.kind] || e.kind || '—';
        var at = e.at || {};
        var dist = at.r ? Math.round(at.r) + ' m' : '—';
        out += '<tr><td>' + (e.id || '—') + '</td><td>' + kind + '</td><td>' + dist + '</td><td>' +
          params(e) + '</td></tr>';
      });
      if (target) { target.innerHTML = out; }
    }
    rows(a.ecm && a.ecm.entities, $('entA'));
    rows(b.ecm && b.ecm.entities, $('entB'));

    var erp = 0;
    (b.ecm && b.ecm.entities || []).forEach(function (e) { if (e.erp_w && e.erp_w > erp) { erp = e.erp_w; } });
    if ($('statErp') && erp) { $('statErp').textContent = Math.round(erp) + ' W ERP'; }
  }

  /* --------------------------------------------------------------- recipes */

  var KIND = {
    chaff_cloud: '箔条云', corner_reflector: '角反射器', decoy: '拖曳诱饵',
    infrared_flare: '红外诱饵弹', noise_jammer: '宽带噪声压制', false_target: 'DRFM 假目标',
    velocity_gate_pull_off: '速度波门拖引'
  };

  var BACKDROP = {
    skyclear: '高空晴空', cumulus: '积云层', overcast: '航拍云顶',
    city: '城市环岛道路', rural: '乡村田野公路', industrial: '工业区停车场',
    open: '开阔海面', port: '集装箱港口', island: '近岸岛礁'
  };
  var DOMAIN = { air: '空域', land: '陆域', sea: '海域' };

  function buildWall() {
    var wall = $('wall');
    if (!wall || !S || !S.recipes || !S.recipes.length) { return; }
    var order = ['air', 'land', 'sea'];
    var html = '';
    order.forEach(function (dom) {
      var list = S.recipes.filter(function (r) { return r.domain === dom; });
      if (!list.length) { return; }
      html += '<div class="domain-row"><b>' + (DOMAIN[dom] || dom) + '</b><i></i><b>' + list.length + ' 组</b></div>';
      list.forEach(function (r) {
        var parts = r.id.split('_');
        var variant = parts[2] || '';
        var name = BACKDROP[variant] || variant;
        var ecm = r.ecm === 'active' ? '有源' : '无源';
        var meta = dom + ' · ' + (r.band || '—') + ' · ' + n2((r.range_m || 0) / 1000) + ' km';
        html += '<button type="button" class="tile" data-img="data/' + r.thumb + '"' +
          ' data-cap="' + r.id + ' · ' + name + ' · ' + ecm + '">' +
          '<img src="data/' + r.thumb + '" width="620" height="465" loading="lazy" alt="' + name + '，' + ecm + '干扰配置">' +
          '<span class="cap">' + name + ' · ' + ecm + '<em>' + meta + '</em></span></button>';
      });
    });
    wall.innerHTML = html;
    wall.addEventListener('click', function (ev) {
      var b = ev.target.closest('.tile');
      if (!b) { return; }
      openLb(b.getAttribute('data-img'), b.getAttribute('data-cap'));
    });
  }

  /* -------------------------------------------------------------- lightbox */

  function openLb(src, cap) {
    var lb = $('lb');
    if (!lb || !src) { return; }
    $('lbImg').src = src;
    $('lbCap').textContent = cap || '';
    lb.hidden = false;
    document.body.style.overflow = 'hidden';
    $('lbX').focus();
  }
  function closeLb() {
    var lb = $('lb');
    if (!lb) { return; }
    lb.hidden = true;
    document.body.style.overflow = '';
  }

  /* ----------------------------------------------------------------- wiring */

  function wire() {
    var scrub = $('scrub');
    if (scrub) {
      scrub.addEventListener('input', function () {
        stop();
        render(parseInt(scrub.value, 10));
      });
    }
    var playBtn = $('play');
    if (playBtn) {
      playBtn.addEventListener('click', function () {
        if (!S) { return; }
        if (st.playing) { stop(); } else { play(); }
      });
    }
    var chans = document.querySelectorAll('.chan button');
    for (var i = 0; i < chans.length; i++) {
      chans[i].addEventListener('click', function (ev) {
        var b = ev.currentTarget;
        st.chan = b.getAttribute('data-chan');
        for (var k = 0; k < chans.length; k++) { chans[k].setAttribute('aria-pressed', 'false'); }
        b.setAttribute('aria-pressed', 'true');
        if (S) { render(st.frame); }
      });
    }
    var temp = $('temp');
    if (temp) {
      temp.addEventListener('input', function () { renderPlanck(parseInt(temp.value, 10)); });
    }
    var lb = $('lb');
    if (lb) {
      $('lbX').addEventListener('click', closeLb);
      lb.addEventListener('click', function (ev) { if (ev.target === lb) { closeLb(); } });
    }
    var bar = document.querySelector('.bar');
    if (bar) {
      var prog = document.createElement('i');
      prog.className = 'prog';
      prog.setAttribute('aria-hidden', 'true');
      bar.appendChild(prog);
      var onScroll = function () {
        var h = document.documentElement;
        var max = h.scrollHeight - window.innerHeight;
        prog.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + '%';
      };
      window.addEventListener('scroll', onScroll, { passive: true });
      onScroll();
    }
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') { closeLb(); }
      if (!S || ev.target.tagName === 'INPUT') { return; }
      if (ev.key === 'ArrowRight') { stop(); render((st.frame + 1) % S.run.frames); }
      if (ev.key === 'ArrowLeft') { stop(); render((st.frame - 1 + S.run.frames) % S.run.frames); }
      if (ev.key === ' ') { ev.preventDefault(); if (st.playing) { stop(); } else { play(); } }
    });
    // click the optical panel to open the full frame
    var opt = $('optImg');
    if (opt) {
      opt.style.cursor = 'zoom-in';
      opt.addEventListener('click', function () {
        if (!S) { return; }
        openLb(opt.src, '帧 ' + pad6(st.frame) + ' · ' + (st.chan === 'optical' ? '可见光' :
          (st.chan === 'mwir' ? '中波红外 3–5 µm' : '长波红外 8–12 µm')));
      });
    }
  }

  /* ---------------------------------------------------------------- preload */

  function preload() {
    if (!S) { return; }
    var queue = [], i;
    for (i = 0; i < S.run.frames; i++) {
      queue.push(imgSrc(S.run.id, 'optical', i));
      queue.push(imgSrc(S.run.id, 'mwir', i));
      queue.push(imgSrc(S.run.id, 'lwir', i));
      queue.push(imgSrc(S.run.id, 'rd', i));
    }
    var k = 0;
    function kick() {
      if (k >= queue.length) { return; }
      var im = new Image();
      im.onload = im.onerror = kick;
      im.src = queue[k++];
    }
    function later() {
      if (window.requestIdleCallback) { window.requestIdleCallback(kick, { timeout: 1200 }); }
      else { window.setTimeout(kick, 60); }
    }
    var start = reduced ? 0 : 6;
    for (i = 0; i < start; i++) { later(); }
    window.addEventListener('load', function () { for (i = 0; i < 3; i++) { later(); } });
  }

  /* ------------------------------------------------------------------- init */

  function init(data) {
    if (!data || !data.run || !data.run.series || !data.run.series.length) { return; }
    S = data;
    var gen = $('gen'); if (gen && data.generated) { gen.textContent = data.generated; }
    buildStrip();
    buildPlanck();
    renderPlanck(st.temp);
    buildSnr();
    buildEcm();
    buildWall();
    wire();
    render(12);
    preload();
    document.documentElement.setAttribute('data-site', 'ready');
  }

  function boot() {
    if (window.SIMKIT_SHOWCASE) { init(window.SIMKIT_SHOWCASE); return; }
    if (!window.fetch) { return; }
    fetch('data/showcase.json', { cache: 'no-cache' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { if (d) { init(d); } })
      .catch(function () { /* static markup stands */ });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
