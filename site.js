/* site.js — simkit air showcase (classic script, no modules, no build).
 *
 * Data contract
 *   data/curated.json  {generated, domain, scenes:[{id,title,cls,band,aspect,
 *                        mods,frames,hero:{mod,frame},note,caption}]}
 *   data/manifest.json fallback when curated.json is missing or empty.
 *
 * Image URL rules
 *   full:  data/<id>/<mod>/<frame:6>.webp   (falls back to .png)
 *   thumb: data/_t/<id>/<mod>/<frame:6>.webp (falls back to .png, then full)
 *
 * Every builder tolerates missing fields. A missing or broken data file must
 * degrade to the static markup already present in index.html, never blank.
 */
(function () {
  'use strict';

  var MODS = {
    optical: { label: '可见光', short: '可见光' },
    mwir: { label: '中波红外', short: '中波' },
    lwir: { label: '长波红外', short: '长波' },
    radar: { label: '雷达 RD', short: '雷达 RD' },
    radar_ra: { label: '雷达 RA', short: '雷达 RA' },
    lidar_range: { label: '激光雷达 距离', short: '激光 距离' },
    lidar_intensity: { label: '激光雷达 强度', short: '激光 强度' },
    sar: { label: 'SAR', short: 'SAR' }
  };
  var HERO_MODS = ['optical', 'mwir', 'lwir', 'radar'];
  var DEFAULT_MODS = ['optical', 'mwir', 'lwir', 'radar'];
  var SIZE = {
    optical: [1280, 960], mwir: [640, 480], lwir: [640, 480],
    radar: [1280, 720], radar_ra: [1280, 800],
    lidar_range: [256, 192], lidar_intensity: [256, 192], sar: [1024, 1024]
  };
  // The showcase was air only. One curated maritime scene is explicitly opted
  // in; everything else matching NON_AIR stays out, including
  // demo_three_ships_plate (an AIR run whose id merely mentions ships).
  // Keeping the opt-in exact means the published air cards do not change when
  // a sea scene is added.
  var NON_AIR = /(ground|surface|sea|ship|boat|vehicle|terrain|city)/i;
  var SEA_ALLOW = /^demo_sea_surface$/;
  function isNonAir(id) { return NON_AIR.test(id) && !SEA_ALLOW.test(id); }

  // frame: -1 means "no choice yet" so the scene's curated hero frame wins.
  var state = { mod: '', frame: -1, split: 52 };
  var booted = false;

  /* ------------------------------------------------------------------ utils */

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function pad6(n) {
    var s = String(Math.max(0, Math.round(Number(n) || 0)));
    while (s.length < 6) { s = '0' + s; }
    return s;
  }
  function toInt(v, dflt) {
    var n = parseInt(v, 10);
    return isFinite(n) ? n : dflt;
  }
  function known(mod) { return !!(mod && MODS[mod]); }
  function label(mod) { return MODS[mod] ? MODS[mod].label : String(mod || '—'); }
  function sizeOf(mod) { return SIZE[mod] || [1280, 720]; }
  function isArr(v) { return Object.prototype.toString.call(v) === '[object Array]'; }
  function byId(doc, id) { return doc && typeof doc.getElementById === 'function' ? doc.getElementById(id) : null; }
  function setText(doc, id, txt) {
    var el = byId(doc, id);
    if (el) { el.textContent = txt == null ? '' : String(txt); }
  }

  /* ------------------------------------------------------------- url builders */

  function frameUrl(id, mod, frame, ext) {
    return 'data/' + id + '/' + mod + '/' + pad6(frame) + '.' + (ext || 'webp');
  }
  function thumbUrl(id, mod, frame, ext) {
    return 'data/_t/' + id + '/' + mod + '/' + pad6(frame) + '.' + (ext || 'webp');
  }
  // Everything is served as webp per the data contract; png is only a safety net
  // because some manifests still carry the pre-conversion .png paths.
  function fullFallbacks(id, mod, frame) {
    return frameUrl(id, mod, frame, 'png');
  }
  function thumbFallbacks(id, mod, frame) {
    return [
      thumbUrl(id, mod, frame, 'png'),
      frameUrl(id, mod, frame, 'webp'),
      frameUrl(id, mod, frame, 'png')
    ].join(' ');
  }

  /* ------------------------------------------------------------- model layer */

  function metricsOf(manifestScene) {
    if (!manifestScene || !isArr(manifestScene.frames) || manifestScene.frames.length < 2) { return null; }
    var frames = manifestScene.frames;
    var a = frames[0] || {}, b = frames[frames.length - 1] || {};
    var ra = Number(a.range_m), rb = Number(b.range_m);
    var ta = Number(a.t), tb = Number(b.t);
    if (!isFinite(ra) || !isFinite(rb) || !isFinite(ta) || !isFinite(tb) || tb === ta) { return null; }
    return { rangeM: ra, deltaM: rb - ra, seconds: tb - ta, speed: Math.abs(rb - ra) / Math.abs(tb - ta), closing: (rb - ra) < 0 };
  }

  function deriveNote(scene, manifestScene) {
    var parts = [scene.frames.length + ' 帧'];
    var m = metricsOf(manifestScene);
    if (m) {
      if (isFinite(m.rangeM)) { parts.push('距离 ' + (m.rangeM / 1000).toFixed(2) + ' km'); }
      if (isFinite(m.speed) && m.speed > 0.5) { parts.push((m.closing ? '闭速 ' : '开速 ') + Math.round(m.speed) + ' m/s'); }
    }
    return parts.join(' · ');
  }

  function normalizeMods(raw, manifestScene) {
    var list = isArr(raw) ? raw.filter(function (m) { return typeof m === 'string' && known(m); }) : [];
    if (!list.length && manifestScene && isArr(manifestScene.mods)) {
      list = manifestScene.mods
        .map(function (m) { return m && typeof m === 'object' ? m.key : m; })
        .filter(function (m) { return typeof m === 'string' && known(m); });
    }
    if (!list.length) { list = DEFAULT_MODS.slice(); }
    var seen = {};
    return list.filter(function (m) { if (seen[m]) { return false; } seen[m] = 1; return true; });
  }

  function normalizeFrames(raw, manifestScene) {
    var frames = [];
    if (isArr(raw)) {
      raw.forEach(function (f) {
        var n = toInt(f, -1);
        if (n >= 0 && frames.indexOf(n) < 0) { frames.push(n); }
      });
    }
    if (!frames.length && manifestScene && isArr(manifestScene.frames)) {
      manifestScene.frames.slice(0, 6).forEach(function (f) {
        var n = toInt(f && f.i, -1);
        if (n >= 0 && frames.indexOf(n) < 0) { frames.push(n); }
      });
    }
    if (!frames.length) { frames = [0]; }
    return frames;
  }

  function normalizeScene(raw, manifestScene) {
    if (!raw || typeof raw !== 'object') { return null; }
    var id = typeof raw.id === 'string' ? raw.id.trim() : '';
    if (!id || isNonAir(id)) { return null; }
    var ms = manifestScene && manifestScene.id === id ? manifestScene : null;
    var mods = normalizeMods(raw.mods, ms);
    var frames = normalizeFrames(raw.frames, ms);
    var hero = raw.hero && typeof raw.hero === 'object' ? raw.hero : {};
    var heroMod = known(hero.mod) && mods.indexOf(hero.mod) >= 0 ? hero.mod : (mods.indexOf('optical') >= 0 ? 'optical' : mods[0]);
    // Without curated guidance, reuse the frame the static markup already ships
    // (index 4) so the fallback path does not refetch the hero images.
    var fallbackFrame = frames.indexOf(4) >= 0 ? 4 : frames[Math.min(frames.length - 1, 4)];
    var heroFrame = toInt(hero.frame, fallbackFrame);
    if (frames.indexOf(heroFrame) < 0) { heroFrame = fallbackFrame; }
    var note = typeof raw.note === 'string' ? raw.note.trim() : '';
    var scene = {
      id: id,
      title: (typeof raw.title === 'string' && raw.title.trim()) || (ms && ms.title) || id,
      cls: typeof raw.cls === 'string' ? raw.cls : (ms && ms.cls) || '',
      band: typeof raw.band === 'string' && raw.band ? raw.band : (ms && ms.band) || '—',
      aspect: typeof raw.aspect === 'string' ? raw.aspect : '',
      mods: mods,
      frames: frames,
      hero: { mod: heroMod, frame: heroFrame },
      note: note,
      caption: typeof raw.caption === 'string' ? raw.caption.trim() : '',
      metrics: metricsOf(ms)
    };
    if (!scene.note) { scene.note = deriveNote(scene, ms); }
    return scene;
  }

  function manifestScenes(manifest) {
    if (!manifest || !isArr(manifest.scenes)) { return []; }
    var out = [];
    manifest.scenes.forEach(function (ms) {
      if (!ms || typeof ms !== 'object' || typeof ms.id !== 'string') { return; }
      if (isNonAir(ms.id)) { return; }
      var scene = normalizeScene({ id: ms.id, title: ms.title, cls: ms.cls, band: ms.band }, ms);
      if (scene) { out.push(scene); }
    });
    return out;
  }

  function buildModel(curated, manifest) {
    var model = { generated: '', domain: 'air', scenes: [], source: 'none', skipped: 0 };
    var byId = {};
    if (manifest && isArr(manifest.scenes)) {
      manifest.scenes.forEach(function (ms) { if (ms && typeof ms.id === 'string') { byId[ms.id] = ms; } });
    }
    if (curated && typeof curated === 'object') {
      if (typeof curated.generated === 'string') { model.generated = curated.generated; }
      if (typeof curated.domain === 'string' && curated.domain) { model.domain = curated.domain; }
      model.source = 'curated';
      if (isArr(curated.scenes)) {
        var seen = {};
        curated.scenes.forEach(function (raw) {
          var scene = normalizeScene(raw, raw && byId[raw.id]);
          if (!scene || seen[scene.id]) { model.skipped++; return; }
          seen[scene.id] = 1;
          model.scenes.push(scene);
        });
      }
    }
    if (!model.scenes.length) {
      var fromManifest = manifestScenes(manifest);
      if (fromManifest.length) {
        model.source = 'manifest';
        model.scenes = fromManifest;
        if (!model.generated && manifest && typeof manifest.generated === 'string') { model.generated = manifest.generated; }
      } else if (!model.scenes.length) {
        model.source = 'none';
      }
    }
    return model;
  }

  /* ------------------------------------------------------------------- views */

  function heroScene(model) {
    return model && isArr(model.scenes) && model.scenes.length ? model.scenes[0] : null;
  }

  function compareFallback(scene) {
    var pref = ['mwir', 'lwir', 'radar'];
    for (var i = 0; i < pref.length; i++) {
      if (scene.mods.indexOf(pref[i]) >= 0 && pref[i] !== scene.hero.mod) { return pref[i]; }
    }
    for (var j = 0; j < scene.mods.length; j++) {
      if (scene.mods[j] !== scene.hero.mod) { return scene.mods[j]; }
    }
    return scene.hero.mod;
  }

  function layerOf(scene, mod, frame, role) {
    var wh = sizeOf(mod);
    return {
      mod: mod,
      role: role,
      label: label(mod),
      src: frameUrl(scene.id, mod, frame, 'webp'),
      fb: fullFallbacks(scene.id, mod, frame),
      w: wh[0],
      h: wh[1],
      alt: scene.title + ' · ' + label(mod) + ' · 帧 ' + pad6(frame)
    };
  }

  function heroView(model, s) {
    var scene = heroScene(model);
    if (!scene) {
      return { empty: true, buttons: [], meta: { scene: '—', frame: '—', band: '—', note: '' }, cap: '' };
    }
    var st = s || {};
    var mod = known(st.mod) && scene.mods.indexOf(st.mod) >= 0 ? st.mod : compareFallback(scene);
    var frame = scene.frames.indexOf(toInt(st.frame, -1)) >= 0 ? toInt(st.frame, -1) : scene.hero.frame;
    var leftMod = scene.mods.indexOf(scene.hero.mod) >= 0 ? scene.hero.mod : scene.mods[0];
    var compare = mod !== leftMod;
    return {
      empty: false,
      scene: scene.id,
      frame: frame,
      compare: compare,
      left: layerOf(scene, leftMod, frame, 'reference'),
      right: layerOf(scene, mod, frame, 'compare'),
      meta: {
        scene: scene.id,
        frame: pad6(frame),
        band: scene.band || '—',
        note: scene.note || '—'
      },
      cap: scene.caption || (scene.title + ' · 左：' + label(leftMod) + '　右：' + label(mod)),
      title: scene.title,
      buttons: scene.mods.filter(function (m) { return HERO_MODS.indexOf(m) >= 0; }).map(function (m) {
        return { mod: m, label: label(m), on: m === mod, pressed: m === mod ? 'true' : 'false' };
      })
    };
  }

  function physicsView(model) {
    var scene = heroScene(model);
    var plan = [
      { key: 'optical', kinds: ['optical'] },
      { key: 'infrared', kinds: ['lwir', 'mwir'] },
      { key: 'radar', kinds: ['radar'] }
    ];
    var frame = scene ? scene.hero.frame : 0;
    var cards = plan.map(function (p) {
      if (!scene) { return { key: p.key, mod: null }; }
      var mod = p.kinds.filter(function (k) { return scene.mods.indexOf(k) >= 0; })[0];
      if (!mod) { return { key: p.key, mod: null }; }
      return {
        key: p.key,
        mod: mod,
        src: frameUrl(scene.id, mod, frame, 'webp'),
        fb: fullFallbacks(scene.id, mod, frame),
        link: frameUrl(scene.id, mod, frame, 'webp'),
        cap: label(mod) + ' · 帧 ' + pad6(frame),
        alt: scene.title + ' · ' + label(mod) + ' · 帧 ' + pad6(frame)
      };
    });
    return { scene: scene ? scene.id : '', frame: frame, cards: cards };
  }

  var CARD_MODS = ['optical', 'mwir', 'lwir', 'radar', 'radar_ra'];

  function cardMod(scene) {
    // Open the card on the modality the curator chose for the hero, not on
    // "whatever comes first". A 7.5 km target is 31 px in the optical frame --
    // an almost empty field -- while the same frame reads clearly in LWIR, and
    // that is precisely why the hero is LWIR there.
    if (scene.hero && scene.hero.mod && scene.mods.indexOf(scene.hero.mod) >= 0) {
      return scene.hero.mod;
    }
    var pref = ['optical', 'mwir', 'lwir', 'radar', 'radar_ra'];
    for (var i = 0; i < pref.length; i++) {
      if (scene.mods.indexOf(pref[i]) >= 0) { return pref[i]; }
    }
    return scene.mods[0];
  }

  function cardView(scene) {
    var shown = scene.mods.filter(function (m) { return CARD_MODS.indexOf(m) >= 0; });
    if (!shown.length) { shown = scene.mods.slice(0, 5); }
    var mod = cardMod(scene);
    var frame = scene.frames.indexOf(scene.hero.frame) >= 0 ? scene.hero.frame : scene.frames[0];
    var wh = sizeOf(mod);
    return {
      id: scene.id,
      title: scene.title,
      cls: scene.cls,
      band: scene.band,
      note: scene.note,
      caption: scene.caption || ('帧 ' + pad6(frame) + ' · ' + label(mod)),
      modCount: shown.length,
      frameCount: scene.frames.length,
      hero: { mod: mod, frame: frame },
      mods: shown.map(function (m) { return { key: m, label: label(m), on: m === mod }; }),
      frames: scene.frames.map(function (i) { return { i: i, on: i === frame, label: pad6(i) }; }),
      img: {
        mod: mod,
        frame: frame,
        src: frameUrl(scene.id, mod, frame, 'webp'),
        fb: fullFallbacks(scene.id, mod, frame),
        w: wh[0],
        h: wh[1],
        alt: scene.title + ' · ' + label(mod) + ' · 帧 ' + pad6(frame)
      }
    };
  }

  function cardViews(model) {
    return (model && isArr(model.scenes) ? model.scenes : []).map(cardView);
  }

  function thumbSrc(id, mod, frame) {
    var wh = sizeOf(mod);
    var tw = 420;
    var th = Math.max(1, Math.round(tw * wh[1] / wh[0]));
    // Prefer the 420 px thumbnail; the chain falls back to the full frame.
    return { src: thumbUrl(id, mod, frame, 'webp'), fb: thumbFallbacks(id, mod, frame), w: tw, h: th };
  }

  function scenesHtml(cards) {
    if (!cards || !cards.length) {
      return '<p class="loading">没有读到精选数据（data/curated.json 与 data/manifest.json 都不可用）。' +
        '原图仍可直接访问 <code>data/&lt;scene&gt;/&lt;mod&gt;/&lt;frame&gt;.webp</code>。</p>';
    }
    return cards.map(function (c) {
      var thumb = thumbSrc(c.id, c.img.mod, c.img.frame);
      var head = '<div class="card-top"><h3>' + esc(c.title) + '</h3>' +
        (c.cls ? '<span class="tag">' + esc(c.cls) + '</span>' : '') + '</div>';
      var params = '<p class="card-params">' +
        '<span>波段 ' + esc(c.band || '—') + '</span>' +
        '<span>' + c.modCount + ' 模态</span>' +
        '<span>' + c.frameCount + ' 帧</span>' +
        '<span class="res" data-res>—</span></p>';
      var fig = '<figure class="card-fig">' +
        '<img class="shot' + (c.img.mod === 'radar' ? ' rd' : '') + '" src="' + esc(c.img.src) +
        '" data-fb="' + esc(c.img.fb) + '" width="' + c.img.w + '" height="' + c.img.h +
        '" loading="lazy" alt="' + esc(c.img.alt) + '">' +
        '<figcaption>' + esc(c.caption) + '</figcaption></figure>';
      var modbar = '<div class="card-mods" role="group" aria-label="切换模态">' +
        c.mods.map(function (m) {
          return '<button type="button" data-pick-mod="' + esc(m.key) + '"' +
            (m.on ? ' class="on" aria-pressed="true"' : ' aria-pressed="false"') + '>' + esc(m.label) + '</button>';
        }).join('') + '</div>';
      var strip = '<div class="card-strip" role="group" aria-label="精选帧">' +
        c.frames.map(function (f) {
          return '<button type="button" class="thumb' + (f.on ? ' on' : '') + '" data-pick-frame="' + f.i +
            '" aria-label="帧 ' + f.label + '" aria-pressed="' + (f.on ? 'true' : 'false') + '">' +
            '<img src="' + esc(thumb.src) + '" data-fb="' + esc(thumb.fb) + '" width="' + thumb.w + '" height="' + thumb.h +
            '" loading="lazy" alt="帧 ' + f.label + ' 缩略图"><span>' + f.i + '</span></button>';
        }).join('') + '</div>';
      var note = c.note ? '<p class="card-note">' + esc(c.note) + '</p>' : '';
      return '<article class="card" data-scene="' + esc(c.id) + '" data-mod="' + esc(c.img.mod) + '">' +
        head + params + fig + modbar + strip + note + '</article>';
    }).join('');
  }

  function rdView(model) {
    var list = model && isArr(model.scenes) ? model.scenes : [];
    for (var i = 0; i < list.length; i++) {
      var s = list[i];
      if (s.mods.indexOf('radar') < 0) { continue; }
      var frame = s.frames.indexOf(s.hero.frame) >= 0 ? s.hero.frame : s.frames[s.frames.length - 1];
      return {
        scene: s.id,
        frame: frame,
        src: frameUrl(s.id, 'radar', frame, 'webp'),
        fb: fullFallbacks(s.id, 'radar', frame),
        cap: '雷达距离-多普勒图 · ' + s.title + ' · 帧 ' + pad6(frame) + ' · ' + s.id,
        note: s.note
      };
    }
    return null;
  }

  /* ------------------------------------------------------------ dom plumbing */

  function setImg(img, o) {
    if (!img || !o || !o.src) { return; }
    img.setAttribute('src', o.src);
    if (o.fb) { img.setAttribute('data-fb', o.fb); } else if (img.removeAttribute) { img.removeAttribute('data-fb'); }
    if (o.w) { img.setAttribute('width', o.w); }
    if (o.h) { img.setAttribute('height', o.h); }
    if (o.alt) { img.setAttribute('alt', o.alt); }
  }

  function applyHero(doc, v) {
    var stage = byId(doc, 'stage');
    // No model: keep the static markup (and its split view) exactly as shipped.
    if (!v || v.empty) { return; }
    if (stage) {
      if (stage.classList) {
        if (v.right && v.right.mod === 'radar') { stage.classList.add('rd'); } else { stage.classList.remove('rd'); }
        if (v.compare) { stage.classList.remove('no-cmp'); } else { stage.classList.add('no-cmp'); }
      }
      if (v.compare && stage.style && stage.style.setProperty) {
        stage.style.setProperty('--split', state.split + '%');
      }
    }
    setImg(byId(doc, 'stageBase'), v.right);
    setImg(byId(doc, 'stageTop'), v.left);
    setText(doc, 'edgeLeft', v.left ? v.left.label : '');
    setText(doc, 'edgeRight', v.right ? v.right.label : '');
    setText(doc, 'metaScene', v.meta.scene);
    setText(doc, 'metaFrame', v.meta.frame);
    setText(doc, 'metaBand', v.meta.band);
    setText(doc, 'metaNote', v.meta.note);
    setText(doc, 'heroCap', v.cap);
    var bar = byId(doc, 'heroMods');
    if (bar && typeof bar.querySelectorAll === 'function') {
      var btns = bar.querySelectorAll('button[data-mod]');
      for (var i = 0; i < btns.length; i++) {
        var b = btns[i];
        var on = v.right && b.getAttribute('data-mod') === v.right.mod;
        if (on && b.classList) { b.classList.add('on'); } else if (b.classList) { b.classList.remove('on'); }
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      }
    }
  }

  function applyPhysics(doc, v) {
    if (!v || !v.cards) { return; }
    v.cards.forEach(function (c) {
      if (!c || !c.src) { return; }
      var img = doc.querySelector ? doc.querySelector('img[data-fill="' + c.key + '"]') : null;
      if (img) {
        setImg(img, c);
        if (img.classList) {
          if (c.mod === 'radar') { img.classList.add('rd'); } else { img.classList.remove('rd'); }
        }
      }
      var cap = doc.querySelector ? doc.querySelector('[data-fill-cap="' + c.key + '"]') : null;
      if (cap) { cap.textContent = c.cap; }
      var link = doc.querySelector ? doc.querySelector('a[data-fill-link="' + c.key + '"]') : null;
      if (link) { link.setAttribute('href', c.link); }
    });
  }

  function bindResolution(doc) {
    var grid = byId(doc, 'sceneGrid');
    if (!grid || typeof grid.querySelectorAll !== 'function') { return; }
    var shots = grid.querySelectorAll('.shot');
    for (var i = 0; i < shots.length; i++) {
      (function (img) {
        function update() {
          var card = img.closest ? img.closest('.card') : null;
          if (!card || !img.naturalWidth) { return; }
          var res = card.querySelector('[data-res]');
          if (res) { res.textContent = img.naturalWidth + '×' + img.naturalHeight; }
        }
        if (typeof img.addEventListener === 'function') { img.addEventListener('load', update); }
        if (img.complete && img.naturalWidth) { update(); }
      })(shots[i]);
    }
  }

  function applyScenes(doc, cards) {
    var grid = byId(doc, 'sceneGrid');
    if (!grid) { return; }
    grid.innerHTML = scenesHtml(cards || []);
    bindResolution(doc);
  }

  function applyRd(doc, v) {
    if (!v) { return; }
    var img = byId(doc, 'rdImg');
    if (img) { setImg(img, { src: v.src, fb: v.fb, w: 1280, h: 720, alt: v.cap }); }
    setText(doc, 'rdCap', v.cap);
    var raw = byId(doc, 'rdRaw');
    if (raw) { raw.setAttribute('href', v.src); }
  }

  function applyFooter(doc, model) {
    var g = model && model.generated ? model.generated : '';
    setText(doc, 'footGenerated', g || '—');
  }

  function renderAll(doc, model) {
    var hero = heroView(model, state);
    state.mod = hero.right ? hero.right.mod : '';
    state.frame = hero.frame || 0;
    applyHero(doc, hero);
    applyPhysics(doc, physicsView(model));
    applyScenes(doc, cardViews(model));
    applyRd(doc, rdView(model));
    applyFooter(doc, model);
    return state;
  }

  /* ---------------------------------------------------------- interactions */

  function applySplit(doc, pct) {
    var p = Math.max(0, Math.min(100, Math.round(Number(pct) || 0)));
    state.split = p;
    var stage = byId(doc, 'stage');
    if (stage && stage.style && stage.style.setProperty) { stage.style.setProperty('--split', p + '%'); }
    var range = byId(doc, 'splitRange');
    if (range) {
      range.value = String(p);
      if (range.style && range.style.setProperty) { range.style.setProperty('--fill', p + '%'); }
    }
  }

  function refreshHero(doc, model) {
    var v = heroView(model, state);
    if (!v.empty) {
      state.frame = v.frame;
      if (v.right) { state.mod = v.right.mod; }
    }
    applyHero(doc, v);
  }

  function wireHero(doc, model) {
    var bar = byId(doc, 'heroMods');
    if (bar && typeof bar.addEventListener === 'function') {
      bar.addEventListener('click', function (ev) {
        var t = ev.target && ev.target.closest ? ev.target.closest('button[data-mod]') : null;
        if (!t) { return; }
        state.mod = t.getAttribute('data-mod');
        refreshHero(doc, model);
      });
    }
    var range = byId(doc, 'splitRange');
    if (range && typeof range.addEventListener === 'function') {
      // The range element seeds the split once; runtime state stays authoritative.
      var seeded = Number(range.value);
      if (isFinite(seeded) && seeded >= 0 && seeded <= 100 && range.value !== '') { state.split = seeded; }
      range.addEventListener('input', function () { applySplit(doc, range.value); });
      applySplit(doc, state.split);
    }
    var stage = byId(doc, 'stage');
    if (stage && typeof stage.addEventListener === 'function') {
      var dragging = false;
      function at(ev) {
        var rect = stage.getBoundingClientRect();
        if (!rect || !rect.width) { return; }
        applySplit(doc, ((ev.clientX - rect.left) / rect.width) * 100);
      }
      stage.addEventListener('pointerdown', function (ev) {
        dragging = true;
        if (stage.setPointerCapture && ev.pointerId != null) {
          try { stage.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
        }
        at(ev);
      });
      stage.addEventListener('pointermove', function (ev) { if (dragging) { at(ev); } });
      stage.addEventListener('pointerup', function () { dragging = false; });
      stage.addEventListener('pointercancel', function () { dragging = false; });
      stage.addEventListener('pointerleave', function () { dragging = false; });
    }
  }

  function updateCardImg(card, scene, mod, frame) {
    if (!card || !scene) { return; }
    var wh = sizeOf(mod);
    var shot = card.querySelector('.shot');
    if (shot) {
      setImg(shot, {
        src: frameUrl(scene.id, mod, frame, 'webp'),
        fb: fullFallbacks(scene.id, mod, frame),
        w: wh[0], h: wh[1],
        alt: scene.title + ' · ' + label(mod) + ' · 第 ' + (frame + 1) + ' 帧'
      });
      if (shot.classList) {
        if (mod === 'radar') { shot.classList.add('rd'); } else { shot.classList.remove('rd'); }
      }
    }
    var mods = card.querySelectorAll ? card.querySelectorAll('[data-pick-mod]') : [];
    for (var i = 0; i < mods.length; i++) {
      var m = mods[i];
      var on = m.getAttribute('data-pick-mod') === mod;
      if (on && m.classList) { m.classList.add('on'); } else if (m.classList) { m.classList.remove('on'); }
      m.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
    var frames = card.querySelectorAll ? card.querySelectorAll('[data-pick-frame]') : [];
    for (var j = 0; j < frames.length; j++) {
      var f = frames[j];
      var sel = toInt(f.getAttribute('data-pick-frame'), -1) === frame;
      if (sel && f.classList) { f.classList.add('on'); } else if (f.classList) { f.classList.remove('on'); }
      f.setAttribute('aria-pressed', sel ? 'true' : 'false');
      var img = f.querySelector ? f.querySelector('img') : null;
      if (img) {
        var th = thumbSrc(scene.id, mod, toInt(f.getAttribute('data-pick-frame'), 0));
        img.setAttribute('src', th.src);
        img.setAttribute('data-fb', th.fb);
      }
    }
    card.setAttribute('data-mod', mod);
    card.setAttribute('data-frame', String(frame));
  }

  function sceneById(model, id) {
    var list = model && isArr(model.scenes) ? model.scenes : [];
    for (var i = 0; i < list.length; i++) { if (list[i].id === id) { return list[i]; } }
    return null;
  }

  function wireScenes(doc, model) {
    var grid = byId(doc, 'sceneGrid');
    if (!grid || typeof grid.addEventListener !== 'function') { return; }
    grid.addEventListener('click', function (ev) {
      var t = ev.target;
      if (!t || !t.closest) { return; }
      var card = t.closest('.card');
      if (!card) { return; }
      var scene = sceneById(model, card.getAttribute('data-scene'));
      if (!scene) { return; }
      var modBtn = t.closest('[data-pick-mod]');
      if (modBtn) {
        var mod = modBtn.getAttribute('data-pick-mod');
        var frame = toInt(card.getAttribute('data-frame'), scene.hero.frame);
        if (scene.frames.indexOf(frame) < 0) { frame = scene.hero.frame; }
        updateCardImg(card, scene, mod, frame);
        return;
      }
      var frameBtn = t.closest('[data-pick-frame]');
      if (frameBtn) {
        var f = toInt(frameBtn.getAttribute('data-pick-frame'), scene.hero.frame);
        var m = card.getAttribute('data-mod') || scene.hero.mod;
        updateCardImg(card, scene, modKnown(scene, m), f);
        return;
      }
      var shot = t.closest('.shot');
      if (shot) {
        var mm = card.getAttribute('data-mod') || scene.hero.mod;
        var ff = toInt(card.getAttribute('data-frame'), scene.hero.frame);
        var wh = sizeOf(mm);
        openLightbox(doc, {
          src: frameUrl(scene.id, mm, ff, 'webp'),
          fb: fullFallbacks(scene.id, mm, ff),
          w: wh[0], h: wh[1],
          cap: scene.title + ' · ' + label(mm) + ' · 帧 ' + pad6(ff) + ' · ' + scene.id
        });
      }
    });
  }

  function modKnown(scene, mod) {
    return known(mod) && scene.mods.indexOf(mod) >= 0 ? mod : scene.hero.mod;
  }

  function openLightbox(doc, o) {
    var lb = byId(doc, 'lightbox');
    if (!lb || !o) { return; }
    var img = byId(doc, 'lbImg');
    if (img) { setImg(img, { src: o.src, fb: o.fb, w: o.w, h: o.h, alt: o.cap || '放大查看的精选帧' }); }
    setText(doc, 'lbCap', o.cap || '');
    lb.removeAttribute('hidden');
    if (lb.classList) { lb.classList.add('on'); }
    // Freeze the page behind the overlay.
    if (doc.body && doc.body.style) {
      lb.__prevOverflow = doc.body.style.overflow || '';
      doc.body.style.overflow = 'hidden';
    }
  }

  function closeLightbox(doc) {
    var lb = byId(doc, 'lightbox');
    if (!lb) { return; }
    lb.setAttribute('hidden', 'hidden');
    if (lb.classList) { lb.classList.remove('on'); }
    if (doc.body && doc.body.style) { doc.body.style.overflow = lb.__prevOverflow || ''; }
  }

  function wireLightbox(doc) {
    var lb = byId(doc, 'lightbox');
    if (!lb) { return; }
    if (typeof lb.addEventListener === 'function') {
      lb.addEventListener('click', function (ev) {
        var t = ev.target;
        if (t === lb || (t && t.closest && t.closest('.lb-close'))) { closeLightbox(doc); }
      });
    }
    if (doc && typeof doc.addEventListener === 'function') {
      doc.addEventListener('keydown', function (ev) {
        if (ev && (ev.key === 'Escape' || ev.key === 'Esc')) { closeLightbox(doc); }
      });
    }
  }

  function wire(doc, model) {
    wireHero(doc, model);
    wireScenes(doc, model);
    wireLightbox(doc);
    applySplit(doc, state.split);
  }

  /* ------------------------------------------------------------------- boot */

  function getJSON(url, fetchImpl) {
    var f = fetchImpl || (typeof fetch === 'function' ? fetch : null);
    if (!f) { return Promise.resolve(null); }
    return Promise.resolve(f(url, { cache: 'no-cache' })).then(function (res) {
      if (!res || !res.ok) { throw new Error(url + ' -> ' + (res && res.status)); }
      return res.json();
    }).catch(function () { return null; });
  }

  function loadModel(fetchImpl) {
    return getJSON('data/curated.json', fetchImpl).then(function (curated) {
      var hasScenes = !!(curated && isArr(curated.scenes) && curated.scenes.length);
      if (hasScenes) { return buildModel(curated, null); }
      return getJSON('data/manifest.json', fetchImpl).then(function (manifest) {
        return buildModel(curated, manifest);
      });
    }).catch(function () { return buildModel(null, null); });
  }

  // The manifest is ~400 KB; when curated.json is absent, fetch it only after the
  // first screen has loaded, so it never competes with the hero images.
  function defer(fn) {
    if (typeof requestIdleCallback === 'function') { requestIdleCallback(fn, { timeout: 1200 }); }
    else { setTimeout(fn, 250); }
  }

  function afterFirstScreen(doc, fn) {
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function' && doc.readyState !== 'complete') {
      window.addEventListener('load', function () { defer(fn); }, { once: true });
    } else {
      defer(fn);
    }
  }

  function finish(doc, model) {
    try {
      renderAll(doc, model);
      wire(doc, model);
    } catch (err) {
      if (typeof console === 'object' && console.warn) { console.warn('[site] render failed', err); }
    }
  }

  function boot() {
    if (booted || typeof document === 'undefined' || !document.getElementById) { return; }
    booted = true;
    var doc = document;
    if (doc.addEventListener) {
      doc.addEventListener('error', function (ev) {
        var img = ev && ev.target;
        if (!img || img.tagName !== 'IMG' || !img.getAttribute) { return; }
        var chain = (img.getAttribute('data-fb') || '').split(/\s+/).filter(Boolean);
        if (!chain.length) { return; }
        img.setAttribute('src', chain.shift());
        img.setAttribute('data-fb', chain.join(' '));
      }, true);
    }
    getJSON('data/curated.json', null).then(function (curated) {
      if (curated && isArr(curated.scenes) && curated.scenes.length) {
        finish(doc, buildModel(curated, null));
        return;
      }
      afterFirstScreen(doc, function () {
        getJSON('data/manifest.json', null).then(function (manifest) {
          finish(doc, buildModel(curated, manifest));
        });
      });
    }).catch(function () {
      finish(doc, buildModel(null, null));
    });
  }

  var API = {
    MODS: MODS,
    state: state,
    pad6: pad6,
    frameUrl: frameUrl,
    thumbUrl: thumbUrl,
    buildModel: buildModel,
    normalizeScene: normalizeScene,
    heroView: heroView,
    physicsView: physicsView,
    cardViews: cardViews,
    scenesHtml: scenesHtml,
    rdView: rdView,
    renderAll: renderAll,
    loadModel: loadModel,
    applySplit: applySplit,
    boot: boot
  };
  if (typeof window !== 'undefined') { window.__SIMKIT_SITE = API; }

  if (typeof document !== 'undefined' && typeof window !== 'undefined' && document.getElementById) {
    if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', boot); } else { boot(); }
  }
})();
