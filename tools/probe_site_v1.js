/* CDP probe for the CURRENT single-page demo (index.html + site.js).
 *
 * tools/probe_web.js still probes the retired 11-tab app.js shell, so it
 * reports "querySelectorAll of null" on every tab even when this site is
 * perfectly healthy. This probe checks what the shipped page actually does:
 * curated.json is parsed, the three v1 cards render, and no image is broken.
 *
 *   node tools/probe_site_v1.js
 *   node tools/probe_site_v1.js https://ccccty.github.io/RGB_SAR_LIDAR_SIMU/
 */
const { spawn } = require('child_process');
const { serve } = require('./static_site');
const fs = require('fs');
const path = require('path');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PORT = 9225;
const OUT = path.resolve(__dirname, '..', 'out', 'shots');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  let site = null;
  let URL = process.argv[2];
  if (!URL) { site = await serve(); URL = 'http://127.0.0.1:' + site.port + '/'; }
  fs.mkdirSync(OUT, { recursive: true });
  // A fresh profile per run: never bulk-delete a previous Chrome profile.
  const userDir = path.join(OUT, '_site_v1_' + Date.now());
  const proc = spawn(CHROME, ['--headless=new', '--disable-gpu', '--no-sandbox',
    '--remote-debugging-port=' + PORT, '--user-data-dir=' + userDir,
    '--window-size=1400,1000', 'about:blank'], { stdio: 'ignore' });

  let targets = null;
  for (let i = 0; i < 60; i++) {
    await sleep(250);
    try {
      const r = await fetch('http://127.0.0.1:' + PORT + '/json/list');
      targets = await r.json();
      if (targets && targets.length) break;
    } catch (e) { /* Chrome not up yet */ }
  }
  if (!targets || !targets.length) {
    console.log('CHROME FAILED TO START'); proc.kill(); process.exit(1);
  }
  const page = targets.find((t) => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.addEventListener('open', res); ws.addEventListener('error', rej);
  });

  let seq = 0;
  const pending = new Map();
  const errors = [];
  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    if (m.method === 'Runtime.exceptionThrown') {
      errors.push(String((m.params.exceptionDetails || {}).text || 'exception'));
    }
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
      errors.push(m.params.entry.text);
    }
  });
  const send = (method, params) => new Promise((resolve) => {
    const id = ++seq;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ id: id, method: method, params: params || {} }));
  });

  await send('Runtime.enable');
  await send('Log.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: URL });
  await sleep(4000);

  const expr = `(() => {
    const imgs = Array.from(document.images);
    // A lightbox <img> has no src until a card is opened; it is a placeholder,
    // not a broken frame. Only count images that actually point at a file.
    const broken = imgs.filter(i => i.getAttribute('src') &&
                                    i.complete && i.naturalWidth === 0)
                       .map(i => i.getAttribute('src'));
    const meta = document.getElementById('metaScene');
    const hero = document.getElementById('stageTop');
    return {
      title: document.title,
      metaScene: meta ? meta.textContent.trim() : null,
      heroSrc: hero ? hero.getAttribute('src') : null,
      heroNatural: hero ? [hero.naturalWidth, hero.naturalHeight] : null,
      imageCount: imgs.length,
      broken: broken,
      sceneCards: Array.from(document.querySelectorAll('[data-scene]'))
                       .map(e => e.getAttribute('data-scene')),
      sections: Array.from(document.querySelectorAll('section[id]'))
                     .map(e => e.id)
    };
  })()`;
  const out = await send('Runtime.evaluate', { expression: expr, returnByValue: true });
  const value = out && out.result && out.result.result ? out.result.result.value : null;
  const report = { url: URL, page: value, jsErrors: errors };
  console.log(JSON.stringify(report, null, 2));
  const v1Cards = value && value.sceneCards
    ? value.sceneCards.filter((s) => s && s.indexOf('v1_') === 0) : [];
  const ok = value && value.broken.length === 0 && errors.length === 0 &&
    v1Cards.length === 3;
  console.log(ok ? 'SITE PROBE OK' : 'SITE PROBE FAILED');
  try { ws.close(); } catch (e) {}
  proc.kill();
  process.exit(ok ? 0 : 1);
})();
