/* CDP probe + screenshotter for the 2026-11-12 showcase page.
 *
 *   node tools/probe_showcase.js                 # local server, local shots
 *   node tools/probe_showcase.js https://...     # same checks against a deploy
 *
 * Checks: no JS exception, no failed request, no broken image, the frame
 * reader actually shows a frame, and the strip/wall rendered. Writes one PNG
 * per section to out/shots/showcase_<name>.png for eyeballing.
 */
const { spawn } = require('child_process');
const { serve } = require('./static_site');
const fs = require('fs');
const path = require('path');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PORT = 9337;
const OUT = path.resolve(__dirname, '..', 'out', 'shots');
const VIEW = { width: 1440, height: 980 };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  let site = null;
  let URL = process.argv[2];
  if (!URL) { site = await serve(); URL = 'http://127.0.0.1:' + site.port + '/'; }
  fs.mkdirSync(OUT, { recursive: true });
  const userDir = path.join(OUT, '_showcase_' + Date.now());
  const proc = spawn(CHROME, ['--headless=new', '--disable-gpu', '--no-sandbox',
    '--hide-scrollbars', '--force-color-profile=srgb',
    '--remote-debugging-port=' + PORT, '--user-data-dir=' + userDir,
    '--window-size=' + VIEW.width + ',' + VIEW.height, 'about:blank'], { stdio: 'ignore' });

  let targets = null;
  for (let i = 0; i < 80; i++) {
    await sleep(250);
    try {
      const r = await fetch('http://127.0.0.1:' + PORT + '/json/list');
      targets = await r.json();
      if (targets && targets.length) break;
    } catch (e) { /* not up yet */ }
  }
  if (!targets || !targets.length) { console.log('CHROME FAILED'); proc.kill(); process.exit(1); }
  const page = targets.find((t) => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.addEventListener('open', res); ws.addEventListener('error', rej); });

  let seq = 0;
  const pending = new Map();
  const errors = [];
  const failed = [];
  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    if (m.method === 'Runtime.exceptionThrown') {
      const d = m.params.exceptionDetails || {};
      errors.push((d.exception && d.exception.description) || d.text || 'exception');
    }
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') errors.push(m.params.entry.text);
    if (m.method === 'Network.loadingFailed') failed.push(m.params.errorText + ' ' + (m.params.type || ''));
    if (m.method === 'Network.responseReceived' && m.params.response.status >= 400) {
      failed.push(m.params.response.status + ' ' + m.params.response.url);
    }
  });
  const send = (method, params) => new Promise((resolve) => {
    const id = ++seq;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ id: id, method: method, params: params || {} }));
  });

  await send('Runtime.enable');
  await send('Log.enable');
  await send('Network.enable');
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', {
    width: VIEW.width, height: VIEW.height, deviceScaleFactor: 1, mobile: false
  });
  await send('Page.navigate', { url: URL });

  let ready = false;
  for (let i = 0; i < 60; i++) {
    await sleep(300);
    const r = await send('Runtime.evaluate', { expression: "document.documentElement.getAttribute('data-site') === 'ready'", returnByValue: true });
    if (r.result && r.result.result && r.result.result.value) { ready = true; break; }
  }
  await sleep(1200);

  // Walk the page so lazy images actually fetch: without this the recipe wall
  // screenshots as 18 empty cards and the broken-image count reads 0.
  const H = await send('Runtime.evaluate', { expression: 'document.documentElement.scrollHeight', returnByValue: true });
  const total = H.result.result.value || 0;
  // scroll-behavior:smooth turns these into animations, so a fast loop never
  // actually reaches the recipe wall. Pin it to auto for the walk.
  await send('Runtime.evaluate', { expression: "document.documentElement.style.scrollBehavior='auto'" });
  for (let y = 0; y < total; y += 500) {
    await send('Runtime.evaluate', { expression: 'window.scrollTo(0,' + y + ')' });
    await sleep(160);
  }
  await send('Runtime.evaluate', { expression: 'window.scrollTo(0,0)' });
  await sleep(900);

  const stats = await send('Runtime.evaluate', {
    expression: "(function(){var g=function(id){var e=document.getElementById(id);return e?e.textContent:'';};" +
      "return JSON.stringify({title: document.title, strip: document.querySelectorAll('#strip button').length," +
      " tiles: document.querySelectorAll('.tile').length," +
      " frame: g('tFrame'), range: g('tRange'), snr: g('tSnr')," +
      " pkMwir: g('pkMwir'), pkLwir: g('pkLwir')," +
      " planckPaths: document.querySelectorAll('#planckSvg path').length," +
      " snrPaths: document.querySelectorAll('#snrSvg path').length," +
      " imgs: document.images.length," +
      " broken: Array.prototype.filter.call(document.images, function(i){return i.getAttribute('src') && i.complete && i.naturalWidth === 0;}).length," +
      " scrollH: document.documentElement.scrollHeight});})()",
    returnByValue: true
  });
  const raw = stats.result && stats.result.result ? stats.result.result.value : null;
  if (!raw) {
    console.log('page did not expose the showcase DOM (old build or hard failure)');
    console.log('js errors:', errors.slice(0, 4), 'failed:', failed.slice(0, 4));
    proc.kill();
    process.exit(1);
  }
  const s = JSON.parse(raw);

  const sections = [['hero', '#rack'], ['bands', '#bands'], ['ecm', '#ecm'], ['recipes', '#recipes'], ['limits', '#limits']];
  for (const [name, sel] of sections) {
    const rect = await send('Runtime.evaluate', {
      expression: "(() => { const e = document.querySelector('" + sel + "');" +
        " if (!e) return null; const r = e.getBoundingClientRect();" +
        " return JSON.stringify({y: Math.round(r.top + window.scrollY), h: Math.round(r.height)}); })()",
      returnByValue: true
    });
    if (!rect.result.result.value) { console.log('missing section', sel); continue; }
    const box = JSON.parse(rect.result.result.value);
    const shot = await send('Page.captureScreenshot', {
      format: 'png',
      clip: { x: 0, y: box.y, width: VIEW.width, height: Math.min(box.h, 2600), scale: 1 },
      captureBeyondViewport: true
    });
    if (shot.result && shot.result.data) {
      fs.writeFileSync(path.join(OUT, 'showcase_' + name + '.png'), Buffer.from(shot.result.data, 'base64'));
    } else {
      console.log('screenshot failed for', name, JSON.stringify(shot).slice(0, 200));
    }
  }

  console.log('ready:', ready);
  console.log('stats:', JSON.stringify(s));
  console.log('js errors:', errors.length ? errors : 'none');
  console.log('failed requests:', failed.length ? failed.slice(0, 8) : 'none');
  const ok = ready && !errors.length && !failed.length && s.broken === 0 && s.tiles === 18 && s.strip === 30;
  console.log(ok ? 'SHOWCASE OK' : 'SHOWCASE ISSUES');
  proc.kill();
  if (!ok) process.exitCode = 1;
  process.exit(process.exitCode || 0);
})();
