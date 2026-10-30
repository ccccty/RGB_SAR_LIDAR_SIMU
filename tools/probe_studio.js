// Probe the studio UI: does the form populate, does a preview really run, does
// the gallery load images -- all inside headless Chrome, no manual clicking.
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PORT = 9227;
const URL = process.argv[2] || 'http://127.0.0.1:8808/';
const OUT = path.resolve(__dirname, '..', 'out', 'shots');
fs.mkdirSync(OUT, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));

function send(ws, id, method, params) {
  return new Promise((resolve, reject) => {
    const onMsg = ev => {
      const m = JSON.parse(ev.data);
      if (m.id === id) { ws.removeEventListener('message', onMsg); resolve(m); }
    };
    ws.addEventListener('message', onMsg);
    ws.send(JSON.stringify({ id, method, params: params || {} }));
    setTimeout(() => { try { ws.removeEventListener('message', onMsg); } catch (e) {} reject(new Error('timeout ' + method)); }, 30000);
  });
}

(async () => {
  // A unique profile per run: Chrome keeps a lock on the old one and rmSync then
  // fails with EPERM on Windows.
  const userDir = path.join(OUT, '_studio_profile_' + Date.now());
  const proc = spawn(CHROME, ['--headless=new', '--disable-gpu', '--no-sandbox',
    '--remote-debugging-port=' + PORT, '--user-data-dir=' + userDir,
    '--window-size=1500,1200', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });

  let targets = null;
  for (let i = 0; i < 60; i++) {
    await sleep(250);
    try { const r = await fetch('http://127.0.0.1:' + PORT + '/json/list'); targets = await r.json(); if (targets.length) break; } catch (e) {}
  }
  if (!targets || !targets.length) { console.log('CHROME FAILED'); proc.kill(); process.exit(1); }
  const page = targets.find(t => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.addEventListener('open', res); ws.addEventListener('error', rej); });

  const errors = [];
  ws.addEventListener('message', ev => {
    const m = JSON.parse(ev.data);
    if (m.method === 'Runtime.exceptionThrown') {
      const d = m.params.exceptionDetails;
      errors.push('EXC: ' + ((d.exception && d.exception.description) || d.text));
    }
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
      const t = m.params.entry.text;
      if (t.indexOf('favicon') < 0) errors.push('LOG: ' + t);
    }
  });

  let idc = 1;
  await send(ws, idc++, 'Runtime.enable');
  await send(ws, idc++, 'Log.enable');
  await send(ws, idc++, 'Page.enable');
  await send(ws, idc++, 'Emulation.setDeviceMetricsOverride', { width: 1500, height: 1200, deviceScaleFactor: 1, mobile: false });
  await send(ws, idc++, 'Page.navigate', { url: URL });
  await sleep(3500);

  async function ev(expr) {
    const r = await send(ws, idc++, 'Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.result && r.result.exceptionDetails) return { ERR: r.result.exceptionDetails.text };
    return r.result && r.result.result ? r.result.result.value : null;
  }

  const form = await ev("(function(){return {" +
    "presets: document.querySelectorAll('#preset button').length," +
    "weather: document.querySelectorAll('#weather button').length," +
    "sensors: document.querySelectorAll('#sensors button').length," +
    "bands: document.querySelectorAll('#band option').length," +
    "planRows: document.querySelectorAll('#plan table tr').length," +
    "estimate: (document.getElementById('estimate')||{}).textContent||''" +
    "};})()");

  // start a real preview through the UI button
  await ev("document.getElementById('btnPreview').click()");
  let imgs = 0, jobState = '';
  for (let i = 0; i < 60; i++) {
    await sleep(2000);
    const s = await ev("(function(){return {" +
      "imgs: document.querySelectorAll('#gallery img').length," +
      "job: (document.querySelector('.job .state')||{}).textContent||''" +
      "};})()");
    imgs = s.imgs; jobState = s.job;
    if (jobState === 'done' && imgs > 0) break;
    if (jobState === 'failed') break;
  }
  const loaded = await ev("(function(){var a=[].map.call(document.querySelectorAll('#gallery img'),function(i){return i.complete&&i.naturalWidth>0;});return {n:a.length, ok:a.filter(Boolean).length};})()");

  const shot = await send(ws, idc++, 'Page.captureScreenshot', { format: 'png' });
  if (shot.result && shot.result.data) {
    fs.writeFileSync(path.join(OUT, 'studio.png'), Buffer.from(shot.result.data, 'base64'));
  }

  console.log(JSON.stringify({ form, jobState, galleryImgs: imgs, loaded, errors,
                               shot: 'out/shots/studio.png' }, null, 1));
  proc.kill();
  process.exit(0);
})().catch(e => { console.log('PROBE ERROR', e.message); process.exit(1); });
