// CDP probe: drive the real page in headless Chrome, collect JS errors,
// check every tab renders, and save one screenshot per tab.
const { spawn } = require('child_process');
const { serve } = require('./static_site');
const fs = require('fs');
const path = require('path');

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PORT = 9223;
let URL = '';   // set once the embedded server is up
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
    setTimeout(() => { try { ws.removeEventListener('message', onMsg); } catch (e) {} reject(new Error('timeout ' + method)); }, 20000);
  });
}

(async () => {
  // Default: spin up a throwaway server over the working copy, so the probe
  // verifies what is on disk. Pass a URL to check a deployed copy instead.
  let site = null;
  if (process.argv[2]) { URL = process.argv[2]; }
  else { site = await serve(); URL = 'http://127.0.0.1:' + site.port + '/'; }
  const userDir = path.join(OUT, '_profile');
  fs.rmSync(userDir, { recursive: true, force: true });
  const proc = spawn(CHROME, ['--headless=new', '--disable-gpu', '--no-sandbox',
    '--remote-debugging-port=' + PORT, '--user-data-dir=' + userDir,
    '--window-size=1500,1000', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });

  let targets = null;
  for (let i = 0; i < 60; i++) {
    await sleep(250);
    try { const r = await fetch('http://127.0.0.1:' + PORT + '/json/list'); targets = await r.json(); if (targets.length) break; } catch (e) {}
  }
  if (!targets || !targets.length) { console.log('CHROME FAILED TO START'); proc.kill(); process.exit(1); }
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
      errors.push('LOG: ' + m.params.entry.text);
    }
  });

  let idc = 1;
  await send(ws, idc++, 'Runtime.enable');
  await send(ws, idc++, 'Log.enable');
  await send(ws, idc++, 'Page.enable');
  await send(ws, idc++, 'Emulation.setDeviceMetricsOverride', { width: 1500, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send(ws, idc++, 'Page.navigate', { url: URL });
  await sleep(3000);

  async function evalJs(expr) {
    const r = await send(ws, idc++, 'Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.result && r.result.exceptionDetails) return { ERR: r.result.exceptionDetails.text + ' ' + JSON.stringify(r.result.exceptionDetails.exception && r.result.exceptionDetails.exception.description) };
    return r.result && r.result.result ? r.result.result.value : null;
  }

  const tabs = ['gallery', 'compare', 'lidar', 'radar', 'explore', 'about'];
  const report = {};
  for (let i = 0; i < tabs.length; i++) {
    const t = tabs[i];
    const click = await evalJs('document.querySelectorAll("#tabs button")[' + i + '].click()');
    await sleep(900);
    const q = '(() => { var s=document.getElementById("' + t + '");' +
      'var imgs=[].slice.call(s.querySelectorAll("img")).map(function(i){return {w:i.naturalWidth,h:i.naturalHeight};});' +
      'var cvs=[].slice.call(s.querySelectorAll("canvas")).map(function(c){return c.width+"x"+c.height;});' +
      'return {on: s.classList.contains("on"), onList: [].slice.call(document.querySelectorAll("section")).filter(function(x){return x.classList.contains("on");}).map(function(x){return x.id;}), imgs: imgs.length, broken: imgs.filter(function(i){return i.w===0;}).length, cvs: cvs, h: s.offsetHeight}; })()';
    report[t] = await evalJs(q);
    if (click) report[t].clickErr = click;
    const shot = await send(ws, idc++, 'Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
    fs.writeFileSync(path.join(OUT, t + '.png'), Buffer.from(shot.result.data, 'base64'));
  }
  // Every modality button the gallery offers must actually resolve to an image.
  // A button whose frame entry is missing renders a blank pane, which is the
  // failure mode this catches.
  await evalJs('document.querySelectorAll("#tabs button")[0].click()');
  await sleep(600);
  const mods = [];
  const nmods = await evalJs('document.querySelectorAll("#galMod button").length');
  // Wait for the decode, not for a guessed delay: over a real network a 700 ms
  // nap reports a slow image as a missing one.
  const settle = 'new Promise(function(res){ var im=document.getElementById("galImg");' +
    'var t0=Date.now(); (function tick(){ if (im.complete && im.naturalWidth>0) return res(1);' +
    'if (Date.now()-t0>8000) return res(0); setTimeout(tick,100); })(); })';
  for (let i = 0; i < (nmods || 0); i++) {
    await evalJs('document.querySelectorAll("#galMod button")[' + i + '].click()');
    await evalJs(settle);
    await sleep(120);
    mods.push(await evalJs('(() => { var im=document.getElementById("galImg");' +
      'var b=[].slice.call(document.querySelectorAll("#galMod button"))[' + i + '];' +
      'return {label: b ? b.textContent : "?", src: (im.getAttribute("src") || ""),' +
      'w: im.naturalWidth, h: im.naturalHeight, shown: im.style.display !== "none"}; })()'));
  }
  const badMods = mods.filter(m => !m || m.w === 0 || !m.shown);

  console.log(JSON.stringify({ errors: errors, report: report,
    mods: mods, badMods: badMods }, null, 1));
  ws.close();
  proc.kill();
  await sleep(400);
  process.exit(0);
})();