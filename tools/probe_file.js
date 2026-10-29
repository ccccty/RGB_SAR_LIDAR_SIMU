// Verify the site loads with no server at all (file://), which is the
// air-gapped case: no fetch, no CDN, everything from data/boot.js.
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');
const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PORT = 9231;
const OUT = path.resolve(__dirname, '..', 'out', 'shots');
fs.mkdirSync(OUT, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const ROOT = path.resolve(__dirname, '..');
const FILE_URL = 'file:///' + path.join(ROOT, 'index.html').replace(/\\/g, '/');

function send(ws, id, method, params) {
  return new Promise((resolve, reject) => {
    const onMsg = ev => { const m = JSON.parse(ev.data); if (m.id === id) { ws.removeEventListener('message', onMsg); resolve(m); } };
    ws.addEventListener('message', onMsg);
    ws.send(JSON.stringify({ id, method, params: params || {} }));
    setTimeout(() => reject(new Error('timeout ' + method)), 20000);
  });
}

(async () => {
  const userDir = path.join(OUT, '_profile_file');
  fs.rmSync(userDir, { recursive: true, force: true });
  const proc = spawn(CHROME, ['--headless=new', '--disable-gpu', '--no-sandbox',
    '--allow-file-access-from-files',
    '--remote-debugging-port=' + PORT, '--user-data-dir=' + userDir,
    '--window-size=1500,1000', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });
  let targets = null;
  for (let i = 0; i < 60; i++) {
    await sleep(250);
    try { const r = await fetch('http://127.0.0.1:' + PORT + '/json/list'); targets = await r.json(); if (targets.length) break; } catch (e) {}
  }
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
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') errors.push('LOG: ' + m.params.entry.text);
  });
  let idc = 1;
  await send(ws, idc++, 'Runtime.enable');
  await send(ws, idc++, 'Log.enable');
  await send(ws, idc++, 'Page.enable');
  await send(ws, idc++, 'Emulation.setDeviceMetricsOverride', { width: 1500, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send(ws, idc++, 'Page.navigate', { url: FILE_URL });
  await sleep(4000);
  const ev = async e => {
    const r = await send(ws, idc++, 'Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true });
    return r.result && r.result.result ? r.result.result.value : null;
  };
  const res = {};
  res.url = await ev('location.protocol');
  res.hasBoot = await ev('!!(window.SIMKIT_DATA && window.SIMKIT_DATA.manifest)');
  res.scenes = await ev('window.SIMKIT_DATA ? window.SIMKIT_DATA.manifest.scenes.length : -1');
  const tabs = ['gallery','compare','lidar','radar','explore','about'];
  res.tabs = {};
  for (let i = 0; i < tabs.length; i++) {
    await ev('document.querySelectorAll("#tabs button")[' + i + '].click()');
    await sleep(800);
    res.tabs[tabs[i]] = await ev('(() => { var s=document.getElementById("' + tabs[i] + '");' +
      'var imgs=[].slice.call(s.querySelectorAll("img"));' +
      'return {on:s.classList.contains("on"), imgs:imgs.length, broken:imgs.filter(function(i){return i.naturalWidth===0;}).length};})()');
  }
  res.errors = errors;
  console.log(JSON.stringify(res, null, 1));
  ws.close(); proc.kill(); await sleep(300); process.exit(0);
})();