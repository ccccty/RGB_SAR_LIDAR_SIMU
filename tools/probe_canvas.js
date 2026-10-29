// Verify that the canvases actually painted content (not blank).
const { spawn } = require('child_process');
const { serve } = require('./static_site');
const fs = require('fs');
const path = require('path');
const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PORT = 9224;
let URL = '';   // set once the embedded server is up
const OUT = path.resolve(__dirname, '..', 'out', 'shots');
const sleep = ms => new Promise(r => setTimeout(r, ms));
function send(ws, id, method, params) {
  return new Promise((resolve, reject) => {
    const onMsg = ev => { const m = JSON.parse(ev.data); if (m.id === id) { ws.removeEventListener('message', onMsg); resolve(m); } };
    ws.addEventListener('message', onMsg);
    ws.send(JSON.stringify({ id, method, params: params || {} }));
    setTimeout(() => reject(new Error('timeout ' + method)), 20000);
  });
}
const CANVAS_INK = `(sel) => {
  const c = document.querySelector(sel);
  if (!c) return { err: 'no canvas' };
  const g = c.getContext('2d');
  const d = g.getImageData(0, 0, c.width, c.height).data;
  let ink = 0, sum = 0;
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i] + d[i+1] + d[i+2];
    sum += v;
    if (v > 40) ink++;
  }
  return { w: c.width, h: c.height, inkPct: +(100 * ink / (d.length / 4)).toFixed(2), mean: +(sum / (d.length / 4) / 3).toFixed(1) };
}`;
(async () => {
  // Default: spin up a throwaway server over the working copy, so the probe
  // verifies what is on disk. Pass a URL to check a deployed copy instead.
  let site = null;
  if (process.argv[2]) { URL = process.argv[2]; }
  else { site = await serve(); URL = 'http://127.0.0.1:' + site.port + '/'; }
  const userDir = path.join(OUT, '_profile2');
  fs.rmSync(userDir, { recursive: true, force: true });
  const proc = spawn(CHROME, ['--headless=new', '--disable-gpu', '--no-sandbox', '--remote-debugging-port=' + PORT,
    '--user-data-dir=' + userDir, '--window-size=1500,1000', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });
  let targets = null;
  for (let i = 0; i < 60; i++) { await sleep(250); try { const r = await fetch('http://127.0.0.1:' + PORT + '/json/list'); targets = await r.json(); if (targets.length) break; } catch (e) {} }
  const ws = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.addEventListener('open', res); ws.addEventListener('error', rej); });
  let idc = 1;
  await send(ws, idc++, 'Runtime.enable');
  await send(ws, idc++, 'Page.enable');
  await send(ws, idc++, 'Emulation.setDeviceMetricsOverride', { width: 1500, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send(ws, idc++, 'Page.navigate', { url: URL });
  await sleep(3000);
  const ev = async expr => {
    const r = await send(ws, idc++, 'Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    return r.result && r.result.exceptionDetails ? { ERR: r.result.exceptionDetails.text } : r.result.result.value;
  };
  const res = {};
  await ev('document.querySelectorAll("#tabs button")[2].click()');
  await sleep(1200);
  res.lidar = await ev('(' + CANVAS_INK.toString() + ')("#pcCanvas")');
  await ev('document.querySelectorAll("#tabs button")[3].click()');
  await sleep(900);
  res.radarBoth = await ev('(' + CANVAS_INK.toString() + ')("#rdCanvas")');
  await ev('document.querySelectorAll("#rdView button")[2].click()');
  await sleep(900);
  res.radarPts = await ev('(' + CANVAS_INK.toString() + ')("#rdCanvas")');
  await ev('document.querySelectorAll("#tabs button")[4].click()');
  await sleep(900);
  res.chartRadar = await ev('(' + CANVAS_INK.toString() + ')("#exRadarCanvas")');
  res.chartOpt = await ev('(' + CANVAS_INK.toString() + ')("#exOptCanvas")');
  res.table = await ev('document.getElementById("exTable").innerText.replace(/\\s+/g," ").slice(0,200)');
  // move a slider and confirm the numbers change
  await ev('var s=document.getElementById("exRain"); s.value=40; s.dispatchEvent(new Event("input"))');
  await sleep(400);
  res.afterRain40 = await ev('document.getElementById("exTable").innerText.replace(/\\s+/g," ").slice(0,200)');
  await ev('document.querySelectorAll("#tabs button")[2].click()');
  await sleep(300);
  res.pcTL = await ev('document.getElementById("pcTL").innerText');
  res.pcBR = await ev('document.getElementById("pcBR").innerText');
  await ev('document.querySelectorAll("#tabs button")[3].click()');
  await sleep(400);
  res.rdBR = await ev('document.getElementById("rdBR").innerText');
  res.rdTL = await ev('document.getElementById("rdTL").innerText');
  res.rdPtTL = await ev('document.getElementById("rdPtTL").innerText');
  res.rdPtBR = await ev('document.getElementById("rdPtBR").innerText');
  res.rdBudget = await ev('document.getElementById("rdBudget").innerText.replace(/\s+/g," ").slice(0,240)');
  // walk to the next frame: the budget + point count must follow
  await ev('document.getElementById("rdNext").click()');
  await sleep(500);
  res.rdPtBRNext = await ev('document.getElementById("rdPtBR").innerText');
  res.rdTLNext = await ev('document.getElementById("rdTL").innerText');
  console.log(JSON.stringify(res, null, 1));
  ws.close(); proc.kill(); if (site) site.server.close(); await sleep(300); process.exit(0);
})();