/* studio/app.js 的无浏览器初始化冒烟测试（零依赖，Node 原生 vm）。
 *
 * 目的：抓出首屏 init() 阶段的 ReferenceError / TypeError —— 例如拼错函数名、
 * 访问 HTML 里不存在的 id、在 presets 回来之前读了未初始化的结构。真实交互
 * 与 XHR 不在范围内：所有网络请求挂起，定时器不执行回调。
 *
 * 用法：node tools/studio_smoke_dom.js
 * 退出码：0 = 初始化未抛异常；1 = 有运行时错误（打印堆栈）。
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(ROOT, 'studio', 'index.html'), 'utf8');
const code = fs.readFileSync(path.join(ROOT, 'studio', 'app.js'), 'utf8');

/* 只让真实存在于 index.html 的 id 取到元素，其余返回 null：这样既能验证
   「JS 引用了不存在的 id」的判空容错，也不会用假 DOM 掩盖问题。 */
const htmlIds = new Set();
const idRe = /\bid="([A-Za-z0-9_-]+)"/g;
let m;
while ((m = idRe.exec(html))) { htmlIds.add(m[1]); }

function makeEl(tag) {
  const el = {
    tagName: String(tag || 'div').toUpperCase(),
    children: [], parentNode: null, style: {}, dataset: {}, attrs: {},
    id: '', className: '', textContent: '', innerHTML: '', value: '',
    hidden: false, disabled: false, checked: false, open: false,
    title: '', href: '', src: '', type: '', tabIndex: 0,
    onclick: null, onchange: null, oninput: null, oncontextmenu: null,
    firstChild: null, lastChild: null, width: 960, height: 720,
    classList: {
      add() {}, remove() {}, toggle() {}, contains() { return false; }
    },
    appendChild(c) {
      this.children.push(c);
      c.parentNode = this;
      this.firstChild = this.children[0];
      this.lastChild = this.children[this.children.length - 1];
      return c;
    },
    removeChild(c) {
      const i = this.children.indexOf(c);
      if (i >= 0) { this.children.splice(i, 1); }
      c.parentNode = null;
      this.firstChild = this.children[0] || null;
      this.lastChild = this.children[this.children.length - 1] || null;
      return c;
    },
    insertBefore(c) { return this.appendChild(c); },
    remove() { if (this.parentNode) { this.parentNode.removeChild(this); } },
    setAttribute(k, v) {
      this.attrs[k] = String(v);
      if (k === 'id') { this.id = String(v); }
    },
    getAttribute(k) {
      return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null;
    },
    removeAttribute(k) { delete this.attrs[k]; },
    hasAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k); },
    addEventListener() {}, removeEventListener() {},
    focus() {}, blur() {}, click() {}, scrollIntoView() {},
    querySelector() { return null; }, querySelectorAll() { return []; },
    getElementsByTagName() { return []; },
    getBoundingClientRect() {
      return { width: 960, height: 720, top: 0, left: 0, right: 960, bottom: 720 };
    },
    closest() { return null; }, contains() { return false; },
    cloneNode() { return makeEl(this.tagName); },
    getContext() { return null; }, toDataURL() { return ''; }
  };
  return el;
}

const cache = new Map();
function byId(id) {
  if (!htmlIds.has(id)) { return null; }
  if (!cache.has(id)) {
    const el = makeEl('div');
    el.id = id;
    el.setAttribute('id', id);
    cache.set(id, el);
  }
  return cache.get(id);
}

const BOX_KEYS = ['scene', 'target', 'ecm', 'view', 'capture', 'output'];
const boxes = BOX_KEYS.map(function (key) {
  const el = byId('box-' + key) || makeEl('section');
  el.className = key === 'scene' ? 'box open' : 'box';
  el.setAttribute('data-box', key);
  el.setAttribute('aria-expanded', key === 'scene' ? 'true' : 'false');
  return el;
});

const document = {
  body: makeEl('body'),
  documentElement: makeEl('html'),
  getElementById: byId,
  createElement: makeEl,
  createTextNode: function (t) { const n = makeEl('#text'); n.textContent = String(t); return n; },
  querySelector: function (sel) {
    if (sel === '.box.open') { return boxes[0]; }
    return null;
  },
  querySelectorAll: function (sel) {
    if (sel === '.box') { return boxes.slice(0); }
    return [];
  },
  addEventListener: function () {},
  removeEventListener: function () {},
  onkeydown: null
};

function FakeXHR() {
  this.readyState = 0; this.status = 0; this.responseText = '';
  this.onreadystatechange = null; this.onerror = null; this.ontimeout = null;
}
FakeXHR.prototype.open = function () {};
FakeXHR.prototype.setRequestHeader = function () {};
FakeXHR.prototype.send = function () {};   /* 挂起：不回调 */
FakeXHR.prototype.abort = function () {};
FakeXHR.prototype.addEventListener = function () {};

function FakeImage() { this.width = 0; this.height = 0; this.onload = null; this.onerror = null; }
function FakeFileReader() { this.onload = null; this.onerror = null; }
FakeFileReader.prototype.readAsDataURL = function () {};

const store = {};
const localStorage = {
  getItem: function (k) { return Object.prototype.hasOwnProperty.call(store, k) ? store[k] : null; },
  setItem: function (k, v) { store[k] = String(v); },
  removeItem: function (k) { delete store[k]; }
};

const win = {
  document: document,
  console: console,
  navigator: { userAgent: 'node-smoke', clipboard: { writeText: function () {} } },
  location: { href: 'http://127.0.0.1:8899/', origin: 'http://127.0.0.1:8899' },
  localStorage: localStorage,
  XMLHttpRequest: FakeXHR,
  Image: FakeImage,
  FileReader: FakeFileReader,
  FormData: function () {},
  URL: { createObjectURL: function () { return ''; }, revokeObjectURL: function () {} },
  addEventListener: function () {}, removeEventListener: function () {},
  alert: function () {}, confirm: function () { return true; }, prompt: function () { return ''; },
  getComputedStyle: function () { return {}; },
  requestAnimationFrame: function () { return 0; },
  cancelAnimationFrame: function () {},
  /* 定时器只登记、不执行回调：init() 的同步路径才是本次冒烟的目标。 */
  setTimeout: function () { return 0; },
  clearTimeout: function () {},
  setInterval: function () { return 0; },
  clearInterval: function () {},
  encodeURIComponent: encodeURIComponent,
  decodeURIComponent: decodeURIComponent,
  parseInt: parseInt, parseFloat: parseFloat, isNaN: isNaN, isFinite: isFinite,
  Math: Math, JSON: JSON, Date: Date, Promise: Promise
};
win.window = win;
win.self = win;
win.globalThis = win;

const ctx = vm.createContext(win);
let failed = false;
try {
  vm.runInContext(code, ctx, { filename: 'studio/app.js', timeout: 10000 });
  console.log('INIT OK  |  app.js loaded, init() completed without throwing');
  console.log('         |  html ids=' + htmlIds.size + '  resolved=' + cache.size +
              '  boxes=' + boxes.length);
} catch (err) {
  failed = true;
  console.error('INIT FAIL | ' + (err && err.stack ? err.stack : String(err)));
}

/* 顺带检查关键函数确实被定义（防止拼错后被静默跳过）。 */
const required = [
  'openBox', 'renderFlow', 'boxStatus', 'renderNextAction', 'smartGenerate',
  'autoPrepare', 'focusField', 'focusFirstError', 'renderChecks', 'cancelAuto',
  'allErrors', 'localWarnings', 'errorLine', 'schedulePlan'
];
/* app.js 是 IIFE，内部函数不暴露到 vm 全局，改用源码静态检查。 */
const missing = required.filter(function (fn) {
  return code.indexOf('function ' + fn + '(') < 0;
});
if (missing.length) {
  failed = true;
  console.error('MISSING FUNCTIONS | ' + missing.join(', '));
} else {
  console.log('FUNCTIONS OK | ' + required.length + ' smart-flow functions defined');
}

/* 纯函数抽样实测：把连续的 issueKey/sameIssue/dedupe 从源码切出来单独跑，
   验证「前端长句 + 后端短句」确实合并为一条，而不同问题不会被误合并。 */
const start = code.indexOf('  function issueKey(');
const end = code.indexOf('  /* 硬错误', start);
if (start < 0 || end < 0) {
  failed = true;
  console.error('DEDUPE SOURCE | issueKey/sameIssue/dedupe snippet not found');
} else {
  const testCtx = {
    each: function (list, fn) { for (let i = 0; i < list.length; i++) { fn(list[i], i); } }
  };
  vm.createContext(testCtx);
  vm.runInContext(
    code.slice(start, end) + '\nthis.__api = { sameIssue: sameIssue, dedupe: dedupe };',
    testCtx
  );
  const api = testCtx.__api;
  const longMsg = '至少选一个模态：在「采集」箱里打开一个传感器开关。';
  const shortMsg = '至少选一个模态';
  const merged = api.dedupe([{ message: longMsg }, { message: shortMsg }]);
  const distinct = api.dedupe([
    { message: '窄视场宽必须是 64–4096 的整数' },
    { message: '宽视场宽超出后端允许范围' }
  ]);
  const ok = api.sameIssue(longMsg, shortMsg) &&
             merged.length === 1 && distinct.length === 2;
  console.log((ok ? 'DEDUPE OK  ' : 'DEDUPE FAIL') +
              '| long/short merged=' + merged.length +
              '  distinct kept=' + distinct.length);
  if (!ok) { failed = true; }
}
process.exit(failed ? 1 : 0);
