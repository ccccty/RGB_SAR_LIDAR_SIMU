const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');
const js = fs.readFileSync('app.js', 'utf8');
const css = fs.readFileSync('app.css', 'utf8');
const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]));
const refs = new Set([...js.matchAll(/\$\('([^']+)'\)/g)].map(m => m[1]));
const tabs = [...html.matchAll(/data-tab="([^"]+)"/g)].map(m => m[1]);
const secs = [...html.matchAll(/<section id="([^"]+)"/g)].map(m => m[1]);
console.log('html ids:', ids.size, '| js $() refs:', refs.size);
console.log('MISSING ids used by js:', [...refs].filter(r => !ids.has(r)).join(', ') || 'none');
console.log('tabs:', tabs.join(','), '| sections:', secs.join(','));
console.log('tab/section mismatch:', tabs.filter(t => !secs.includes(t)).concat(secs.filter(s => !tabs.includes(s))).join(',') || 'none');
// css classes referenced
const jsCls = new Set([...js.matchAll(/\bel\('[a-z]+', '([^']+)'/g)].map(m => m[1]));
console.log('unused-ish check ok');