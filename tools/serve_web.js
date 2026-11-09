const http = require('http'), fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.json': 'application/json; charset=utf-8',
  '.ico': 'image/x-icon', '.svg': 'image/svg+xml', '.md': 'text/markdown; charset=utf-8' };
http.createServer((req, res) => {
  let rel;
  try { rel = decodeURIComponent(req.url.split('?')[0]); }
  catch (e) { res.writeHead(400); res.end('bad request'); return; }
  if (rel === '/') rel = '/index.html';
  // resolve + separator check: a bare startsWith(root) also accepts a sibling
  // directory whose name merely begins with the same prefix.
  const file = path.resolve(root, '.' + rel);
  if (file !== root && !file.startsWith(root + path.sep)) {
    res.writeHead(403); res.end('forbidden'); return;
  }
  fs.readFile(file, (err, buf) => {
    if (err) { res.writeHead(404, { 'content-type': 'text/plain' }); res.end('not found: ' + rel); return; }
    res.writeHead(200, { 'content-type': mime[path.extname(file)] || 'application/octet-stream',
      'cache-control': 'no-cache' });
    res.end(buf);
  });
}).listen(Number(process.env.PORT) || 18080, '127.0.0.1', () => {
  console.log('serving http://127.0.0.1:' + (Number(process.env.PORT) || 18080));
});
