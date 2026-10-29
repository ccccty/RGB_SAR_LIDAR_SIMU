const http = require('http'), fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.json': 'application/json; charset=utf-8',
  '.ico': 'image/x-icon', '.svg': 'image/svg+xml', '.md': 'text/markdown; charset=utf-8' };
http.createServer((req, res) => {
  let rel = decodeURIComponent(req.url.split('?')[0]);
  if (rel === '/') rel = '/index.html';
  const file = path.join(root, rel);
  if (!file.startsWith(root)) { res.writeHead(403); res.end('forbidden'); return; }
  fs.readFile(file, (err, buf) => {
    if (err) { res.writeHead(404, { 'content-type': 'text/plain' }); res.end('not found: ' + rel); return; }
    res.writeHead(200, { 'content-type': mime[path.extname(file)] || 'application/octet-stream',
      'cache-control': 'no-cache' });
    res.end(buf);
  });
}).listen(Number(process.env.PORT) || 18080, '127.0.0.1', () => {
  console.log('serving http://127.0.0.1:' + (Number(process.env.PORT) || 18080));
});
