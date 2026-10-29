// Shared by the probe scripts: serve the repo root on an ephemeral loopback
// port so a verification run never depends on a server someone else started
// (and never collides with one).
const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.png': 'image/png',
  '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon', '.md': 'text/markdown; charset=utf-8'
};

function serve(root) {
  return new Promise((resolve, reject) => {
    const s = http.createServer((req, res) => {
      let rel = decodeURIComponent(req.url.split('?')[0]);
      if (rel === '/') rel = '/index.html';
      const file = path.join(root || ROOT, rel);
      if (!file.startsWith(root || ROOT)) { res.writeHead(403); res.end('forbidden'); return; }
      fs.readFile(file, (err, buf) => {
        if (err) { res.writeHead(404, { 'content-type': 'text/plain' }); res.end('not found: ' + rel); return; }
        res.writeHead(200, {
          'content-type': MIME[path.extname(file)] || 'application/octet-stream',
          'cache-control': 'no-cache'
        });
        res.end(buf);
      });
    });
    s.on('error', reject);
    s.listen(0, '127.0.0.1', () => resolve({ server: s, port: s.address().port }));
  });
}

module.exports = { serve, ROOT };
