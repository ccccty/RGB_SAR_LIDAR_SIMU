// Read the stored GitHub credential (never printed) and drive the REST API.
const { execFileSync } = require('child_process');

function cred() {
  const out = execFileSync('git', ['credential', 'fill'], {
    input: 'protocol=https\nhost=github.com\n\n',
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
    encoding: 'utf8'
  });
  const kv = {};
  for (const line of out.split(/\r?\n/)) {
    const i = line.indexOf('=');
    if (i > 0) kv[line.slice(0, i)] = line.slice(i + 1);
  }
  return kv;
}

async function api(method, path, body) {
  const c = cred();
  const r = await fetch('https://api.github.com' + path, {
    method,
    headers: {
      authorization: 'token ' + c.password,
      accept: 'application/vnd.github+json',
      'user-agent': 'simkit-ci',
      'content-type': 'application/json'
    },
    body: body ? JSON.stringify(body) : undefined
  });
  let data = null;
  const text = await r.text();
  try { data = JSON.parse(text); } catch (e) { data = text.slice(0, 300); }
  return { status: r.status, data };
}

(async () => {
  const who = await api('GET', '/user');
  console.log('auth:', who.status, who.data && who.data.login);
  const scopes = who.status === 200 ? 'ok' : JSON.stringify(who.data).slice(0, 200);
  console.log('  ', scopes);

  const repo = await api('GET', '/repos/ccccty/RGB_SAR_LIDAR_SIMU');
  console.log('repo:', repo.status, repo.data.full_name,
    'private=' + repo.data.private, 'default=' + repo.data.default_branch,
    'pages=' + (repo.data.has_pages));

  let pages = await api('GET', '/repos/ccccty/RGB_SAR_LIDAR_SIMU/pages');
  console.log('pages get:', pages.status, JSON.stringify(pages.data).slice(0, 200));

  if (pages.status === 404) {
    const created = await api('POST', '/repos/ccccty/RGB_SAR_LIDAR_SIMU/pages',
      { source: { branch: 'main', path: '/' } });
    console.log('pages create:', created.status, JSON.stringify(created.data).slice(0, 300));
    pages = await api('GET', '/repos/ccccty/RGB_SAR_LIDAR_SIMU/pages');
    console.log('pages after:', pages.status, JSON.stringify(pages.data).slice(0, 300));
  } else if (pages.status === 200) {
    const put = await api('PUT', '/repos/ccccty/RGB_SAR_LIDAR_SIMU/pages', { source: { branch: 'main', path: '/' } });
    console.log('pages update:', put.status, JSON.stringify(put.data).slice(0, 200));
  }
})();
