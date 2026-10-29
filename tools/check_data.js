const fs = require('fs');
// Minimal static check that the app's expected data shape matches the manifest.
const m = JSON.parse(fs.readFileSync('data/manifest.json', 'utf8'));
let problems = [];
m.scenes.forEach(s => {
  if (!s.frames.length) problems.push(s.id + ': no frames');
  s.frames.forEach(f => {
    ['optical', 'mwir', 'lwir', 'radar_ra', 'lidar_range', 'lidar_intensity'].forEach(k => {
      if (!f.mods[k]) problems.push(s.id + ' f' + f.i + ': missing ' + k);
      else if (!fs.existsSync('data/' + f.mods[k].img)) problems.push('missing file ' + f.mods[k].img);
    });
    if (f.cloud) {
      ['x', 'y', 'z', 'i', 'o'].forEach(k => {
        if (!Array.isArray(f.cloud[k])) problems.push(s.id + ' f' + f.i + ': cloud.' + k + ' not array');
      });
      const n = f.cloud.x.length;
      if (f.cloud.y.length !== n || f.cloud.z.length !== n) problems.push(s.id + ' f' + f.i + ': cloud array length mismatch');
    }
    if (f.radar) f.radar.pts.forEach(p => { if (p.length !== 5) problems.push(s.id + ' f' + f.i + ': radar pt len ' + p.length); });
  });
});
console.log(problems.length ? 'PROBLEMS:\n' + problems.slice(0, 20).join('\n') : 'DATA SHAPE OK (' + m.scenes.length + ' scenes)');