const fs = require('fs');
// Minimal static check that the app's expected data shape matches the manifest.
const m = JSON.parse(fs.readFileSync('data/manifest.json', 'utf8'));
// Legacy six-modality contract for the original air/maritime scenes.
const LEGACY = ['optical', 'mwir', 'lwir', 'radar_ra', 'lidar_range', 'lidar_intensity'];

// A scene may declare its own non-empty list of modality keys (the v3 jam
// scenes ship only optical/lwir/radar). Honour that list; otherwise keep the
// legacy six so the original six scenes stay green.
function sceneMods(s) {
  var raw = s.mods;
  if (Array.isArray(raw) && raw.length &&
      raw.every(function (x) { return typeof x === 'string' && x; })) {
    return raw;
  }
  return LEGACY;
}

let problems = [];
m.scenes.forEach(s => {
  if (!s.frames.length) problems.push(s.id + ': no frames');
  const mods = sceneMods(s);
  s.frames.forEach(f => {
    mods.forEach(k => {
      if (!f.mods || !f.mods[k]) problems.push(s.id + ' f' + f.i + ': missing ' + k);
      else if (!fs.existsSync('data/' + f.mods[k].img)) problems.push('missing file ' + f.mods[k].img);
    });
    if (f.cloud) {
      ['x', 'y', 'z', 'i', 'o'].forEach(k => {
        if (!Array.isArray(f.cloud[k])) problems.push(s.id + ' f' + f.i + ': cloud.' + k + ' not array');
      });
      const n = f.cloud.x.length;
      if (f.cloud.y.length !== n || f.cloud.z.length !== n) problems.push(s.id + ' f' + f.i + ': cloud array length mismatch');
    }
    if (f.radar && Array.isArray(f.radar.pts)) f.radar.pts.forEach(p => { if (p.length !== 5) problems.push(s.id + ' f' + f.i + ': radar pt len ' + p.length); });
  });
});
console.log(problems.length ? 'PROBLEMS:\n' + problems.slice(0, 20).join('\n') : 'DATA SHAPE OK (' + m.scenes.length + ' scenes)');
