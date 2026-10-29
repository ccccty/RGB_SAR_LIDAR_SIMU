"""Write REPORT.md summarising every simkit run under a directory."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FENCE = chr(96) * 3


def main():
    out_root = sys.argv[1]
    lines = ['# simkit simulation report', '']
    for name in sorted(os.listdir(out_root)):
        run = os.path.join(out_root, name)
        stats_p = os.path.join(run, 'stats.json')
        if not os.path.isdir(run) or not os.path.exists(stats_p):
            continue
        with open(stats_p, 'r', encoding='utf-8') as fh:
            st = json.load(fh)
        lines.append('## %s' % name)
        lines.append('')
        lines.append(FENCE + 'json')
        lines.append(json.dumps(st, ensure_ascii=False, indent=1))
        lines.append(FENCE)
        lines.append('')
    p = os.path.join(out_root, 'REPORT.md')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(chr(10).join(lines))
    print('wrote', p)


main()
