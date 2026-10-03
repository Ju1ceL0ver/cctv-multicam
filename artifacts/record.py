"""Claude's answers, kept apart from the owner's: they are not ground truth and never enter
the acceptance count. Fragments are named by their number on the sheet, candidates by letter."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STORE = HERE / 'answers_claude.json'


def record(day, number, kind, purity, foreign=(), edges=None, confidence='high', notes=''):
    q = json.load(open(HERE / ('queue_%s.json' % day)))
    v = q['visits'][number - 1]
    letters, letter = {}, ord('A')
    for side in ('after', 'before'):
        for c in v[side]['ask']:
            letters[chr(letter)] = c; letter += 1
    edges = edges or {}
    if set(edges) != set(letters):
        raise SystemExit('визит %s №%d: ответы на границы %s, а на листе %s'
                         % (day, number, sorted(edges), sorted(letters)))
    store = json.loads(STORE.read_text()) if STORE.exists() else {}
    store['%s/%s' % (day, v['id'])] = {
        'day': day, 'number': number, 'id': v['id'], 'nodes': v['nodes'],
        'kind': kind, 'purity': purity,
        'foreign': [v['fragments'][i - 1]['key'] for i in foreign],
        'edges': {letters[k]['node']: {'decision': d, 'side': letters[k]['side'], 'gap_s': letters[k]['gap_s'],
                                       'a': letters[k]['a'], 'b': letters[k]['b'],
                                       'evidence_a': letters[k]['evidence_a'], 'evidence_b': letters[k]['evidence_b']}
                  for k, d in edges.items()},
        'confidence': confidence, 'notes': notes, 'source': 'claude'}
    STORE.write_text(json.dumps(store, ensure_ascii=False, indent=1))
    return len(store)


if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        if line and not line.startswith('#'):
            a = json.loads(line)
            n = record(**a)
    print('записано визитов:', n)
