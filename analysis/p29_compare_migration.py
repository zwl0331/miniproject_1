"""Compare two node replays and historical banks, without selecting a model."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from songpop.phase2.banks import load_bank_v2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    refs = json.loads(Path('configs/phase2/p2.9/references.json').read_text())
    first = Path('artifacts/phase2/p29/portability')
    repeat = Path('artifacts/phase2/p29/portability_repeat')
    rows = []
    node = {}
    ids = [refs['H']] + [m['run_id'] for m in refs['E_members']]
    for run_id in ids:
        a, b = first / (run_id + '.npz'), repeat / (run_id + '.npz')
        with np.load(a) as x, np.load(b) as y:
            errors = {name: float(np.max(np.abs(x[name] - y[name]))) for name in x.files}
            byte_equal = all(np.array_equal(x[name], y[name]) for name in x.files)
            node[run_id] = {name: x[name].copy() for name in x.files}
        receipt = json.loads((first / (run_id + '.json')).read_text())
        old = load_bank_v2(Path('artifacts/phase2/bank', run_id + '.npz'))
        rows.append({'run_id': run_id, 'family': receipt['recipe']['family'],
            'first_sha256': sha(a), 'repeat_sha256': sha(b), 'node_repeat_byte_equal': byte_equal,
            'node_repeat_max_absolute_error': errors,
            'historical_max_absolute_error': receipt['maximum_absolute_error'],
            'historical_mean_fold_auc': float(old.fold_auc.mean()),
            'node_mean_fold_auc': float(node[run_id]['fold_auc'].mean())})
    old_e = load_bank_v2(Path('artifacts/phase2/bank', refs['E'] + '.npz'))
    e = sum(m['weight'] * node[m['run_id']]['oof'] for m in refs['E_members'])
    auc = np.array([[roc_auc_score(old_e.y[old_e.fold_id[r] == k], e[r, old_e.fold_id[r] == k])
                     for k in range(5)] for r in range(3)])
    result = {'schema': 'p29_migration_comparison_v1', 'script_sha256': sha(__file__),
        'scope': 'same unchanged historical recipes; migration diagnostic, not new candidate selection',
        'all_node_repeats_byte_equal': all(r['node_repeat_byte_equal'] for r in rows), 'atoms': rows,
        'E': {'historical_mean_fold_auc': float(old_e.fold_auc.mean()),
              'node_diagnostic_mean_fold_auc': float(auc.mean()),
              'node_diagnostic_repeat_means': auc.mean(axis=1).tolist(),
              'node_diagnostic_fold_auc': auc.tolist(),
              'mean_fold_auc_difference': float(auc.mean() - old_e.fold_auc.mean()),
              'max_oof_probability_difference': float(np.max(np.abs(e - old_e.oof)))}}
    output = first.parent / 'migration_comparison.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
