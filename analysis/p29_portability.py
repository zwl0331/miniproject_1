"""Two score-free historical CPU refits on the GPU node, first saved fold only.

This detects migration mismatches without declaring a complete bank replay or
using pilot AUC to select candidates. Historical source files and banks stay fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from songpop import artifacts
from songpop.phase2.banks import load_bank_v2
from songpop.phase2.encoders import FitContext
from songpop.phase2.engine import fit_recipe
from songpop.phase2.runs import Context, packages_for


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    args = parser.parse_args()
    ctx = Context.load('.')
    refs = json.loads(Path('configs/phase2/p2.9/references.json').read_text())
    ids = {'H': refs['H'], 'TabM_B': 'p2_p25_tabm_anchor_seeds155-2026-1029_984d3248a783'}
    results = {}
    for name, run_id in ids.items():
        path = Path('artifacts/phase2/bank', run_id + '.npz')
        bank = load_bank_v2(path)
        assert np.array_equal(bank.train_row_id, ctx.data.row_ids)
        assert np.array_equal(bank.y, ctx.data.y) and np.array_equal(bank.group_id, ctx.groups)
        recipe = bank.metadata['recipe']
        train, valid = np.flatnonzero(bank.fold_id[0] != 0), np.flatnonzero(bank.fold_id[0] == 0)
        started = time.perf_counter()
        fit = fit_recipe(recipe, ctx.rows.iloc[train], ctx.data.y[train], ctx.groups[train],
                         FitContext(int(bank.split_seeds[0]), 0))
        predictions = fit.predict_high(ctx.rows.iloc[valid])
        test_predictions = fit.predict_high(ctx.test_rows)
        errors = {'validation': float(np.max(np.abs(predictions - bank.oof[0, valid]))),
                  'test': float(np.max(np.abs(test_predictions - bank.test_by_fold[0, 0])))}
        changed = [p for p, expected in bank.metadata['source_manifest']['files'].items()
                   if not Path(p).exists() or sha(p) != expected]
        results[name] = {'run_id': run_id, 'bank_sha256': sha(path), 'split_seed': int(bank.split_seeds[0]),
            'fold': 0, 'fitting_rows': len(train), 'validation_rows': len(valid), 'maximum_absolute_error': errors,
            'matches_historical_within_1e_12': max(errors.values()) <= 1e-12,
            'node_receipt': fit.receipt, 'historical_receipt': bank.metadata['fit_receipts'][0],
            'packages': packages_for(recipe), 'historical_source_files_differ_from_current': changed,
            'prediction_sha256': hashlib.sha256(predictions.tobytes()).hexdigest(),
            'test_prediction_sha256': hashlib.sha256(test_predictions.tobytes()).hexdigest(),
            'elapsed_seconds': time.perf_counter() - started}
        print(f'{name}: first-fold historical max error {max(errors.values()):.6g}; no AUC computed', flush=True)
    result = {'schema': 'p29_portability_diagnostic_v1', 'script_sha256': sha(__file__),
        'environment': artifacts.environment('.'), 'data_hashes': ctx.data.hashes,
        'scope': 'first saved production fold only; no full replay or method acceptance', 'results': results}
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
