"""Bounded replay of H and E's six historical tree atoms on the Linux node.

Writes diagnostic predictions outside the production bank/log namespaces.
No changed recipe, new search, reweighting, or selection is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_atom(run_id, output):
    import numpy as np
    from songpop import artifacts
    from songpop.phase2.banks import load_bank_v2, source_manifest
    from songpop.phase2.engine import run_recipe_cv
    from songpop.phase2.runs import Context, packages_for

    ctx = Context.load('.')
    path = Path('artifacts/phase2/bank', run_id + '.npz')
    old = load_bank_v2(path)
    recipe = old.metadata['recipe']
    assert np.array_equal(old.train_row_id, ctx.data.row_ids)
    assert np.array_equal(old.fold_id, ctx.fold_id(old.split_seeds))
    manifest = source_manifest('.')
    started = time.perf_counter()
    bank = run_recipe_cv(recipe, ctx.rows, ctx.data.y, ctx.data.row_ids, ctx.groups,
        old.fold_id, old.split_seeds.tolist(), ctx.test_rows, ctx.data.test_id)
    assert source_manifest('.') == manifest, 'source changed during replay'
    errors = {name: float(np.max(np.abs(getattr(bank, name) - getattr(old, name))))
              for name in ['oof', 'test_by_fold', 'test_mean', 'fold_auc']}
    output = Path(output)
    with output.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, oof=bank.oof, test_by_fold=bank.test_by_fold,
                            test_mean=bank.test_mean, fold_auc=bank.fold_auc)
    record = {'run_id': run_id, 'historical_bank_sha256': sha(path), 'recipe': recipe,
        'source_manifest': manifest, 'historical_source_manifest': old.metadata['source_manifest'],
        'data_hashes': ctx.data.hashes, 'split_sha256': ctx.split_sha256,
        'environment': artifacts.environment('.'), 'packages': packages_for(recipe),
        'maximum_absolute_error': errors, 'matches_historical_within_1e_12': max(errors.values()) <= 1e-12,
        'elapsed_seconds': time.perf_counter() - started, 'fit_receipts': bank.metadata['fit_receipts'],
        'prediction_file': str(output.with_suffix('.npz')), 'prediction_sha256': sha(output.with_suffix('.npz'))}
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(f'{run_id}: max error {max(errors.values()):.6g}, {record["elapsed_seconds"]:.1f} seconds', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    parser.add_argument('--output')
    parser.add_argument('--directory', default='artifacts/phase2/p29/portability')
    args = parser.parse_args()
    if args.run_id:
        run_atom(args.run_id, args.output)
        return
    refs = json.loads(Path('configs/phase2/p2.9/references.json').read_text())
    ids = [refs['H']] + [m['run_id'] for m in refs['E_members']]
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1')
    def launch(run_id):
        output = directory / (run_id + '.json')
        subprocess.run([sys.executable, '-m', 'analysis.p29_replay', '--run-id', run_id, '--output', str(output)],
                       env=env, check=True, timeout=1800)
        return json.loads(output.read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(launch, ids))
    import numpy as np
    from songpop.phase2.banks import load_bank_v2
    files = {r['run_id']: r['prediction_file'] for r in results}
    old = load_bank_v2(Path('artifacts/phase2/bank', refs['E'] + '.npz'))
    ensemble = {}
    for name in ['oof', 'test_by_fold', 'test_mean']:
        total = np.zeros_like(getattr(old, name))
        for member in refs['E_members']:
            with np.load(files[member['run_id']]) as values:
                total += member['weight'] * values[name]
        ensemble[name] = float(np.max(np.abs(total - getattr(old, name))))
    summary = {'schema': 'p29_tree_portability_v1', 'scope': 'historical tree recipes only; no candidate selection',
        'workers': 4, 'threads_per_worker': 4, 'atom_count': len(results),
        'all_atoms_within_1e_12': all(r['matches_historical_within_1e_12'] for r in results),
        'E_maximum_absolute_error': ensemble,
        'records': {r['run_id']: sha(directory / (r['run_id'] + '.json')) for r in results}}
    with (directory / 'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
