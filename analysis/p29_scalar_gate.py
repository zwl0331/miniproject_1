"""Score-free, fresh-process, full 15-fold scalar adapter acceptance gate.

Each timing case first repeats the old adapter in two new processes. Only after
that exact comparison passes does it fit the raw adapter. Three cases may run
concurrently, each with four CPU threads. This writes diagnostics, never banks
or search records. Run from a fixed source snapshot while development proceeds.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [155, 2026, 1029]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    paths = sorted(set(ROOT.glob('songpop/**/*.py')) | set(ROOT.glob('configs/phase2/p2.9/**/*.json')) |
                   set(ROOT.glob('configs/phase2/p2.9/install/*')) |
                   {ROOT / 'analysis/p29_scalar_gate.py', ROOT / 'configs/phase2/p2.5/timing.json',
                    ROOT / 'data/train.csv', ROOT / 'data/test.csv', ROOT / 'artifacts/splits/production_v1.csv'})
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def fit_case(case, route, output):
    from songpop.phase2 import engine, neural_backend, raw_neural, raw_recipes, tabm
    from songpop.phase2.encoders import FitContext
    from songpop.phase2.runs import Context

    started = time.monotonic()
    source = inventory()
    cases = json.loads((ROOT / 'configs/phase2/p2.5/timing.json').read_text())['cases']
    legacy = next(c['recipe'] for c in cases if c['case'] == case)
    recipe = legacy if route == 'old' else raw_recipes.from_legacy(legacy)
    ctx = Context.load(ROOT)
    rows, query, y, groups = ctx.rows, ctx.test_rows, ctx.data.y, ctx.groups
    folds = ctx.fold_id(SEEDS)
    oof = np.full(folds.shape, np.nan)
    test = np.full((3, 5, len(query)), np.nan)
    count = np.zeros(folds.shape, dtype=int)
    receipts, states, inputs, timings = [], [], [], []
    for r, seed in enumerate(SEEDS):
        for k in range(5):
            t0 = time.monotonic()
            tr, va = np.flatnonzero(folds[r] != k), np.flatnonzero(folds[r] == k)
            fit = engine.fit_recipe if route == 'old' else raw_neural.fit_raw
            fitted = fit(recipe, rows.iloc[tr], y[tr], groups[tr], FitContext(seed, k))
            oof[r, va] = fitted.predict_high(rows.iloc[va])
            each = fitted.predict_each(query)
            Z = fitted.model_inputs(query)
            for q in each:
                assert max((float(np.ptp(q[c])) for c in engine.input_classes(Z)), default=0) <= 1e-12
            test[r, k] = np.mean(each, axis=0)
            count[r, va] += 1
            receipts.append({key: fitted.receipt[key] for key in ('split_seed', 'fold', 'fitting_rows_sha256', 'inner', 'tabm')})
            st = [est.state_ for est in fitted.estimators] if route == 'old' else fitted.states
            states.append([neural_backend.state_digest(s) for s in st])
            inputs.append(neural_backend.array_digest(Z))
            timings.append(time.monotonic() - t0)
            print(json.dumps({'case': case, 'route': route, 'split_seed': seed, 'fold': k,
                              'seconds': timings[-1], 'chosen_epoch': receipts[-1]['tabm'][0]['chosen_epoch']}), flush=True)
    assert np.all(count == 1) and np.isfinite(oof).all() and np.isfinite(test).all()
    assert source == inventory(), 'gate source changed while fitting'
    assert 'torch' not in sys.modules, 'parent imported Torch'
    np.savez_compressed(output.with_suffix('.npz'), oof=oof, test_by_fold=test, test_mean=test.mean(axis=(0, 1)),
                        fold_id=folds, row_id=ctx.data.row_ids, group_id=groups, test_id=ctx.data.test_id)
    save_json(output.with_suffix('.json'), {'schema': 'p29_scalar_gate_v1', 'case': case, 'route': route,
              'recipe': recipe, 'source_sha256': source, 'split_sha256': ctx.split_sha256,
              'data_hashes': ctx.data.hashes, 'receipts': receipts, 'state_sha256': states,
              'query_input_sha256': inputs, 'fit_seconds': timings,
              'elapsed_seconds': time.monotonic() - started, 'array_file_sha256': digest(output.with_suffix('.npz')),
              'worker': raw_neural.Worker.for_device('cpu').hello if route == 'new' else None})
    raw_neural.Worker.close_all()
    tabm.Worker.close()


def compare(a, b, *, exact):
    ma, mb = (json.loads(p.with_suffix('.json').read_text()) for p in (a, b))
    for key in ('source_sha256', 'split_sha256', 'data_hashes', 'receipts', 'state_sha256', 'query_input_sha256'):
        if ma[key] != mb[key]:
            raise AssertionError(f'{a.name}/{b.name}: {key} differs')
    differences = {}
    with np.load(a.with_suffix('.npz'), allow_pickle=False) as x, np.load(b.with_suffix('.npz'), allow_pickle=False) as y:
        assert set(x.files) == set(y.files)
        for key in x.files:
            difference = float(np.max(np.abs(x[key] - y[key])))
            differences[key] = difference
            if exact or key in ('fold_id', 'row_id', 'group_id', 'test_id'):
                np.testing.assert_array_equal(x[key], y[key])
            else:
                np.testing.assert_allclose(x[key], y[key], rtol=0, atol=1e-12)
    # Even allowed nonzero prediction differences are reported for investigation.
    return {'arrays_max_abs': differences, 'receipts_equal': True, 'states_equal': True, 'inputs_equal': True}


def run_case(case, output):
    paths = [output / f'{case}_{name}' for name in ('old0', 'old1', 'new')]
    declared = json.loads((ROOT / 'configs/phase2/p2.9/device_policy.json').read_text())
    env = dict(os.environ, **declared['worker_environment'], PYTHONDONTWRITEBYTECODE='1')
    result = {'case': case, 'status': 'running'}
    for i, path in enumerate(paths):
        if path.with_suffix('.json').exists() or path.with_suffix('.npz').exists():
            raise FileExistsError(f'gate output already exists: {path}')
        if i == 2:
            result['old_fresh_process_repeat'] = compare(paths[0], paths[1], exact=True)
        command = [sys.executable, '-u', '-m', 'analysis.p29_scalar_gate', '--case', case,
                   '--route', 'new' if i == 2 else 'old', '--output', str(path)]
        with path.with_suffix('.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
    result['old_new'] = compare(paths[0], paths[2], exact=False)
    result['status'] = 'passed'
    save_json(output / f'{case}_comparison.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=['light', 'representative', 'expensive'])
    ap.add_argument('--route', choices=['old', 'new'])
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.case:
        if args.route is None:
            ap.error('--case requires --route')
        fit_case(args.case, args.route, args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output / 'launch.json', {'pid': os.getpid(), 'source_root': str(ROOT),
                  'source_sha256': inventory(), 'cpu_workers': 3, 'threads_per_worker': 4,
                  'purpose': 'correctness gate, no AUC computed', 'processes': 9, 'outer_fits': 135})
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda case: run_case(case, args.output), ['light', 'representative', 'expensive']))
        save_json(args.output / 'verification.json', {'status': 'passed', 'cases': results})


if __name__ == '__main__':
    main()
