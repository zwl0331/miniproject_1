"""Score-free raw-row TabM device prototype at production and nested sizes.

This is a bounded engineering pilot, not an executable recipe_v2 adapter or its
acceptance gate. No outer labels enter a worker and no AUC is calculated. Each
device/population/repeat uses a fresh process; torch stays outside the parent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def padded_logits(torch, net, Z, device):
    import numpy as np
    net.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(Z), 256):
            part = Z[start:start + 256]
            padded = np.zeros((256, Z.shape[1]), dtype=np.float32)
            padded[:len(part)] = part
            values.append(net(torch.from_numpy(padded).to(device)).squeeze(-1)[:len(part)].cpu().double())
    return torch.cat(values)


def train(torch, p, Z, y, epochs, init_seed, batch_seed, device, Zv=None, yv=None):
    import numpy as np
    from songpop.phase2 import tabm as old
    net = old.network(torch, p, Z.shape[1], init_seed).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'],
                                  **old.ADAMW)
    generator = torch.Generator(device='cpu').manual_seed(batch_seed)
    inputs = torch.from_numpy(Z).to(device)
    labels = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    losses, updates = [], 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(inputs), p['batch_size']):
            indices = perm[start:start + p['batch_size']].to(device)
            logits = net(inputs[indices]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[indices, None].expand_as(logits))
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            optimizer.zero_grad()
            loss.backward()
            if not all(torch.isfinite(q.grad).all() for q in net.parameters() if q.grad is not None):
                raise ValueError('non-finite gradient')
            optimizer.step()
            updates += 1
        if Zv is not None:
            losses.append(old.ensemble_log_loss(torch, padded_logits(torch, net, Zv, device), yv))
            if old.stop_rule(losses, p['tolerance'], p['patience'])[0]:
                break
    return net, losses, updates


def worker(payload, device):
    import numpy as np
    from songpop.phase2 import tabm as old
    from songpop.phase2.raw_features import RawFeaturePipeline
    from songpop.phase2.raw_splits import partition, population
    from songpop.phase2.encoders import FitContext
    assert not any(k in sys.modules for k in ['lightgbm', 'xgboost', 'catboost'])
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device == 'cuda':
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    recipe, rows, y, groups = [payload[k] for k in ['recipe', 'rows', 'y', 'groups']]
    population(y, groups, rows.index)
    if recipe['family'] != 'tabm' or recipe['blocks'] or recipe['drop'] or recipe['estimator_seeds'] != [0]:
        raise ValueError('this pilot accepts only the single-seed historical scalar B recipe')
    ctx = FitContext(**payload['context'])
    p = dict(old.TabMClassifier(**recipe['params'], scaler=recipe['preprocessing']['scaler'], random_state=0).get_params())
    seeds = old.stage_seeds(ctx, 0)
    fit, valid = partition(y, groups, old.inner_split(y, groups, p['inner_folds'], seeds['inner_split']))
    stop_features = RawFeaturePipeline(recipe)
    Xs = stop_features.fit_transform(rows.iloc[fit], y[fit], groups[fit], ctx)
    Xv = stop_features.transform(rows.iloc[valid])
    stop_prep = old.TabMPreprocessor(p['scaler']).fit(Xs)
    Zs, Zv = stop_prep.transform(Xs), stop_prep.transform(Xv)
    _, losses, stop_updates = train(torch, p, Zs, y[fit], p['max_epochs'], seeds['stop_init'],
                                    seeds['stop_batches'], device, Zv, y[valid])
    stopped, epoch = old.stop_rule(losses, p['tolerance'], p['patience'])
    final_features = RawFeaturePipeline(recipe)
    X = final_features.fit_transform(rows, y, groups, ctx)
    prep = old.TabMPreprocessor(p['scaler']).fit(X)
    net, _, final_updates = train(torch, p, prep.transform(X), y, epoch, seeds['final_init'],
                                 seeds['final_batches'], device)
    predictions = {}
    for name, query in payload['queries'].items():
        Z = prep.transform(final_features.transform(query))
        predictions[name] = old.mean_probability(torch, padded_logits(torch, net, Z, device))
        if not np.isfinite(predictions[name]).all():
            raise ValueError('non-finite query probability')
    state = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        a = value.detach().cpu().numpy()
        state.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # The same legacy receipt permits direct CPU comparison on this first fold.
    receipt = {'policy': old.ITERATIONS, 'seeds': seeds,
        'inner_train_rows_sha256': old.rows_sha256(rows.index, fit),
        'inner_validation_rows_sha256': old.rows_sha256(rows.index, valid),
        'inner_train_rows': len(fit), 'inner_validation_rows': len(valid),
        'stopping_preprocessing_rows_sha256': stop_prep.fitted_rows_sha256_,
        'stopping_preprocessing_digest': stop_prep.digest_,
        'final_preprocessing_rows_sha256': prep.fitted_rows_sha256_, 'final_preprocessing_digest': prep.digest_,
        'encoded_columns': prep.output_columns(), 'epochs_evaluated': len(losses),
        'best_epoch': int(np.argmin(losses)) + 1, 'chosen_epoch': epoch,
        'stopped_by': 'patience' if stopped else 'max_epochs', 'validation_losses': losses,
        'stopping_updates': stop_updates, 'final_updates': final_updates,
        'versions': {'torch': str(torch.__version__), 'tabm': str(__import__('tabm').__version__)}, 'estimator_seed': 0}
    modules = {str(Path(m.__file__).resolve().relative_to(Path.cwd())): sha(m.__file__)
               for m in list(sys.modules.values()) if (getattr(m, '__file__', None) or '').endswith('.py')
               and Path(m.__file__).is_file() and Path.cwd() in Path(m.__file__).resolve().parents}
    return {'device': device, 'fitting_rows': len(rows), 'recipe': recipe, 'params': p,
        'receipt': receipt, 'prediction_sha256': {k: array_hash(v) for k, v in predictions.items()},
        'state_sha256': state.hexdigest(), 'fit_and_predict_seconds': elapsed, 'source_files': modules,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else None,
        'torch_runtime': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'subfits': {'feature_fits': 2, 'preprocessing_fits': 2, 'network_trainings': 2}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?')
    parser.add_argument('--worker', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if args.worker:
        payload = pickle.load(sys.stdin.buffer)
        out = os.fdopen(os.dup(1), 'wb')
        os.dup2(2, 1)
        pickle.dump(worker(payload, args.worker), out, protocol=pickle.HIGHEST_PROTOCOL)
        out.flush()
        return
    if not args.output:
        parser.error('output is required')
    import numpy as np
    from songpop.phase2.banks import load_bank_v2
    from songpop.phase2.runs import Context
    from songpop.phase2.encoders import inner_group_splits
    ctx = Context.load('.')
    bank_path = Path('artifacts/phase2/bank/p2_p25_tabm_anchor_seeds155-2026-1029_984d3248a783.npz')
    bank = load_bank_v2(bank_path)
    assert np.array_equal(bank.train_row_id, ctx.data.row_ids)
    assert np.array_equal(bank.y, ctx.data.y) and np.array_equal(bank.group_id, ctx.groups)
    full = np.flatnonzero(bank.fold_id[0] != 0)
    queries = {'validation': ctx.rows.iloc[np.flatnonzero(bank.fold_id[0] == 0)], 'test': ctx.test_rows}
    smaller, _ = inner_group_splits(ctx.data.y[full], ctx.groups[full], 20261101, n_splits=3)[0]
    policy_path = Path('configs/phase2/p2.9/device_policy.json')
    policy = json.loads(policy_path.read_text())
    env = dict(os.environ, **policy['worker_environment'], PYTHONDONTWRITEBYTECODE='1')
    records = {}
    for population_name, positions in [('production', full), ('nested_size', full[smaller])]:
        payload = {'recipe': bank.metadata['recipe'], 'rows': ctx.rows.iloc[positions],
            'y': ctx.data.y[positions], 'groups': ctx.groups[positions], 'queries': queries,
            'context': {'split_seed': 155, 'fold': 0}}
        records[population_name] = {}
        for device in ['cpu', 'cuda']:
            pair = []
            for repeat in range(2):
                started = time.perf_counter()
                result = subprocess.run([sys.executable, '-m', 'analysis.p29_tabm_device_pilot', '--worker', device],
                    input=pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), capture_output=True,
                    env=env, timeout=900)
                if result.returncode:
                    raise RuntimeError(result.stderr.decode())
                row = pickle.loads(result.stdout)
                row['process_seconds'] = time.perf_counter() - started
                row['stderr'] = result.stderr.decode()
                pair.append(row)
                print(f'{population_name} {device} repeat {repeat}: {row["process_seconds"]:.2f}s, '
                      f'epoch {row["receipt"]["chosen_epoch"]}; no AUC', flush=True)
            for key in ['receipt', 'prediction_sha256', 'state_sha256', 'source_files']:
                assert pair[0][key] == pair[1][key], (population_name, device, key)
            records[population_name][device] = pair
    old_probe = json.loads(Path('doc/llm_usage/2026-11-01-p29a-node/portability_cpu4_a.json').read_text())['results']['TabM_B']
    new = records['production']['cpu'][0]
    parity = {'validation': new['prediction_sha256']['validation'] == old_probe['prediction_sha256'],
              'test': new['prediction_sha256']['test'] == old_probe['test_prediction_sha256'],
              'receipt': new['receipt'] == old_probe['node_receipt']['tabm'][0]}
    result = {'schema': 'p29_tabm_device_pilot_v1', 'scope': 'one historical B recipe; score-free prototype, not adapter acceptance',
        'script_sha256': sha(__file__), 'policy_sha256': sha(policy_path), 'historical_bank_sha256': sha(bank_path),
        'lock_sha256': sha('configs/phase2/p2.9/install/linux-cuda.txt'), 'data_hashes': ctx.data.hashes,
        'old_new_cpu_first_fold_exact': parity, 'all_fresh_repeats_exact': True, 'records': records}
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    assert all(parity.values()), parity
    print('All fresh-process repeats exact; first-fold CPU predictions/receipt match the old adapter', flush=True)


if __name__ == '__main__':
    main()
