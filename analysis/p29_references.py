"""Bind P2.9's immutable E/H references to the historical banks and inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from songpop.phase2.banks import load_bank_v2
from songpop.phase2.recipes import recipe_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    args = parser.parse_args()
    menu = json.loads(Path('configs/phase2/menu.json').read_text())
    candidates = {c['id']: c for c in menu['candidates']}
    e = json.loads(Path('artifacts/phase2/selections/p26b_blend_eq_all.json').read_text())['run_id']
    h = candidates['atom_p26_anchor']['run_id']
    members = candidates['blend_eq_all']['members']
    ids = [e, h] + [m['run_id'] for m in members]
    banks, records = {}, {}
    for run_id in ids:
        path = Path('artifacts/phase2/bank', run_id + '.npz')
        bank = load_bank_v2(path)
        assert bank.metadata['run_id'] == run_id
        banks[run_id] = bank
        md = bank.metadata
        source = md['source_manifest']
        record = {'path': str(path), 'sha256': sha(path), 'config_sha256': md['config_sha256'],
                  'source_manifest': source, 'source_files_differ_from_current': [p for p, digest in
                   source['files'].items() if not Path(p).exists() or sha(p) != digest],
                  'data_hashes': md['config'].get('data_hashes'), 'split_seeds': bank.split_seeds.tolist(),
                  'mean_fold_auc': float(bank.fold_auc.mean())}
        if 'recipe' in md:
            record.update(recipe=md['recipe'], recipe_sha256=recipe_hash(md['recipe']))
            assert md['recipe_sha256'] == record['recipe_sha256']
        records[run_id] = record
    reference = banks[e]
    for bank in banks.values():
        for field in ['train_row_id', 'y', 'group_id', 'split_seeds', 'fold_id', 'test_id']:
            assert np.array_equal(getattr(bank, field), getattr(reference, field)), field
    blend_errors = {}
    for field in ['oof', 'test_by_fold', 'test_mean']:
        rebuilt = sum(m['weight'] * getattr(banks[m['run_id']], field) for m in members)
        error = float(np.max(np.abs(rebuilt - getattr(reference, field))))
        assert error <= 1e-12, (field, error)
        blend_errors[field] = error
    inputs = ['data/train.csv', 'data/test.csv', 'data/sample_submission.csv',
              'artifacts/splits/production_v1.csv', 'artifacts/phase2/splits/nested_outer_v1.csv',
              'configs/phase2/menu.json', 'configs/phase2/p2.6/procedures.json',
              'artifacts/phase2/p27/decision.json', 'artifacts/phase2/p27/verification.json']
    result = {'schema': 'p29_references_v1', 'E': e, 'H': h, 'E_members': members,
              'input_sha256': {p: sha(p) for p in inputs}, 'banks': records,
              'E_reconstruction_max_absolute_error': blend_errors,
              'scope': 'historical artifacts validated and hashed; no node refit or new performance claim'}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(f'{len(banks)} historical banks bound; E reconstruction max error {max(blend_errors.values()):.3g}')


if __name__ == '__main__':
    main()
