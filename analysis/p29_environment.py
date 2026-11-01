"""P2.9 Linux wheel locks and exact before/after dependency inventories.

Bootstrap with Python 3.12 and pip in .venv. Resolve with pip --dry-run --report,
then use `lock`, install its output with --no-deps --require-hashes, and `verify`.
Historical Mac locks and protected inventories are read-only inputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from urllib.parse import urlsplit

from packaging.utils import canonicalize_name

from scripts.install_locks import additions


def inventory():
    result = {}
    for item in importlib.metadata.distributions():
        name = canonicalize_name(item.metadata['Name'])
        if name in result:
            raise ValueError(f'duplicate installed distribution: {name}')
        result[name] = item.version
    return dict(sorted(result.items()))


def write_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(text)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock(report_path, output, constraints_path):
    report = json.loads(Path(report_path).read_text())
    resolved = additions(report)
    before = inventory()
    expected = dict(before)
    expected.update({name: version for name, (version, _, _) in resolved.items()})
    constraints = {}
    for line in Path(constraints_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            name, version = line.split('==')
            constraints[canonicalize_name(name)] = version
    changed = {name: [version, expected[name]] for name, version in constraints.items()
               if name in expected and expected[name] != version}
    if changed:
        raise ValueError(f'historical version constraints would change: {changed}')
    for name, (_, url, sha) in resolved.items():
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname != 'files.pythonhosted.org':
            raise ValueError(f'{name}: unexpected wheel origin {parsed.hostname}')
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError(f'{name}: invalid wheel hash')
    lines = ['# P2.9 Linux CPython 3.12 route. Generated from a saved pip dry-run report.',
             f'# Report: {report_path}; sha256 {digest(report_path)}',
             '# Install with .venv/bin/python -m pip install --no-deps --require-hashes -r <this file>',
             '--no-index', '--only-binary=:all:']
    lines.extend(f'{name} @ {url} --hash=sha256:{sha}'
                 for name, (_, url, sha) in sorted(resolved.items()))
    write_new(output, '\n'.join(lines) + '\n')
    receipt = {'schema': 'p29_linux_install_v1', 'before': before, 'expected_after': expected,
               'report': str(report_path), 'report_sha256': digest(report_path),
               'lock': str(output), 'lock_sha256': digest(output),
               'historical_constraints': str(constraints_path),
               'historical_constraints_sha256': digest(constraints_path),
               'historical_constraints_not_required_on_linux': sorted(set(constraints) - set(expected))}
    write_new(str(output) + '.json', json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(f'{len(resolved)} locked wheels; {len(expected)} expected installed distributions; historical versions preserved')


def verify(receipt_path, output):
    receipt = json.loads(Path(receipt_path).read_text())
    for key in ['report', 'lock', 'historical_constraints']:
        if digest(receipt[key]) != receipt[key + '_sha256']:
            raise ValueError(f'{key} changed after lock creation')
    actual, expected = inventory(), receipt['expected_after']
    differences = {name: [expected.get(name), actual.get(name)] for name in sorted(set(actual) | set(expected))
                   if actual.get(name) != expected.get(name)}
    if differences:
        raise ValueError(f'installed environment differs from the exact locked transaction: {differences}')
    write_new(output, json.dumps({'ok': True, 'receipt_sha256': digest(receipt_path),
                                 'installed': actual}, indent=2, sort_keys=True) + '\n')
    print(f'exact inventory verified: {len(actual)} distributions')


def licenses(output):
    records = {}
    for name in ['torch', 'tabm', 'rtdl-num-embeddings', 'rtdl-revisiting-models',
                 'numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost', 'interpret-core']:
        dist = importlib.metadata.distribution(name)
        files = {}
        for entry in dist.files:
            path = Path(dist.locate_file(entry))
            if 'license' in str(entry).lower() and path.is_file():
                lines = [line.strip() for line in path.read_text(errors='replace').splitlines() if line.strip()]
                files[str(entry)] = {'sha256': digest(path), 'first_nonempty_lines': lines[:3]}
        metadata = dist.metadata
        records[name] = {'version': dist.version, 'license': metadata.get('License'),
            'license_expression': metadata.get('License-Expression'),
            'license_classifiers': [c for c in metadata.get_all('Classifier', []) if c.startswith('License ::')],
            'project_urls': metadata.get_all('Project-URL', []), 'license_files': files}
    write_new(output, json.dumps(records, indent=2, sort_keys=True) + '\n')
    print(f'license identifiers, source URLs and file hashes recorded for {len(records)} modeling distributions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('lock')
    build.add_argument('report')
    build.add_argument('output')
    build.add_argument('--constraints', default='artifacts/phase2/install/constraints_resolved.txt')
    check = sub.add_parser('verify')
    check.add_argument('receipt')
    check.add_argument('output')
    license_cmd = sub.add_parser('licenses')
    license_cmd.add_argument('output')
    args = parser.parse_args()
    if args.command == 'lock':
        lock(args.report, args.output, args.constraints)
    elif args.command == 'verify':
        verify(args.receipt, args.output)
    else:
        licenses(args.output)


if __name__ == '__main__':
    main()
