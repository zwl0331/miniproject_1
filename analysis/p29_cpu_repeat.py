"""Check the recorded first-fold CPU probes from two fresh node processes."""
import hashlib
import json
from pathlib import Path


def main():
    directory = Path('doc/llm_usage/2026-11-01-p29a-node')
    paths = [directory / f'portability_cpu4_{suffix}.json' for suffix in ['a', 'b']]
    a, b = [json.loads(path.read_text()) for path in paths]
    assert a['script_sha256'] == b['script_sha256']
    assert a['data_hashes'] == b['data_hashes']
    checks = {}
    for name in ['H', 'TabM_B']:
        checks[name] = {key: a['results'][name][key] == b['results'][name][key]
                       for key in ['prediction_sha256', 'test_prediction_sha256', 'node_receipt', 'packages']}
        assert all(checks[name].values()), checks[name]
    for result in [a, b]:
        for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
            assert result['environment']['threads'][name] == '4'
    output = {'ok': True, 'scope': 'first production fold only; old CPU adapter on the node, no new adapter gate',
              'checks': checks,
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    with (directory / 'cpu_repeat_verified.json').open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('CPU H and TabM first-fold predictions and full node receipts identical in two fresh four-thread processes')


if __name__ == '__main__':
    main()
