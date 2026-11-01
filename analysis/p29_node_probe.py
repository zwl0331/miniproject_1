"""Score-free CUDA architecture smoke test; synthetic fixtures, two fresh processes.

No project data, validation score or candidate-selection decision is involved.
This does not replace the full-size and nested-size adapter timing pilots.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child():
    import numpy as np
    import rtdl_num_embeddings as emb
    import rtdl_revisiting_models as rtdl
    import tabm
    import torch

    assert torch.cuda.is_available(), 'CUDA unavailable in the pinned environment'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(155)
    cpu_x = torch.randn(128, 12)
    x, y = cpu_x.cuda(), (cpu_x[:, 0] + 0.5 * cpu_x[:, 1] > 0).float().cuda()
    bins, probes = emb.compute_bins(cpu_x, n_bins=8), {}
    for kind in ['tabm_scalar', 'tabm_ple', 'tabm_periodic', 'mlp', 'resnet', 'ft_transformer']:
        torch.manual_seed(20261031)
        if kind.startswith('tabm'):
            embedding = None
            if kind == 'tabm_ple':
                embedding = emb.PiecewiseLinearEmbeddings(bins, 8, activation=False, version='B')
            elif kind == 'tabm_periodic':
                embedding = emb.PeriodicEmbeddings(12, 8, n_frequencies=8,
                    frequency_init_scale=0.01, activation=False, lite=False)
            net = tabm.TabM.make(n_num_features=12, cat_cardinalities=None, d_out=1,
                num_embeddings=embedding, n_blocks=2, d_block=64, dropout=0.1, k=16,
                arch_type='tabm', activation='ReLU',
                start_scaling_init='random-signs' if embedding is None else 'normal')
        elif kind == 'mlp':
            net = rtdl.MLP(d_in=12, d_out=1, n_blocks=2, d_block=64, dropout=0.1)
        elif kind == 'resnet':
            net = rtdl.ResNet(d_in=12, d_out=1, n_blocks=2, d_block=64,
                d_hidden_multiplier=2.0, dropout1=0.1, dropout2=0.0)
        else:
            kw = rtdl.FTTransformer.get_default_kwargs(n_blocks=1)
            kw.pop('_is_default')
            kw.update(d_block=32, attention_n_heads=4, attention_dropout=0.1, ffn_dropout=0.0)
            net = rtdl.FTTransformer(n_cont_features=12, cat_cardinalities=[], d_out=1, **kw)
        net = net.cuda()
        parameters = net.parameters()
        if kind == 'ft_transformer':
            names = {id(p): n for n, p in net.named_parameters()}
            parameters = net.make_parameter_groups()
            for group in parameters:
                group['params'] = sorted(group['params'], key=lambda p: names[id(p)])
        opt = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0001, foreach=False, fused=False)
        torch.cuda.synchronize()
        started, losses = time.perf_counter(), []
        for _ in range(3):
            net.train()
            logits = (net(x, None) if kind == 'ft_transformer' else net(x)).squeeze(-1)
            target = y[:, None].expand_as(logits) if logits.ndim == 2 else y
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        net.eval()
        with torch.inference_mode():
            logits = net(x, None) if kind == 'ft_transformer' else net(x)
            p = torch.sigmoid(logits.double()).squeeze(-1)
            if p.ndim == 2:
                p = p.mean(dim=1)
        torch.cuda.synchronize()
        state = hashlib.sha256()
        for name, value in sorted(net.state_dict().items()):
            a = value.detach().cpu().contiguous().numpy()
            state.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        p = p.cpu().numpy()
        assert p.shape == (128,) and np.isfinite(p).all() and np.ptp(p) > 0
        probes[kind] = {'state_sha256': state.hexdigest(),
            'prediction_sha256': hashlib.sha256(p.tobytes()).hexdigest(), 'losses': losses,
            'parameters': sum(p.numel() for p in net.parameters()), 'elapsed_seconds': time.perf_counter() - started}
        del opt, net
    sources = {}
    for name, module in [('tabm', tabm), ('rtdl-num-embeddings', emb), ('rtdl-revisiting-models', rtdl)]:
        dist = importlib.metadata.distribution(name)
        sources[name] = {'version': dist.version, 'source_sha256': sha(module.__file__),
            'license': dist.metadata.get('License'),
            'license_files_sha256': {str(f): sha(dist.locate_file(f)) for f in dist.files
                if 'license' in str(f).lower() and Path(dist.locate_file(f)).is_file()}}
    props = torch.cuda.get_device_properties(0)
    return {'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda, 'device': props.name,
        'device_memory_bytes': props.total_memory, 'capability': list(torch.cuda.get_device_capability(0)),
        'policy': {'deterministic_algorithms': True, 'fp32_precision': 'ieee', 'mixed_precision': False,
                   'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'], 'threads': 4},
        'sources': sources, 'probes': probes,
        'api_signatures': {name: str(inspect.signature(obj)) for name, obj in
            [('MLP', rtdl.MLP), ('ResNet', rtdl.ResNet), ('FTTransformer', rtdl.FTTransformer),
             ('PeriodicEmbeddings', emb.PeriodicEmbeddings), ('PiecewiseLinearEmbeddings', emb.PiecewiseLinearEmbeddings)]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child(), sort_keys=True))
        return
    if not args.output:
        parser.error('--output is required')
    env = dict(os.environ, PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    runs = []
    for _ in range(2):
        proc = subprocess.run([sys.executable, '-m', 'analysis.p29_node_probe', '--child'], env=env,
            capture_output=True, text=True, timeout=180)
        if proc.returncode:
            raise RuntimeError(f'CUDA child failed ({proc.returncode}): {proc.stderr}')
        runs.append(json.loads(proc.stdout))
    for kind in runs[0]['probes']:
        for key in ['state_sha256', 'prediction_sha256', 'losses', 'parameters']:
            assert runs[0]['probes'][kind][key] == runs[1]['probes'][kind][key], (kind, key)
    result = {'ok': True, 'scope': 'synthetic architecture/API smoke test only; no data or AUC',
        'fresh_processes': 2, 'byte_identical': True, 'script_sha256': sha(__file__), 'runs': runs}
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('CUDA: six architecture probes, two fresh processes, byte-identical states/predictions; no AUC measured')


if __name__ == '__main__':
    main()
