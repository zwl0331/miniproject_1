# Neural adapter LLM use

2026-11-01. Codex development session, parallel to the existing fast-exploration
session. Owner instruction, verbatim:

> I want you to use the same fast mode spirit as the other session, but you are in charge of finishing the nerual adapter so that one can test fancy neural nets method

Codex read the paused P2.9 implementation and coordinated file ownership with
the other Codex session. It reused the raw-row fitting boundary and extended
the worker with embedded TabM, MLP, ResNet and FT-Transformer, recipe validation,
an explicit saved-fold runner, tests and usage documentation. The work uses
the installed general architecture libraries and no external song data.

Output integration: an isolated branch based on master; only new adapter,
runner, tests and documentation files are handed to the primary development
session for integration and complete CV experiments. No old P2.9 process gates
or exploration reviews were restarted. Previous submission files were untouched.

The initial seven scalar checks passed in 33.612 seconds. The initial four
advanced checks passed in 80.894 seconds, including all ten supported
architecture/representation combinations on CPU and CUDA. After cleanup of
expected constant-bin warnings and worker output handles, `check_advanced.txt`
records the final advanced checks and `check_legacy.txt` records scalar
compatibility. `check_bank_smoke.txt` records a complete 15-fold, two-epoch GPU
integration run. It finished in 99.2 seconds, wrote the normal prediction bank
and passed its automatic validation. The result config and predictions are
under `artifacts/neural/adapter_smoke_mlp_ple_2epochs/`; code commit `bf9aa18`.

This record is development evidence, not independent review or a final release.
