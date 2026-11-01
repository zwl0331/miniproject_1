# P2.9a GPU-node startup

2026-11-01. Primary developer: Codex. Consultant and separate reviewer: Claude Fable 5.1 at xhigh. Status: environment and migration diagnostics completed; protocol amendment awaiting independent R2. No new-method search, submission or final selection.

Owner instructions, verbatim:

> Let us start, use fable 5.1 xhigh as consultant and review model

> This is plently of computing power to do the work relatively quickly right?

The primary confirmed the actual allocation and answered that it is substantial for the dataset, while total search time requires measurement. All timing claims remain scoped to actual runs.

The first consultation launch was rejected by automatic approval review because private project context would be sent to Claude. The owner then answered:

> Allow project context for consultation and review

The exact sharing question and approval are in decisions.md. The same read-only consultation command was then allowed. Both consultations explicitly used `--model claude-fable-5-1 --effort xhigh --permission-mode plan --tools Read,Glob,Grep`; no model fallback or helper agent was used. The first had 60 turns and the follow-up 12, in session 73d0bfbd-9b21-4a22-82d4-06a22f867348. JSON responses retain the actual model metadata and verbatim outputs. The independent reviewer uses a separate fresh session through the cross-review skill.

## Records

| File | Purpose |
|---|---|
| claude_prior_positions.md | Verbatim prior Fable positions saved before this task's independent design |
| codex_independent_design.md | Codex's design and critique before reading the new consultation |
| consultation_prompt.txt, consultation_response.json, claude_positions.md | Exact first consultation and extracted response |
| critique_prompt.txt, critique_response.json, claude_draft_critique.md | Exact follow-up critique and extracted response |
| integration.md | Accepted suggestions, evidence-based corrections and pending work |
| node_inventory.json | Initial OS, cgroup, interpreter and GPU observations |
| python_install.json | Conda-forge Python 3.12.16 transaction |
| python_rebuild_dry_run.json, python_rebuild_rejected_header.json | Successful explicit-lock parse and retained failed first parse; exported mamba output needed a comment header and @EXPLICIT marker |
| linux_*resolve.json and .log | Dry-run package resolution, with the final neural report used for the lock |
| linux_install*.log | Two quota-interrupted installs and the successful scratch installation |
| linux_inventory_verified.json, package_licenses.json | Exact final inventory; license identifiers, source URLs and license hashes |
| cuda_probe.json and .log | Six synthetic architectures, two fresh processes, exact-repeat evidence |
| reference_checks.txt | Validation/hashes of eight historical banks; reconstruction from their saved probabilities |
| portability_probe.json and .log | First-fold CPU H/TabM comparison with the Mac banks; no AUC |
| portability_cpu4_* | First-fold repeat diagnostics with all four thread variables explicit |
| tree_replay*.log, migration_comparison.txt | Full historical tree refits, fresh node repeat and migration comparison |
| check_foundation.txt, check_phase2.txt | Existing suites run on the node; the latter was still running at initial record |

The installed environment is node-local at /tmp/miniproject_1-p29-venv-20261101, reached by .venv. The shared filesystem returned errno 122 (disk quota exceeded) after the first installation had written about 5.1 GiB. No project data or banks were moved. The exact same locked package transaction completed on scratch and passed the full inventory and pip consistency checks. Scratch is disposable; the committed lock and interpreter transaction are the rebuild inputs. REVIEW_PYTHON now points to the project interpreter.

Interpreter rebuild syntax was dry-run checked with `mamba create --prefix /tmp/miniproject_1-p29-rebuild-check --file configs/phase2/p2.9/install/python-linux-explicit.txt --dry-run --json`. The initial `mamba list --explicit --sha256 --no-pip` output contained a human-readable header and omitted the @EXPLICIT marker. Correcting those two header lines made the dry run pass; package URLs/hashes were unchanged. A second full environment installation was not performed. Then install `linux-cuda.txt` with the documented no-deps/hash-required pip command and run `analysis.p29_environment verify` against its receipt.

## Integration into the project

Codex wrote the environment/diagnostic scripts and Linux lock, corrected the reviewer path, updated state and decisions, and drafted the source contracts and explicit portability amendment. Findings are recorded as draft Finding 031 before new methods may depend on them. All old model sources, data, banks, submissions and requirements.txt remain unchanged during the migration diagnostics. The new diagnostic prediction arrays live under artifacts/phase2/p29 and are not registered as production banks.

The general-purpose PyPI neural implementations and CUDA dependencies contain no pretrained models or external training data. Their licenses and source hashes were audited. Versioned official PyTorch and author documentation was consulted; links are in the node contracts. No Spotify lookups, external datasets, benchmark data, competitor notebooks or pretrained weights were accessed.

Pending: fresh R2, new adapter/schema implementation and rejection tests, executable ordered manifests, full-route and nested-size cost pilots, and the measured large-job proposal. This checkpoint does not complete P2.9a.
