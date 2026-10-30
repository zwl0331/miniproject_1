"""Batch manifests (plan sections 5.1, 5.6 and 15): the ordered configurations of one search batch, the run ids
they will have, and the pre-batch state its allocation is judged against; written once before the batch's first
fit."""
import json
from pathlib import Path

from songpop import artifacts
from songpop.phase2.recipes import recipe_hash, validate_recipe

BATCH_SCHEMA = "batch_v1"
KINDS = ("initial", "expansion", "refinement")
STATE_KEYS = ("family_anchor", "incumbent", "strongest", "reference_h", "previous_best_blend", "allowance_before")
KEYS = ("schema", "task_id", "protocol_sha256", "source_manifest_sha256", "family", "wave", "kind", "recipes",
        "recipe_sha256", "run_ids", "state", "sources")


def validate_batch(m):
    if not isinstance(m, dict) or sorted(m) != sorted(KEYS):
        raise ValueError(f"batch keys {sorted(m) if isinstance(m, dict) else m!r}, expected {sorted(KEYS)}")
    if m["schema"] != BATCH_SCHEMA or m["kind"] not in KINDS:
        raise ValueError(f"batch schema {m['schema']!r} or kind {m['kind']!r} not recognized")
    if not isinstance(m["wave"], int) or isinstance(m["wave"], bool) or m["wave"] < 1:
        raise ValueError("wave must be a positive integer")
    if not isinstance(m["state"], dict) or sorted(m["state"]) != sorted(STATE_KEYS):
        raise ValueError(f"state keys {sorted(m['state'])}, expected {sorted(STATE_KEYS)}")
    if not m["recipes"] or any(validate_recipe(r)["family"] != m["family"] for r in m["recipes"]):
        raise ValueError("a batch holds at least one recipe, all of its family")
    hashes = [recipe_hash(r) for r in m["recipes"]]
    if hashes != m["recipe_sha256"] or len(set(hashes)) != len(hashes):
        raise ValueError("recipe hashes differ from the recipes, or a recipe appears twice")
    if len(m["run_ids"]) != len(m["recipes"]) or len(set(m["run_ids"])) != len(m["run_ids"]):
        raise ValueError("one distinct run id per recipe")
    return m


def batch_id(m):
    digest = artifacts.sha256_bytes(artifacts.dumps(validate_batch(m)).encode())
    return f"{m['family']}_w{m['wave']}_{m['kind']}_{digest[:12]}"


def manifest_text(m):
    return json.dumps({"batch_id": batch_id(m), **m}, indent=1, sort_keys=True, allow_nan=False) + "\n"


def save_batch(root, directory, m):
    """Write the manifest once (exclusive create); an existing file must hold the same content. Returns
    (batch id, relative path, sha256 of the file)."""
    content = manifest_text(m)
    rel = Path(directory) / f"{batch_id(m)}.json"
    path = Path(root) / rel
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"{rel} exists with different content")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x") as f:
            f.write(content)
    return batch_id(m), rel, artifacts.sha256_bytes(content.encode())
