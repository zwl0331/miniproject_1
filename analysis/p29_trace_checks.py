"""Diagnose selected unchanged legacy assertions with original-line tracebacks.

Only the runner decorator is instrumented in memory. Test bodies, constants,
model sources and assertion expressions are unchanged. Run from the project root.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import traceback


def diagnostic(tb):
    """Small numerical summaries of the two failed historical assertions."""
    import numpy as np
    while tb.tb_next is not None:
        tb = tb.tb_next
    values = tb.tb_frame.f_locals
    if "batch" in values and "alone" in values:
        batch, alone = np.asarray(values["batch"]), np.asarray(values["alone"])
        print("Batch probabilities:", [float(x).hex() for x in batch], flush=True)
        print("Alone probability:", [float(x).hex() for x in alone], flush=True)
        print("Repeated-query absolute differences:", np.abs(batch[[0, 2, 4]] - alone[0]).tolist(), flush=True)
    if "b" in values and "p1" in values and hasattr(values["b"], "oof"):
        for name in ("oof", "test_by_fold", "test_mean", "fold_auc"):
            x, y = getattr(values["b"], name), getattr(values["p1"], name)
            print(f"Anchor {name}: max absolute error {float(np.max(np.abs(x - y))):.17g}; "
                  f"unequal entries {int(np.count_nonzero(x != y))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    path = Path("scripts/check_phase2.py").resolve()
    tree = ast.parse(path.read_text(), filename=str(path))
    check = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check")
    wrap = next(node for node in check.body if isinstance(node, ast.FunctionDef) and node.name == "wrap")
    selected = ast.parse("if not any(term in name for term in TRACE_TERMS):\n    return fn").body[0]
    wrap.body.insert(0, ast.copy_location(selected, wrap.body[0]))
    handler = next(node for node in ast.walk(wrap) if isinstance(node, ast.ExceptHandler)
                   and isinstance(node.type, ast.Name) and node.type.id == "Exception")
    for node in reversed(ast.parse("TRACE_MODULE.print_exc()\nTRACE_DIAGNOSTIC(e.__traceback__)").body):
        handler.body.insert(0, ast.copy_location(node, handler.body[0]))
    ast.fix_missing_locations(tree)
    print(f"Original test sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(f"Filters {args.names!r}; only decorator instrumentation", flush=True)
    namespace = {"__name__": "__main__", "__file__": str(path),
                 "TRACE_TERMS": args.names, "TRACE_MODULE": traceback, "TRACE_DIAGNOSTIC": diagnostic}
    exec(compile(tree, str(path), "exec"), namespace)


if __name__ == "__main__":
    main()
