"""Run only the checks of scripts/check_phase2.py whose name contains argv[1] (scratch helper)."""
import sys
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root))
src = (root / "scripts/check_phase2.py").read_text()
marker = '        if status != "run":'
assert src.count(marker) == 1
src = src.replace(marker, '        if FILTER not in name:\n            return fn\n' + marker, 1)
g = {"__name__": "__main__", "__file__": str(root / "scripts/check_phase2.py"), "FILTER": sys.argv[1]}
exec(compile(src, str(root / "scripts/check_phase2.py"), "exec"), g)
