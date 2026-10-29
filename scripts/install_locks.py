"""Hashed installation locks for the phase 2 routes (doc/plans/phase2-models-features.md section 4.9).

From the project root, with PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.install_locks:
    inventory                 record the protected inventory of .venv and its constraints (P2.1)
    resolve                   dry-run all routes jointly against the protected inventory, then each route against
                              the protected inventory plus the joint additions, saving pip's reports (P2.1)
    lock                      write configs/phase2/install/<route>.txt from the reports, checking every rule (P2.1)
    check-before <route>      the installed inventory is the protected one plus approved additions; saves it (P2.3, P2.5)
    check-after <route>       the installed inventory is the saved before-inventory plus exactly the route's lock;
                              saves it (P2.3, P2.5)
The install itself, reviewed in P2.1 and executed unchanged in P2.3 or P2.5:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pip install --no-deps --require-hashes -r configs/phase2/install/<route>.txt
The locks are additive and specific to this Mac and Python; they are not lock files for a clean environment.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ROUTES = {"catboost": ["catboost==1.2.10"], "ebm": ["interpret-core==0.7.8"], "tabm": ["torch==2.14.1", "tabm==0.0.3"]}
INSTALL_DIR = Path("artifacts/phase2/install")
LOCK_DIR = Path("configs/phase2/install")
PROTECTED = INSTALL_DIR / "inventory_protected.json"
CONSTRAINTS = INSTALL_DIR / "constraints_protected.txt"
RESOLVED = INSTALL_DIR / "constraints_resolved.txt"
JOINT = INSTALL_DIR / "report_joint.json"
PIP = [sys.executable, "-m", "pip"]
DRY = ["install", "--dry-run", "--quiet", "--no-cache-dir", "--disable-pip-version-check", "--only-binary=:all:"]


def inventory():
    out = subprocess.run(PIP + ["list", "--format=json", "--disable-pip-version-check"], capture_output=True, text=True,
                         check=True).stdout
    names = [canonicalize_name(d["name"]) for d in json.loads(out)]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate normalized distribution names in the inventory")
    return {canonicalize_name(d["name"]): d["version"] for d in json.loads(out)}


def write_once(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(text)


def cmd_inventory(root):
    inv = inventory()
    write_once(root / PROTECTED, json.dumps(inv, indent=1, sort_keys=True) + "\n")
    write_once(root / CONSTRAINTS, "".join(f"{n}=={v}\n" for n, v in sorted(inv.items())))
    print(f"protected inventory: {len(inv)} distributions in {PROTECTED}")


def additions(report):
    """{name: (version, url, sha256)} that pip would add, from a dry-run report, with its rules checked."""
    if report.get("version") != "1":
        raise ValueError(f"pip report schema {report.get('version')!r}, expected '1'")
    tags = set(sys_tags())
    out = {}
    for item in report["install"]:
        name, version = canonicalize_name(item["metadata"]["name"]), item["metadata"]["version"]
        info = item["download_info"]
        url, sha = info["url"], info.get("archive_info", {}).get("hashes", {}).get("sha256")
        if sha is None:
            raise ValueError(f"{name}: no sha256 in the report")
        if not url.endswith(".whl"):
            raise ValueError(f"{name}: resolves to a non-wheel {url}")
        wname, wversion, _, wtags = parse_wheel_filename(url.rsplit("/", 1)[-1])
        if canonicalize_name(wname) != name or str(wversion) != version:
            raise ValueError(f"{name}: wheel file name does not match {name} {version}")
        if not wtags & tags:
            raise ValueError(f"{name}: wheel {url.rsplit('/', 1)[-1]} is not supported on this interpreter")
        out[name] = (version, url, sha)
    return out


def cmd_resolve(root):
    protected = json.loads((root / PROTECTED).read_text())
    if inventory() != protected:
        raise SystemExit("the installed inventory differs from the protected one; resolve against it only")
    every = [p for pkgs in ROUTES.values() for p in pkgs]
    res = subprocess.run(PIP + DRY + ["--report", str(root / JOINT), "-c", str(root / CONSTRAINTS), *every],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"joint resolution FAILED (exit {res.returncode}): {res.stderr.strip()[-600:]}")
    joint = additions(json.loads((root / JOINT).read_text()))
    write_once(root / RESOLVED, "".join(f"{n}=={v}\n" for n, v in sorted(protected.items())) +
               "".join(f"{n}=={v}\n" for n, (v, _, _) in sorted(joint.items())))
    print(f"joint: {' '.join(every)}: {len(joint)} additions")
    for route, pkgs in ROUTES.items():
        report = root / INSTALL_DIR / f"report_{route}.json"
        res = subprocess.run(PIP + DRY + ["--report", str(report), "-c", str(root / RESOLVED), *pkgs],
                             capture_output=True, text=True)
        status = "resolved" if res.returncode == 0 else f"FAILED (exit {res.returncode}): {res.stderr.strip()[-400:]}"
        print(f"{route}: {' '.join(pkgs)}: {status}")


def cmd_lock(root):
    protected = json.loads((root / PROTECTED).read_text())
    joint = additions(json.loads((root / JOINT).read_text()))
    closures, problems = {}, []
    for route in ROUTES:
        path = root / INSTALL_DIR / f"report_{route}.json"
        if not path.exists():
            problems.append(f"{route}: no report (resolution failed); the route is dropped under the plan")
            continue
        c = additions(json.loads(path.read_text()))
        changed = sorted(n for n in c if n in protected)
        differs = sorted(n for n in c if joint.get(n) != c[n])
        if changed or differs:
            problems.append(f"{route}: would change installed {changed} or differs from the joint resolution "
                            f"{differs}; the route is dropped under the plan")
            continue
        closures[route] = c
    if not closures:
        raise SystemExit("\n".join(problems))
    for route, c in closures.items():
        lines = [f"# Route {route}: the additive dependency closure of {' '.join(ROUTES[route])}, resolved against",
                 f"# {PROTECTED} jointly with the other routes (doc/plans/phase2-models-features.md section 4.9).",
                 "# Exact wheels by URL and sha256; pip resolves nothing further. Install with:",
                 f"# PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pip install --no-deps --require-hashes -r {LOCK_DIR}/{route}.txt",
                 "--no-index", "--only-binary=:all:"]
        lines += [f"{n} @ {u} --hash=sha256:{h}" for n, (v, u, h) in sorted(c.items())]
        write_once(root / LOCK_DIR / f"{route}.txt", "\n".join(lines) + "\n")
        print(f"{route}: {len(c)} distributions: " + ", ".join(f"{n} {v}" for n, (v, _, _) in sorted(c.items())))
    for p in problems:
        print(p)


def read_lock(root, route):
    """{name: (version, sha256)} from a lock file; the version is read from the wheel file name."""
    entries = {}
    for line in (root / LOCK_DIR / f"{route}.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line in ("--no-index", "--only-binary=:all:"):
            continue
        name, _, rest = line.partition(" @ ")
        url, _, h = rest.partition(" --hash=sha256:")
        if not name or not url.endswith(".whl") or len(h) != 64:
            raise ValueError(f"malformed lock line in {route}: {line!r}")
        _, version, _, _ = parse_wheel_filename(url.rsplit("/", 1)[-1])
        entries[canonicalize_name(name)] = (str(version), h)
    return entries


def approved_additions(root, routes):
    out = {}
    for r in routes:
        if (root / LOCK_DIR / f"{r}.txt").exists():
            out.update({n: v for n, (v, _) in read_lock(root, r).items()})
    return out


def inventory_problems(protected, now, lock, others, before=None):
    """Problems with an inventory. Before an installation (before=None): the protected inventory unchanged, and
    every addition a previously approved one. After it: exactly the saved before-inventory plus the route's
    lock, so nothing installed earlier may disappear or change."""
    if before is None:
        changed = {n: (v, now.get(n)) for n, v in protected.items() if now.get(n) != v}
        extra = {n: v for n, v in now.items() if n not in protected and others.get(n) != v}
        return {k: v for k, v in (("protected changed", changed), ("unapproved additions", extra)) if v}
    expected = {**before, **lock}
    diff = {n: (expected.get(n), now.get(n)) for n in set(expected) | set(now) if expected.get(n) != now.get(n)}
    return {"differs from the before-inventory plus the lock": diff} if diff else {}


def cmd_check(root, route, after):
    protected = json.loads((root / PROTECTED).read_text())
    now = inventory()
    others = approved_additions(root, [r for r in ROUTES if r != route])
    lock = {n: v for n, (v, _) in read_lock(root, route).items()}
    before_path, after_path = root / INSTALL_DIR / f"inventory_before_{route}.json", root / INSTALL_DIR / f"inventory_after_{route}.json"
    if after:
        before = json.loads(before_path.read_text())
        problems = inventory_problems(protected, now, lock, others, before)
    else:
        problems = inventory_problems(protected, now, lock, others)
    if not problems:
        write_once(after_path if after else before_path, json.dumps(now, indent=1, sort_keys=True) + "\n")
    print(f"{'after' if after else 'before'} {route}: {problems or 'OK'}"
          + ("" if problems else f"; inventory saved in {(after_path if after else before_path).relative_to(root)}"))
    sys.exit(0 if not problems else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["inventory", "resolve", "lock", "check-before", "check-after"])
    ap.add_argument("route", nargs="?", choices=sorted(ROUTES))
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if args.command.startswith("check") and not args.route:
        ap.error("check-before and check-after need a route")
    {"inventory": lambda: cmd_inventory(root), "resolve": lambda: cmd_resolve(root), "lock": lambda: cmd_lock(root),
     "check-before": lambda: cmd_check(root, args.route, False),
     "check-after": lambda: cmd_check(root, args.route, True)}[args.command]()


if __name__ == "__main__":
    main()
