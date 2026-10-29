#!/usr/bin/env python3
"""Revision fingerprint for review-controlled work (doc/review_policy.md, section 4.2).

Usage: tools/fingerprint.py [-m] [PATH ...]

File set: the PATHs given, relative to the project root. With none: every tracked and
untracked file that is not ignored, if the project is a git work tree; otherwise every
file under the project root except .git/ and __pycache__/ directories. Review records
reviews/YYYY-MM-DD-*.md are always excluded, so appending a request or a verdict never
changes the fingerprint.

Manifest: one line per file, "<type> <exec> <sha256> <path>", where type is f (regular
file, hash of its content), l (symlink, hash of the link target) or D (tracked file
absent from the working tree; git mode only), and exec is x or -. Lines are sorted
bytewise by path; the fingerprint is the sha256 of the manifest. With -m the manifest
is printed before the "FINGERPRINT <hash>" line; without it only the hash is printed.

No fingerprint is produced (exit 2) on any error: a directory that cannot be listed, a
path whose metadata cannot be read (only a genuinely absent path counts as absent), a
link or file that cannot be read, a missing PATH, a directory or special file, or a
path containing control characters, which would make the manifest ambiguous.
"""
import hashlib
import os
import re
import stat
import subprocess
import sys

RECORD = re.compile(r"^reviews/\d{4}-\d{2}-\d{2}-.*\.md$")


def die(msg):
    print(f"fingerprint: {msg}", file=sys.stderr)
    sys.exit(2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_files(root):
    try:
        r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root,
                           capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if r.returncode != 0 or r.stdout.strip() != "true":
        return None
    r = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root,
                       capture_output=True)
    if r.returncode != 0:
        die("git ls-files failed: " + r.stderr.decode(errors="replace").strip())
    return [p.decode("utf-8", "surrogateescape") for p in r.stdout.split(b"\0") if p]


def walk_files(root):
    def onerror(e):
        die(f"cannot list directory: {e}")

    out = []
    for d, dirs, files in os.walk(root, onerror=onerror):
        keep = []
        for x in dirs:
            p = os.path.join(d, x)
            if (d == root and x == ".git") or x == "__pycache__":
                continue
            try:
                is_link = stat.S_ISLNK(os.lstat(p).st_mode)
            except OSError as e:
                die(f"cannot stat {os.path.relpath(p, root)}: {e}")
            if is_link:
                out.append(os.path.relpath(p, root))
            else:
                keep.append(x)
        dirs[:] = sorted(keep)
        for x in files:
            out.append(os.path.relpath(os.path.join(d, x), root))
    return out


def classify(p, mode):
    """Return (type, exec, hash) for path p, or abort. Only a genuinely absent path
    counts as absent; any other metadata or read error is fatal."""
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        if mode == "git":
            return ("D", "-", "-")
        die(f"missing file: {p}")
    except OSError as e:
        die(f"cannot stat {p}: {e}")
    if stat.S_ISLNK(st.st_mode):
        try:
            target = os.readlink(p)
        except OSError as e:
            die(f"cannot read link {p}: {e}")
        return ("l", "-", hashlib.sha256(target.encode("utf-8", "surrogateescape")).hexdigest())
    if stat.S_ISREG(st.st_mode):
        try:
            h = sha256_file(p)
        except OSError as e:
            die(f"cannot read {p}: {e}")
        return ("f", "x" if st.st_mode & 0o111 else "-", h)
    die(f"unsupported file type (directory or special file): {p}")


def main(argv):
    show = False
    args = list(argv)
    if args and args[0] == "-m":
        show = True
        args = args[1:]
    if any(a.startswith("-") for a in args):
        die("unknown option; usage: tools/fingerprint.py [-m] [PATH ...]")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if args:
        mode = "list"
        paths = [os.path.normpath(a) for a in args]
        for p in paths:
            if os.path.isabs(p) or p == ".." or p.startswith("../") or p == ".":
                die(f"path outside the project root or not a file path: {p}")
    else:
        paths = git_files(root)
        mode = "git" if paths is not None else "tree"
        if paths is None:
            paths = walk_files(root)
    lines, seen = [], set()
    for p in paths:
        p = os.path.normpath(p)
        if RECORD.match(p) or p in seen:
            continue
        seen.add(p)
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in p):
            die(f"path contains control characters: {p!r}")
        t, x, h = classify(p, mode)
        lines.append((t, x, h, p))
    if not lines:
        die("empty file set")
    lines.sort(key=lambda t: t[3].encode("utf-8", "surrogateescape"))
    manifest = "".join(f"{t} {x} {h} {p}\n" for t, x, h, p in lines)
    fp = hashlib.sha256(manifest.encode("utf-8", "surrogateescape")).hexdigest()
    if show:
        sys.stdout.write(manifest)
        print(f"FINGERPRINT {fp}")
    else:
        print(fp)


if __name__ == "__main__":
    main(sys.argv[1:])
