#!/usr/bin/env python3
"""Resolve the EarthSciAST releases each binding is tested against.

Each binding declares the EarthSciAST versions it supports in its manifest:
the Julia `[compat]` entry, the Cargo requirement and the PEP 508 dependency.
This script reads those three ranges, lists the releases on the registry each
binding installs from (General, crates.io, PyPI; yanked and pre-releases
excluded), and prints, per binding, the LOWEST and the LATEST release inside
the range: the two ends conformance.yml tests, so the declared floor is a
tested fact rather than a guess, and the latest is what a user installs
today. When the two ends are the same release it is listed once. The Python
binding is also tested at both ends of its `requires-python` range (the
declared floor and PYTHON_NEWEST), crossed with the two EarthSciAST ends.

A range no release satisfies is an error: it means a manifest asks for an
EarthSciAST that has not been published, and a release made from it would be
uninstallable. Release EarthSciAST first, then raise the floor here.

Usage:
  scripts/earthsci-ast-versions.py                  # print a table
  scripts/earthsci-ast-versions.py --github-output  # also write the matrices
                                                    # to $GITHUB_OUTPUT
  scripts/earthsci-ast-versions.py --self-test      # check the range parsers
  scripts/earthsci-ast-versions.py --admit-checkout DIR
      # CI only (upstream-main.yml, docs.yml): rewrite the Julia and Cargo
      # manifests IN PLACE so they admit the EarthSciAST checkout at DIR when
      # its version is outside the declared range (EarthSciAST `main` bumps
      # its version before it releases). Never commit the result.

Python 3.11+ (tomllib); standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The newest CPython the Python binding is tested on (the top of its
# `requires-python` range; the bottom is read from pyproject.toml).
PYTHON_NEWEST = "3.14"
# The runner for the requires-python floor: the newest Ubuntu image that
# actions/setup-python still has that Python for (3.9 left ubuntu-latest with
# 26.04). When this image retires, raise requires-python.
PYTHON_FLOOR_RUNNER = "ubuntu-24.04"

Version = tuple[int, int, int]
# A range is a union of intervals [lo, hi), hi = None for unbounded.
Interval = tuple[Version, "Version | None"]


def parse_version(s: str) -> Version:
    parts = s.strip().split(".")
    if not 1 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"not a plain release version: {s!r}")
    nums = [int(p) for p in parts] + [0] * (3 - len(parts))
    return (nums[0], nums[1], nums[2])


def show(v: Version) -> str:
    return ".".join(map(str, v))


def bump_written(s: str) -> Version:
    """The first version past every version `s` matches when its missing
    components are wildcards: `0.6` -> 0.7.0, `0` -> 1.0.0, `0.6.2` -> 0.6.3."""
    n = len(s.strip().split("."))
    x = parse_version(s)
    if n == 1:
        return (x[0] + 1, 0, 0)
    if n == 2:
        return (x[0], x[1] + 1, 0)
    return (x[0], x[1], x[2] + 1)


def caret(s: str) -> Interval:
    """`^x.y.z` as Cargo and Julia both define it: the left-most non-zero
    component of the written version may not change."""
    parts = s.strip().split(".")
    lo = parse_version(s)
    major, minor, patch = lo
    if major > 0 or len(parts) == 1:
        hi = (major + 1, 0, 0)
    elif minor > 0 or len(parts) == 2:
        hi = (0, minor + 1, 0)
    else:
        hi = (0, 0, patch + 1)
    return lo, hi


def tilde(s: str) -> Interval:
    """`~x.y.z` as Cargo and Julia both implement it: only components past
    the minor may change (`~x`: past the major). Neither has a 0.0.x special
    case: `~0.0.3` admits 0.0.9 under Julia 1.12's Pkg and Rust's semver 1.x
    alike (see SELF_TEST)."""
    parts = s.strip().split(".")
    lo = parse_version(s)
    hi = (lo[0] + 1, 0, 0) if len(parts) == 1 else (lo[0], lo[1] + 1, 0)
    return lo, hi


def comparator(op: str, v: str, wildcard: bool) -> Interval:
    """One `op version` comparator as an interval (`!=` is not supported).

    `wildcard` is how a partial version reads. Cargo treats the missing
    components as wildcards (`<=0.6` is `<0.7.0`, `>0.6` is `>=0.7.0`, `=0.6`
    is `[0.6.0, 0.7.0)`); PEP 440 and Julia zero-pad them (`<=0.6` is
    `<=0.6.0`; Julia `=0.6` is exactly 0.6.0)."""
    x = parse_version(v)
    nxt = bump_written(v) if wildcard else (x[0], x[1], x[2] + 1)
    return {
        ">=": (x, None),
        ">": (nxt, None),
        "<": ((0, 0, 0), x),
        "<=": ((0, 0, 0), nxt),
        "=": (x, nxt),
        "==": (x, nxt),
    }[op]


def intersect(a: Interval, b: Interval) -> Interval:
    lo = max(a[0], b[0])
    if a[1] is None or b[1] is None:
        hi = a[1] if b[1] is None else b[1]
    else:
        hi = min(a[1], b[1])
    return lo, hi


CMP = re.compile(r"^(>=|<=|==|=|>|<|\^|~=|~)?\s*([0-9][0-9.]*)$")


def cargo_range(req: str) -> list[Interval]:
    """A Cargo requirement: comma-separated comparators, all of which hold."""
    acc: Interval = ((0, 0, 0), None)
    for part in req.split(","):
        m = CMP.match(part.strip())
        if not m or m.group(1) == "~=":
            raise ValueError(f"unsupported Cargo requirement: {req!r}")
        op, v = m.group(1) or "^", m.group(2)
        if op == "^":
            iv = caret(v)
        elif op == "~":
            iv = tilde(v)
        else:
            iv = comparator(op, v, wildcard=True)
        acc = intersect(acc, iv)
    return [acc]


def pep440_range(spec: str) -> list[Interval]:
    """A PEP 440 specifier set: comma-separated clauses, all of which hold."""
    acc: Interval = ((0, 0, 0), None)
    for part in spec.split(","):
        m = CMP.match(part.strip())
        if not m or m.group(1) in (None, "^", "~", "="):
            raise ValueError(f"unsupported PEP 440 specifier: {spec!r}")
        op, v = m.group(1), m.group(2)
        if op == "~=":
            n = len(v.split("."))
            if n < 2:
                raise ValueError(f"`~=` needs two components: {spec!r}")
            lo = parse_version(v)
            hi = (lo[0] + 1, 0, 0) if n == 2 else (lo[0], lo[1] + 1, 0)
            iv: Interval = (lo, hi)
        else:
            iv = comparator(op, v, wildcard=False)
        acc = intersect(acc, iv)
    return [acc]


def julia_range(compat: str) -> list[Interval]:
    """A Julia `[compat]` entry: comma-separated specifiers, any of which
    holds (bare = caret, `~`, `=`, `>=`, and `a - b` hyphen ranges)."""
    out: list[Interval] = []
    for part in compat.split(","):
        p = part.strip()
        if " - " in p:
            a, b = (x.strip() for x in p.split(" - "))
            out.append((parse_version(a), bump_written(b)))
            continue
        m = CMP.match(p)
        if not m or m.group(1) not in (None, "^", "~", "=", ">="):
            raise ValueError(f"unsupported Julia compat entry: {compat!r}")
        op, v = m.group(1) or "^", m.group(2)
        if op == "^":
            out.append(caret(v))
        elif op == "~":
            out.append(tilde(v))
        else:
            out.append(comparator(op, v, wildcard=False))
    return out


def contains(rng: list[Interval], v: Version) -> bool:
    return any(lo <= v and (hi is None or v < hi) for lo, hi in rng)


# Every row was checked against the real resolvers: Julia 1.12's
# `Pkg.Types.semver_spec` and Rust's `semver` 1.x `VersionReq::matches`
# (the PEP 440 rows follow the spec's zero-padding). --self-test asserts them
# in the `resolve` job, so a parser that drifts from a resolver fails there.
SELF_TEST: list[tuple[str, str, str, bool]] = [
    ("julia", "0.6", "0.6.9", True),
    ("julia", "0.6", "0.7.0", False),
    ("julia", "=0.6", "0.6.0", True),
    ("julia", "=0.6", "0.6.1", False),
    ("julia", "~0.0.3", "0.0.5", True),
    ("julia", "~0.0.3", "0.1.0", False),
    ("julia", "~0.2.3", "0.2.9", True),
    ("julia", "~1", "1.5.0", True),
    ("julia", "0.6 - 0.7", "0.7.9", True),
    ("julia", "0.6 - 0.7.2", "0.7.2", True),
    ("julia", "0.6 - 0.7.2", "0.7.3", False),
    ("julia", ">=0.6", "9.0.0", True),
    ("julia", "0.4, 0.6", "0.5.0", False),
    ("rust", "=0.6", "0.6.3", True),
    ("rust", "<=0.6", "0.6.9", True),
    ("rust", "<=0.6", "0.7.0", False),
    ("rust", ">0.6", "0.6.5", False),
    ("rust", ">0.6", "0.7.0", True),
    ("rust", "<0.6", "0.5.9", True),
    ("rust", "<0.6", "0.6.0", False),
    ("rust", "~0.0.3", "0.0.9", True),
    ("rust", "~0.0.3", "0.1.0", False),
    ("rust", ">=0.5, <0.7", "0.6.9", True),
    ("rust", "<=0.6.2", "0.6.2", True),
    ("rust", "<=0.6.2", "0.6.3", False),
    ("rust", "^0.0", "0.0.9", True),
    ("rust", "0", "0.9.0", True),
    ("rust", "<=0", "0.9.0", True),
    ("rust", ">0", "0.9.0", False),
    ("rust", ">0", "1.0.0", True),
    ("python", ">=0.4,<0.7", "0.6.9", True),
    ("python", ">=0.4,<0.7", "0.7.0", False),
    ("python", "<=0.6", "0.6.0", True),
    ("python", "<=0.6", "0.6.1", False),
    ("python", ">0.6", "0.6.1", True),
    ("python", "==0.6", "0.6.0", True),
    ("python", "==0.6", "0.6.1", False),
    ("python", "~=0.6", "0.9.0", True),
    ("python", "~=0.6.2", "0.7.0", False),
]

PEP508_TEST: list[tuple[str, str | None]] = [
    ("earthsci-ast>=0.4,<0.7", ">=0.4,<0.7"),
    ("earthsci_ast >= 0.4, < 0.7", ">= 0.4, < 0.7"),
    ("EarthSci.AST[numpy]>=0.4; python_version >= '3.9'", ">=0.4"),
    ("earthsci-ast (>=0.4,<0.7)", ">=0.4,<0.7"),
    ("earthsci-astdiff>=1", None),
]

PARSERS = {"julia": julia_range, "rust": cargo_range, "python": pep440_range}


def self_test() -> None:
    bad = 0
    for b, req, v, want in SELF_TEST:
        if contains(PARSERS[b](req), parse_version(v)) != want:
            print(
                f"self-test: {b} {req!r} should {'admit' if want else 'reject'} {v}",
                file=sys.stderr,
            )
            bad += 1
    for dep, want in PEP508_TEST:
        got = pep508_spec(dep)
        if got != want:
            print(f"self-test: PEP 508 {dep!r} gave {got!r}, want {want!r}", file=sys.stderr)
            bad += 1
    if bad:
        sys.exit(1)
    print(f"self-test: all {len(SELF_TEST) + len(PEP508_TEST)} cases pass")


# ── the manifests ────────────────────────────────────────────────────────────


def normalize(name: str) -> str:
    """A distribution name as PEP 503 compares it."""
    return re.sub(r"[-_.]+", "-", name).lower()


PEP508 = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*"
    r"(?:\[[^\]]*\])?\s*(?P<rest>[^;]*?)\s*(?:;.*)?$"
)


def pep508_spec(dep: str, name: str = "earthsci-ast") -> str | None:
    """The version specifier of a PEP 508 requirement on `name`, or None if
    the requirement is on another distribution. Extras and environment
    markers are dropped; a URL requirement is an error."""
    m = PEP508.match(dep)
    if not m or normalize(m.group("name")) != normalize(name):
        return None
    rest = m.group("rest").strip()
    if rest.startswith("@"):
        raise SystemExit(f"pyproject.toml: {dep!r} is a URL requirement, not a release range")
    if rest.startswith("(") and rest.endswith(")"):
        rest = rest[1:-1].strip()
    return rest


def manifests() -> tuple[dict, dict, dict]:
    jl = tomllib.loads((ROOT / "pkg/EarthSciASTDiff.jl/Project.toml").read_text())
    rs = tomllib.loads((ROOT / "pkg/earthsci-astdiff-rs/Cargo.toml").read_text())
    py = tomllib.loads((ROOT / "pkg/earthsci-astdiff-py/pyproject.toml").read_text())
    return jl, rs, py


def declared() -> dict[str, tuple[str, list[Interval]]]:
    jl, rs, py = manifests()
    jreq = jl["compat"]["EarthSciAST"]
    rreq = rs["dependencies"]["earthsci-ast"]
    if isinstance(rreq, dict):
        rreq = rreq["version"]
    specs = {s for s in map(pep508_spec, py["project"]["dependencies"]) if s is not None}
    if len(specs) > 1:
        raise SystemExit(
            f"pyproject.toml: earthsci-ast is required more than once: {sorted(specs)}"
        )
    preq = specs.pop() if specs else ""
    if not preq:
        raise SystemExit("pyproject.toml: earthsci-ast has no version bound")
    return {
        "julia": (jreq, julia_range(jreq)),
        "rust": (rreq, cargo_range(rreq)),
        "python": (preq, pep440_range(preq)),
    }


def python_floor() -> str:
    _, _, py = manifests()
    m = re.fullmatch(r"\s*>=\s*(\d+\.\d+)\s*", py["project"].get("requires-python", ""))
    if not m:
        raise SystemExit("pyproject.toml: requires-python must be a plain `>=X.Y` floor")
    return m.group(1)


# ── the registries ───────────────────────────────────────────────────────────


def fetch(url: str, accept: str | None = None, attempts: int = 4) -> bytes:
    """GET `url`, retrying a timeout, a dropped connection, 429 or a 5xx
    with exponential backoff; any other HTTP error fails at once."""
    headers = {"User-Agent": "EarthSciASTDiff-ci (https://github.com/EarthSciML/EarthSciASTDiff)"}
    if accept:
        headers["Accept"] = accept
    for i in range(attempts):
        last = i == attempts - 1
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if last or (e.code != 429 and e.code < 500):
                raise
            err: Exception = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if last:
                raise
            err = e
        delay = 2 ** (i + 1)
        print(f"warning: {url}: {err}; retrying in {delay}s", file=sys.stderr)
        time.sleep(delay)
    raise AssertionError("unreachable")


def plain(versions) -> list[Version]:
    out = set()
    for s in versions:
        try:
            out.add(parse_version(s))
        except ValueError:  # pre-release or local version
            pass
    return sorted(out)


def pypi_file_version(filename: str) -> str | None:
    """The version in a wheel or sdist filename (PEP 427 / PEP 625)."""
    if filename.endswith(".whl"):
        return filename.split("-")[1]
    for ext in (".tar.gz", ".zip"):
        if filename.endswith(ext):
            return filename[: -len(ext)].rsplit("-", 1)[1]
    return None


def released() -> dict[str, list[Version]]:
    general = tomllib.loads(
        fetch(
            "https://raw.githubusercontent.com/JuliaRegistries/General/master/E/EarthSciAST/Versions.toml"
        ).decode()
    )
    crates = json.loads(fetch("https://crates.io/api/v1/crates/earthsci-ast"))
    # The Simple API (PEP 691 JSON), not the legacy /pypi/<name>/json whose
    # `releases` key is deprecated. A release counts when at least one of its
    # files is not yanked (PEP 592: `yanked` is false or a reason string).
    simple = json.loads(
        fetch("https://pypi.org/simple/earthsci-ast/", accept="application/vnd.pypi.simple.v1+json")
    )
    pyvers = {
        v
        for f in simple["files"]
        if f.get("yanked") is False and (v := pypi_file_version(f["filename"])) is not None
    }
    return {
        "julia": plain(v for v, info in general.items() if not info.get("yanked")),
        "rust": plain(v["num"] for v in crates["versions"] if not v["yanked"]),
        "python": plain(pyvers),
    }


# ── testing against an EarthSciAST checkout ──────────────────────────────────


def admit_checkout(checkout: Path) -> None:
    """Widen the Julia and Cargo manifests, in place, to admit the
    EarthSciAST checkout's own version when the declared range does not.

    Without this, the day EarthSciAST `main` bumps its version ahead of a
    release, Julia's `Pkg.develop` refuses the checkout as outside [compat]
    and Cargo silently ignores the `[patch.crates-io]` (so the Rust tests run
    against crates.io and the docs examples see two `EsmFile` types). Neither
    is a regression here, and the fix is NOT to raise the range to an
    unreleased version. Python needs nothing: the upstream jobs install the
    checkout and import this package from source, so pip never checks the
    requirement."""
    decl = declared()
    jl = tomllib.loads((checkout / "pkg/EarthSciAST.jl/Project.toml").read_text())
    rs = tomllib.loads((checkout / "pkg/earthsci-ast-rs/Cargo.toml").read_text())
    jv, rv = jl["version"], rs["package"]["version"]

    def admitted(binding: str, v: str) -> bool:
        try:
            return contains(decl[binding][1], parse_version(v))
        except ValueError:  # a pre-release (`0.7.0-DEV`) is never admitted
            return False

    if admitted("julia", jv) and admitted("rust", rv):
        print(f"EarthSciAST checkout ({jv} / {rv}) is inside the declared ranges")
    if not admitted("julia", jv):
        # Drop the entry: a pre-release cannot be written as Julia compat.
        path = ROOT / "pkg/EarthSciASTDiff.jl/Project.toml"
        new, n = re.subn(
            r'(?m)^(\[compat\]\n(?:(?!\[).*\n)*?)EarthSciAST\s*=\s*"[^"]*"\n',
            r"\1",
            path.read_text(),
        )
        if n != 1:
            raise SystemExit(f"{path}: no [compat] EarthSciAST entry to drop")
        path.write_text(new)
        print(
            f"::notice::EarthSciAST checkout is {jv}, outside Julia compat "
            f"{decl['julia'][0]!r}: dropped that compat entry for this run only"
        )
    if not admitted("rust", rv):
        path = ROOT / "pkg/earthsci-astdiff-rs/Cargo.toml"
        new, n = re.subn(
            r'(?m)^(earthsci-ast\s*=\s*(?:\{[^}\n]*\bversion\s*=\s*)?)"[^"]*"',
            rf'\1"={rv}"',
            path.read_text(),
            count=1,
        )
        if n != 1:
            raise SystemExit(f"{path}: no earthsci-ast requirement to rewrite")
        path.write_text(new)
        print(
            f"::notice::EarthSciAST checkout is {rv}, outside Cargo requirement "
            f"{decl['rust'][0]!r}: required ={rv} for this run only"
        )


# ── main ─────────────────────────────────────────────────────────────────────


def ends(lo: Version, hi: Version) -> list[tuple[str, Version]]:
    if lo == hi:
        return [(f"{show(lo)}, lowest = latest", lo)]
    return [(f"{show(lo)}, lowest", lo), (f"{show(hi)}, latest", hi)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--github-output", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--admit-checkout", type=Path, metavar="DIR")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.admit_checkout:
        admit_checkout(args.admit_checkout)
        return

    decl = declared()
    rel = released()
    pyfloor = python_floor()
    matrices: dict[str, list[dict]] = {}
    failed = False
    rows = []
    for binding, (req, rng) in decl.items():
        ok = [v for v in rel[binding] if contains(rng, v)]
        if not ok:
            print(
                f"error: {binding}: no published EarthSciAST release satisfies {req!r} "
                f"(published: {', '.join(map(show, rel[binding]))})",
                file=sys.stderr,
            )
            failed = True
            continue
        lo, hi = ok[0], ok[-1]
        if binding == "python":
            legs = [
                {
                    "label": f"{label}; Python {py}",
                    "version": show(v),
                    "python": py,
                    "os": PYTHON_FLOOR_RUNNER if py == pyfloor else "ubuntu-latest",
                }
                for label, v in ends(lo, hi)
                for py in dict.fromkeys([pyfloor, PYTHON_NEWEST])
            ]
        else:
            legs = [{"label": label, "version": show(v)} for label, v in ends(lo, hi)]
        matrices[binding] = legs
        rows.append(f"| {binding} | `{req}` | {show(lo)} | {show(hi)} |")
    table = "\n".join(["| binding | declared | lowest | latest |", "|---|---|---|---|", *rows])
    print(table)
    print(f"\nThe Python binding runs on Python {pyfloor} and {PYTHON_NEWEST}.")
    if failed:
        sys.exit(1)
    if args.github_output:
        lines = "".join(f"{binding}={json.dumps(legs)}\n" for binding, legs in matrices.items())
        out = os.environ.get("GITHUB_OUTPUT")
        if not out:
            print(
                "\n$GITHUB_OUTPUT is not set (GitHub Actions sets it); "
                "the matrices it would receive:\n" + lines,
                end="",
            )
            return
        with open(out, "a") as f:
            f.write(lines)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as f:
                f.write("### EarthSciAST releases under test\n\n" + table + "\n")


if __name__ == "__main__":
    main()
