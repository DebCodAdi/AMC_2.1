"""A/B comparison with R3 containment check and McNemar exact test.

Compares two JSONL result files and reports:
  - Files gained (locked after, not before)
  - Files lost (locked before, not after)  ← R3 VIOLATION if > 0
  - Payload changes (locked both times but different bytes)  ← R4 VIOLATION
  - McNemar exact test p-value and delta
  - Wilson score confidence intervals for both rates

Usage::

    python tools/ab_compare.py before.jsonl after.jsonl

Returns exit code 0 if R3 holds and the improvement is statistically
significant (McNemar p < 0.01 and gained > 0), non-zero otherwise.

Spec reference: §6.4, §2.12, R3, R4 (PS26147).
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path


def _load(path: str | Path) -> dict[str, dict]:
    """Load a JSONL file and return {file: result_dict}."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["file"]] = r
    return out


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval.  Returns (proportion, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact (sign) test p-value for McNemar.  Two-sided.

    b = files locked before only (losses), c = files locked after only (gains).
    """
    try:
        from scipy.stats import binom
        n = b + c
        if n == 0:
            return 1.0
        # Two-sided: P(|B - C| >= |b - c|) under H0: p=0.5
        k_obs = max(b, c)
        p = 2 * binom.sf(k_obs - 1, n, 0.5)
        return float(min(1.0, p))
    except ImportError:
        # Fallback approximation (chi² continuity corrected)
        n = b + c
        if n == 0:
            return 1.0
        chi2 = (abs(b - c) - 1) ** 2 / n
        # Approximate p from chi²(1) CDF
        import math
        return math.erfc(math.sqrt(chi2 / 2))


def compare(before_path: str | Path,
            after_path: str | Path,
            verbose: bool = True) -> bool:
    """Compare two JSONL result sets.

    Returns True if R3 holds and the improvement is statistically
    significant.  Prints a detailed report.
    """
    B = _load(before_path)
    A = _load(after_path)

    def locked(r: dict) -> bool:
        return r.get("status") == "SUCCESS_CRC_LOCKED"

    def payload(r: dict) -> str | None:
        return r.get("payload_hex") or r.get("payload_b64")

    common = sorted(set(B) & set(A))
    n = len(common)

    gained = [f for f in common if not locked(B[f]) and locked(A[f])]
    lost   = [f for f in common if locked(B[f]) and not locked(A[f])]
    changed_payload = [
        f for f in common
        if locked(B[f]) and locked(A[f]) and payload(B[f]) != payload(A[f])
    ]

    # ── Totals ─────────────────────────────────────────────────────────────
    before_locked = sum(1 for f in common if locked(B[f]))
    after_locked  = sum(1 for f in common if locked(A[f]))

    before_p, blo, bhi = _wilson_ci(before_locked, n)
    after_p,  alo, ahi = _wilson_ci(after_locked,  n)

    mcnemar_p = _mcnemar_exact(len(lost), len(gained))

    if verbose:
        print("\n" + "=" * 60)
        print("A/B COMPARISON — PS26147 §6.4")
        print("=" * 60)
        print(f"Before: {before_path}")
        print(f"After:  {after_path}")
        print(f"Files compared: {n}\n")
        print(f"  Before: {before_locked}/{n} = {before_p:.1%}  "
              f"[{blo:.1%}, {bhi:.1%}]  (Wilson 95%)")
        print(f"  After:  {after_locked}/{n} = {after_p:.1%}  "
              f"[{alo:.1%}, {ahi:.1%}]  (Wilson 95%)")
        print(f"\n  Gained: {len(gained)}   Lost: {len(lost)}   "
              f"Payload-changed: {len(changed_payload)}")
        print(f"  McNemar exact p = {mcnemar_p:.2e}   "
              f"delta = {after_p - before_p:+.1%}")

        if lost:
            print(f"\n  *** R3 VIOLATION — regressed files ({len(lost)}): ***")
            for f in lost[:10]:
                print(f"      {f}")
            if len(lost) > 10:
                print(f"      … and {len(lost) - 10} more")

        if changed_payload:
            print(f"\n  *** R4 VIOLATION — payload changed ({len(changed_payload)}): ***")
            for f in changed_payload[:10]:
                print(f"      {f}")
            if len(changed_payload) > 10:
                print(f"      … and {len(changed_payload) - 10} more")

        # Significance verdict
        print()
        if lost:
            print("  VERDICT: FAIL — R3 VIOLATION (regressions present)")
        elif changed_payload:
            print("  VERDICT: FAIL — R4 VIOLATION (payload changed)")
        elif len(gained) == 0:
            print("  VERDICT: NEUTRAL — no improvement")
        elif mcnemar_p >= 0.01:
            print(f"  VERDICT: NOT SIGNIFICANT (p={mcnemar_p:.2e} ≥ 0.01)")
        else:
            print(f"  VERDICT: PASS — improvement significant and regression-free")
        print("=" * 60 + "\n")

    r3_ok = len(lost) == 0
    r4_ok = len(changed_payload) == 0
    significant = len(gained) > 0 and mcnemar_p < 0.01
    return r3_ok and r4_ok and significant


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="A/B comparison with R3 containment — PS26147 §6.4"
    )
    p.add_argument("before", help="JSONL results before the change")
    p.add_argument("after",  help="JSONL results after the change")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress detailed output")
    args = p.parse_args(argv)
    ok = compare(args.before, args.after, verbose=not args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
