"""Run all six phase smoke tests in sequence and report a summary.

Each smoke test fabricates tiny data and launches a real (2-3 step) training /
inference for its phase, so this is an end-to-end check that the whole Task 2
pipeline runs on your torch/GPU install — without needing the challenge dataset.

    python tests/run_all_smoke.py            # run all
    python tests/run_all_smoke.py 2 3 5      # run only phases 2, 3, 5
    python tests/run_all_smoke.py --stop      # stop at first failure

Exit code is 0 only if every selected phase passes.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
PHASES = [1, 2, 3, 4, 5, 6]


def main():
    ap = argparse.ArgumentParser(description="Run all phase smoke tests")
    ap.add_argument("phases", nargs="*", type=int, help="subset e.g. 2 3 5 (default: all)")
    ap.add_argument("--stop", action="store_true", help="stop at first failure")
    args = ap.parse_args()

    phases = args.phases or PHASES
    invalid = [p for p in phases if p not in PHASES]
    if invalid:
        sys.exit(f"Unknown phase(s): {invalid}. Valid: {PHASES}")

    results = {}
    total_t0 = time.time()
    for p in phases:
        script = TESTS / f"smoke_fase{p}.py"
        if not script.exists():
            results[p] = ("MISSING", 0.0)
            print(f"\n{'#'*70}\n# FASE {p} — SKIPPED (missing {script.name})\n{'#'*70}")
            continue
        print(f"\n{'#'*70}\n# FASE {p} smoke test\n{'#'*70}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(script)])
        dt = time.time() - t0
        ok = r.returncode == 0
        results[p] = ("PASS" if ok else "FAIL", dt)
        if not ok and args.stop:
            print(f"\nStopping: FASE {p} failed (--stop).")
            break

    # summary
    print(f"\n{'='*70}\nSMOKE TEST SUMMARY\n{'='*70}")
    all_ok = True
    for p in phases:
        if p not in results:
            continue
        status, dt = results[p]
        mark = "[OK]" if status == "PASS" else ("[--]" if status == "MISSING" else "[XX]")
        print(f"  {mark}  Fase {p}: {status:<8} ({dt:5.1f}s)")
        all_ok = all_ok and status == "PASS"
    print(f"{'='*70}")
    print(f"Total: {time.time() - total_t0:.1f}s | {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
