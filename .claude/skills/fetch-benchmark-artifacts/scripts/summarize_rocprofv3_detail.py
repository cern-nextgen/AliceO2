#!/usr/bin/env python3
"""Group a raw rocprofv3 '<name>_detail.csv' (one row per dispatch x counter)
into per-(Kernel_Name, Counter_Name) mean/min/max/stdev, and flag any counter
that never varies across dispatches -- a sign it wasn't actually collected on
this GPU/driver/capability combination, not a genuine measurement."""

import argparse
import csv
import statistics
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="raw rocprofv3 counter_collection CSV")
    args = parser.parse_args()

    by_key = defaultdict(list)
    with open(args.input) as f:
        for row in csv.DictReader(f):
            key = (row["Kernel_Name"], row["Counter_Name"])
            try:
                by_key[key].append(float(row["Counter_Value"]))
            except ValueError:
                pass

    if not by_key:
        print(f"No (Kernel_Name, Counter_Name) rows found in {args.input} "
              "-- is this really a rocprofv3 raw detail CSV, not an ncu one?",
              file=sys.stderr)
        sys.exit(1)

    kernels = sorted({k[0] for k in by_key})
    counters = sorted({k[1] for k in by_key})

    print(f"{'kernel':30s} {'counter':20s} {'n':>5s} {'mean':>14s} {'min':>14s} {'max':>14s}  flag")
    flagged = []
    for kernel in kernels:
        for counter in counters:
            vals = by_key.get((kernel, counter))
            if not vals:
                continue
            n = len(vals)
            mean = statistics.mean(vals)
            lo, hi = min(vals), max(vals)
            flag = ""
            # Zero variance alone isn't suspicious -- a launch-geometry metric like
            # Wavefronts is *expected* to be identical across dispatches of the same
            # kernel, since it's derived from grid/workgroup size, not measured.
            # What's actually suspicious, based on real failures seen on AMD hardware
            # in this repo, is landing on one of two specific sentinel values every
            # single time: a flat 0.0 (GRBM/GL2C-block counters silently uncollected)
            # or a flat 100.0 (seen once for LDSBankConflict). Anything else constant
            # is more likely a genuine fixed property of the launch than dead data.
            if n > 1 and lo == hi and mean in (0.0, 100.0):
                flag = f"<-- flat {mean:g} every dispatch: likely not actually collected, not a real measurement"
                flagged.append((kernel, counter, mean))
            print(f"{kernel:30s} {counter:20s} {n:5d} {mean:14.6g} {lo:14.6g} {hi:14.6g}  {flag}")

    if flagged:
        print(f"\n{len(flagged)} counter(s) are flat at a known sentinel value on every dispatch:", file=sys.stderr)
        for kernel, counter, val in flagged:
            print(f"  {kernel} / {counter} = {val} for every single dispatch", file=sys.stderr)
        print("Treat these as dead/uncollected, not genuine measurements, unless you've "
              "separately confirmed this GPU/driver/capability combination actually "
              "supports collecting them (see SKILL.md). Other constant columns (e.g. "
              "Wavefronts) are not flagged -- a fixed, non-sentinel value can legitimately "
              "be the same every dispatch when it's derived from launch geometry rather "
              "than measured.", file=sys.stderr)


if __name__ == "__main__":
    main()
