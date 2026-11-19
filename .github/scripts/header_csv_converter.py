#!/usr/bin/env python3

"""Convert between GPUParameters.csv and a flat launch-bounds header, in either
direction. Direction is guessed from which of --from/--to ends in .csv."""

import argparse
import csv
import os
import re
import sys

SECTIONS = ("CORE", "LB", "PAR")
MACRO_PREFIX = {"CORE": "GPUCA_", "LB": "GPUCA_LB_", "PAR": "GPUCA_PAR_"}

# Macros that are not real per-architecture tunables (pure defaults / symbolic
# indirections) - never merge these back into the CSV even if present in a header.
HEADER_EXCLUDE_PATTERNS = [
    r"^GPUCA_LB_GPUTPCCompressionGatherKernels_.*",
    r"^GPUCA_LB_GPUTPCNNClusterizerKernels_.*",
    r"^GPUCA_LB_GPUTPCDecompressionUtilKernels_.*",
    r"^GPUCA_LB_GPUTrackingRefitKernel_.*",
    r"^GPUCA_LB_GPUTRDTrackerKernels_.*",
    r"^GPUCA_LB_GPUMemClean16",
    r"^GPUCA_LB_GPUitoa",
]
DEFINE_RE = re.compile(r"#define\s+(\w+)\s+(.+)")


def fail(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def is_csv(path):
    return path.lower().endswith(".csv")


def read_csv_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        fail(f"Empty CSV file: {path}")
    return rows


def find_column(header_row, name):
    try:
        return header_row.index(name)
    except ValueError:
        fail(f"Unknown column: {name}. Available: {', '.join(header_row[1:])}")


def matches_any(text, patterns):
    return any(re.search(p, text) for p in patterns)


# --- CSV -> header -----------------------------------------------------------

def format_value_for_header(raw):
    val = raw.strip()
    if val.startswith("[") and val.endswith("]"):
        val = val[1:-1].strip()
    return val.replace('"', "")


def csv_to_header(csv_path, arch, out_path):
    rows = read_csv_rows(csv_path)
    header_row = rows[0]
    arch_idx = find_column(header_row, arch)
    default_idx = find_column(header_row, "default")

    current_section = None
    lines = []
    for row in rows[1:]:
        if not row or row[0].strip() == "":
            continue
        key = row[0].strip()

        if key.endswith(":"):
            section_name = key[:-1]
            current_section = section_name if section_name in SECTIONS else None
            continue

        if current_section is None:
            continue

        arch_cell = row[arch_idx].strip() if arch_idx < len(row) else ""
        default_cell = row[default_idx].strip() if default_idx < len(row) else ""
        raw_value = arch_cell or default_cell
        if raw_value == "":
            continue

        value = format_value_for_header(raw_value)
        macro = f"{MACRO_PREFIX[current_section]}{key}"
        lines.append(f"#define {macro} {value}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Wrote {len(lines)} #define(s) to {out_path}", file=sys.stderr)


# --- header -> CSV -------------------------------------------------------------

def parse_header_defines(header_path):
    defines = {}
    with open(header_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#define"):
                continue
            m = DEFINE_RE.match(line)
            if not m:
                continue
            raw_name, value = m.groups()
            value = value.strip()

            if matches_any(raw_name, HEADER_EXCLUDE_PATTERNS):
                continue

            if raw_name.startswith("GPUCA_LB_"):
                section, name = "LB", raw_name[len("GPUCA_LB_"):]
            elif raw_name.startswith("GPUCA_PAR_"):
                section, name = "PAR", raw_name[len("GPUCA_PAR_"):]
            else:
                continue

            if re.match(r"^\d+\s*,\s*\d+$", value):
                nums = [int(x.strip()) for x in value.split(",")]
                cell = f"[{nums[0]}, {nums[1]}]"
            elif re.match(r"^\d+$", value):
                cell = value
            else:
                cell = f'"{value}"'

            defines[(section, name)] = {"cell": cell, "raw": raw_name, "matched": False}
    return defines


def header_to_csv(header_path, arch, csv_path):
    # csv_path is both the existing base to merge into and the output destination.
    defines = parse_header_defines(header_path)
    rows = read_csv_rows(csv_path)
    header_row = rows[0]
    arch_idx = find_column(header_row, arch)

    current_section = None
    changes = []
    for row in rows[1:]:
        if not row or row[0].strip() == "":
            continue
        key = row[0].strip()

        if key.endswith(":"):
            section_name = key[:-1]
            current_section = section_name if section_name in ("LB", "PAR") else None
            continue

        if current_section is None:
            continue

        match_key = (current_section, key)
        if match_key not in defines:
            continue

        defines[match_key]["matched"] = True
        new_value = defines[match_key]["cell"]
        old_value = row[arch_idx].strip() if arch_idx < len(row) else ""

        while len(row) <= arch_idx:
            row.append("")
        row[arch_idx] = new_value

        if old_value != new_value:
            changes.append((key, old_value or "(empty)", new_value))

    print(f"[INFO] {len(changes)} row(s) changed for column '{arch}':", file=sys.stderr)
    for key, old_value, new_value in changes:
        print(f"  {key}: {old_value} -> {new_value}", file=sys.stderr)

    unmatched = [d["raw"] for d in defines.values() if not d["matched"]]
    if unmatched:
        print(f"[WARNING] Header defines with no matching CSV row (ignored): {', '.join(unmatched)}", file=sys.stderr)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in rows:
            writer.writerow(row)
    print(f"[INFO] Wrote updated CSV to {csv_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_path", required=True, help="Source file (.csv, or a header)")
    parser.add_argument("--to", dest="to_path", required=True, help="Destination file (.csv, or a header); written in place")
    parser.add_argument("--arch", required=True, help="Architecture column to read/write, e.g. RDNA, HOPPER")
    args = parser.parse_args()

    if not os.path.isfile(args.from_path):
        fail(f"--from file does not exist: {args.from_path}")

    from_is_csv = is_csv(args.from_path)
    to_is_csv = is_csv(args.to_path)

    if from_is_csv and not to_is_csv:
        csv_to_header(args.from_path, args.arch, args.to_path)
    elif not from_is_csv and to_is_csv:
        if not os.path.isfile(args.to_path):
            fail(f"--to file does not exist (it's the CSV to merge into): {args.to_path}")
        header_to_csv(args.from_path, args.arch, args.to_path)
    elif from_is_csv and to_is_csv:
        fail("Both --from and --to look like .csv files - exactly one side must be the CSV")
    else:
        fail("Neither --from nor --to looks like a .csv file - exactly one side must be the CSV")


if __name__ == "__main__":
    main()
