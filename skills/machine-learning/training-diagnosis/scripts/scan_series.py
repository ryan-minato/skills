"""Find the first sustained anomaly in a metric series (the T0 candidate).

Reads a CSV with a header, computes for one column a robust z-score against
a trailing window's median and median absolute deviation, and reports the
first row at which the score stays beyond a threshold for K of the last M
rows. With --group-by, rows are grouped (for example by rank) and the
ratio of the slowest to the median group is reported per step in
addition, using --step to align groups. Non-finite values (NaN, inf)
are never fed into the statistics: the first one is reported as the
anomaly when it precedes the first sustained deviation, and their rows
are counted under "non_finite_rows".

Usage:
    python3 scan_series.py --input series.csv --column loss
    python3 scan_series.py --input series.csv --column step_time_s --group-by rank --step step

Output: one JSON object on stdout. Diagnostics go to stderr.

Exit codes:
    0  success (an anomaly may or may not have been found; see "first_anomaly")
    1  the input has no usable rows, or a named column is absent or non-numeric throughout
    2  bad arguments (unknown option, missing --input/--column, an unreadable or undecodable
       file, invalid numbers)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MAD_SCALE = 1.4826  # makes MAD comparable to a standard deviation for normal data


def robust_z(values: list[float], window: int) -> list[float | None]:
    """Robust z of each value against the trailing window (excluding the value itself)."""
    out: list[float | None] = []
    for i, x in enumerate(values):
        history = values[max(0, i - window) : i]
        if len(history) < max(3, window // 2):
            out.append(None)
            continue
        med = statistics.median(history)
        mad = statistics.median(abs(h - med) for h in history)
        scale = MAD_SCALE * mad
        if scale == 0.0:
            scale = 1e-12 if x != med else 1.0
        out.append((x - med) / scale)
    return out


def first_sustained(scores: list[float | None], threshold: float, k: int, m: int) -> int | None:
    """Index of the first row from which K of the last M scores exceed the threshold."""
    flags = [s is not None and abs(s) > threshold for s in scores]
    for i in range(len(flags)):
        recent = flags[max(0, i - m + 1) : i + 1]
        if sum(recent) >= k and flags[i]:
            # walk back to the first flagged row of this burst
            j = i
            while j > 0 and flags[j - 1]:
                j -= 1
            return j
    return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(rows: list[dict[str, str]], column: str) -> tuple[list[float | None], list[int]]:
    """Finite values (None where absent or non-numeric) and the row indexes holding NaN or inf."""
    values: list[float | None] = []
    non_finite: list[int] = []
    for i, row in enumerate(rows):
        raw = (row.get(column) or "").strip()
        try:
            value = float(raw)
        except ValueError:
            values.append(None)
            continue
        if math.isfinite(value):
            values.append(value)
        else:
            values.append(None)
            non_finite.append(i)
    return values, non_finite


def first_row_of_step(rows: list[dict[str, str]], step_column: str, step: str) -> int | None:
    """The input row index at which `step` first appears in `step_column`."""
    for i, row in enumerate(rows):
        if row.get(step_column, str(i)) == step:
            return i
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scan_series.py",
        description="Find the first sustained anomaly in a metric series (the T0 candidate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 success; 1 no usable rows or a named column absent/non-numeric;\n"
            "2 bad arguments (unknown option, missing --input/--column, unreadable file).\n\n"
            "Example: python3 scan_series.py --input series.csv --column loss --window 50 --z 4 --k 3 --m 5"
        ),
    )
    parser.add_argument("--input", required=True, help="CSV file with a header row")
    parser.add_argument("--column", required=True, help="numeric column to scan")
    parser.add_argument("--step", default="step", help="step column for output and grouping (default: step)")
    parser.add_argument("--group-by", default=None, help="column that identifies groups (e.g. rank)")
    parser.add_argument("--window", type=int, default=50, help="trailing window size in rows (default: 50)")
    parser.add_argument("--z", type=float, default=4.0, help="robust z threshold (default: 4.0)")
    parser.add_argument("--k", type=int, default=3, help="rows beyond the threshold required (default: 3)")
    parser.add_argument("--m", type=int, default=5, help="among the last M rows (default: 5)")
    args = parser.parse_args(argv)

    if args.window < 3 or args.k < 1 or args.m < args.k:
        parser.error("--window must be >= 3, --k >= 1, and --m >= --k")
    path = Path(args.input)
    if not path.is_file():
        parser.error(f"--input {path} is not a readable file")

    try:
        rows = read_rows(path)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        print(f"Error: cannot read {path} as UTF-8 CSV ({exc}); pass a readable UTF-8 CSV file.", file=sys.stderr)
        return 2
    if not rows:
        print(f"Error: {path} has no data rows; nothing to scan.", file=sys.stderr)
        return 1
    if args.column not in rows[0]:
        print(
            f"Error: column '{args.column}' is not in {path} (columns: {', '.join(rows[0].keys())}); "
            "pass --column with a header name.",
            file=sys.stderr,
        )
        return 1

    result: dict = {
        "input": str(path),
        "column": args.column,
        "rows": len(rows),
        "window": args.window,
        "z_threshold": args.z,
        "k": args.k,
        "m": args.m,
        "first_anomaly": None,
        "non_finite_rows": 0,
    }

    values, non_finite = to_float(rows, args.column)
    result["non_finite_rows"] = len(non_finite)
    first_non_finite = non_finite[0] if non_finite else None

    if args.group_by:
        for name in (args.group_by, args.step):
            if name not in rows[0]:
                print(
                    f"Error: column '{name}' is not in {path}; --group-by needs both the group and the "
                    f"--step column (columns: {', '.join(rows[0].keys())}).",
                    file=sys.stderr,
                )
                return 1
        groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
        by_step: dict[str, list[float]] = defaultdict(list)
        for row, value in zip(rows, values, strict=True):
            if value is None:
                continue
            groups[row[args.group_by]].append((row.get(args.step, ""), value))
            by_step[row.get(args.step, "")].append(value)
        if not by_step and first_non_finite is None:
            print(f"Error: column '{args.column}' has no numeric values.", file=sys.stderr)
            return 1
        skew = []
        for step, values in by_step.items():
            med = statistics.median(values)
            skew.append({"step": step, "max_over_median": (max(values) / med) if med else None, "groups": len(values)})
        result["group_by"] = args.group_by
        result["groups"] = sorted(groups)
        result["skew"] = skew
        ranked = [s for s in skew if s["max_over_median"] is not None]
        result["max_skew"] = max(ranked, key=lambda s: s["max_over_median"], default=None)
        series = [statistics.median(v) for v in by_step.values()]
        steps = list(by_step.keys())
    else:
        indexed = enumerate(zip(rows, values, strict=True))
        pairs = [(row.get(args.step, str(i)), v) for i, (row, v) in indexed if v is not None]
        if not pairs and first_non_finite is None:
            print(f"Error: column '{args.column}' has no numeric values.", file=sys.stderr)
            return 1
        steps = [p[0] for p in pairs]
        series = [p[1] for p in pairs]

    scores = robust_z(series, args.window)
    idx = first_sustained(scores, args.z, args.k, args.m)
    z_row = first_row_of_step(rows, args.step, steps[idx]) if idx is not None else None
    # A non-finite value is the anomaly whenever it comes first: the run's
    # numbers stopped being numbers there, whatever the statistics say later.
    if first_non_finite is not None and (z_row is None or first_non_finite <= z_row):
        row = rows[first_non_finite]
        result["first_anomaly"] = {
            "kind": "non_finite",
            "row_index": first_non_finite,
            "step": row.get(args.step, str(first_non_finite)),
            "value": (row.get(args.column) or "").strip(),
        }
    elif idx is not None:
        result["first_anomaly"] = {
            "kind": "robust_z",
            "row_index": idx,
            "step": steps[idx],
            "value": series[idx],
            "robust_z": scores[idx],
            "direction": "up" if scores[idx] is not None and scores[idx] > 0 else "down",
        }
    result["max_abs_z"] = max((abs(s) for s in scores if s is not None), default=None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
