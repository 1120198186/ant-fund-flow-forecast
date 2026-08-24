#!/usr/bin/env python3
"""Build a daily aggregate holdout from a tracked split definition."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    split_path = args.split if args.split.is_absolute() else project_root / args.split
    split = json.loads(split_path.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(split["holdout_window"]["start"])
    end = datetime.fromisoformat(split["holdout_window"]["end"])
    targets = split["targets"]

    source_path = project_root / split["source"]
    daily: dict[str, dict[str, int]] = {}
    with source_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = parse_date(row["report_date"])
            if start <= day <= end:
                values = daily.setdefault(
                    row["report_date"],
                    {target: 0 for target in targets},
                )
                for target in targets:
                    values[target] += int(row[target] or 0)

    expected_days = (end - start).days + 1
    if len(daily) != expected_days:
        raise RuntimeError(
            f"expected {expected_days} holdout days, found {len(daily)}"
        )

    output = (
        project_root
        / "validation"
        / "locked"
        / f"{split['split_id']}.csv"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite locked labels: {output}")
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["report_date", *targets],
        )
        writer.writeheader()
        for report_date in sorted(daily):
            writer.writerow({"report_date": report_date, **daily[report_date]})

    print(output)


if __name__ == "__main__":
    main()

