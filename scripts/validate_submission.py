#!/usr/bin/env python3
"""Validate a submission against the tracked competition contract."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys


def expected_dates(start: str, rows: int) -> list[str]:
    first = datetime.fromisoformat(start)
    return [
        (first + timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(rows)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (project_root / "docs" / "competition_contract.json").read_text(
            encoding="utf-8"
        )
    )
    forecast = contract["forecast"]
    required_dates = expected_dates(forecast["start"], forecast["rows"])
    failures: list[str] = []

    with args.submission.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if len(rows) != forecast["rows"]:
        failures.append(
            f"expected {forecast['rows']} rows, found {len(rows)}"
        )

    dates: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        if len(row) != 3:
            failures.append(
                f"line {line_number}: expected 3 columns, found {len(row)}"
            )
            continue
        report_date, purchase, redeem = row
        dates.append(report_date)
        if not report_date.isdigit() or len(report_date) != 8:
            failures.append(f"line {line_number}: invalid date {report_date!r}")
        for name, value in [("purchase", purchase), ("redeem", redeem)]:
            if not value.isdigit():
                failures.append(
                    f"line {line_number}: {name} must be a non-negative integer"
                )

    if dates != required_dates:
        failures.append(
            "dates must be unique and ascending from 20140901 through 20140930"
        )

    if failures:
        print("SUBMISSION CHECK FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("SUBMISSION CHECK PASSED: 30 rows, 3 columns, no header")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

