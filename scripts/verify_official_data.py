#!/usr/bin/env python3
"""Verify local official CSV files against the tracked SHA-256 manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys


EXPECTED_HEADERS: dict[str, list[str] | None] = {
    "comp_predict_table.csv": None,
    "mfd_bank_shibor.csv": [
        "mfd_date",
        "Interest_O_N",
        "Interest_1_W",
        "Interest_2_W",
        "Interest_1_M",
        "Interest_3_M",
        "Interest_6_M",
        "Interest_9_M",
        "Interest_1_Y",
    ],
    "mfd_day_share_interest.csv": [
        "mfd_date",
        "mfd_daily_yield",
        "mfd_7daily_yield",
    ],
    "user_balance_table.csv": [
        "user_id",
        "report_date",
        "tBalance",
        "yBalance",
        "total_purchase_amt",
        "direct_purchase_amt",
        "purchase_bal_amt",
        "purchase_bank_amt",
        "total_redeem_amt",
        "consume_amt",
        "transfer_amt",
        "tftobal_amt",
        "tftocard_amt",
        "share_amt",
        "category1",
        "category2",
        "category3",
        "category4",
    ],
    "user_profile_table.csv": [
        "user_id",
        "sex",
        "city",
        "constellation",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw" / "official"
    manifest_path = raw_root / "manifest.sha256"
    inventory_path = raw_root / "inventory.csv"

    expected_hashes: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        expected_hashes[name.lstrip(" *")] = digest

    failures: list[str] = []
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        inventory_rows = list(csv.DictReader(handle))
    inventory = {row["file"]: row for row in inventory_rows}
    if len(inventory) != len(inventory_rows):
        failures.append("inventory contains duplicate file rows")

    manifest_names = set(expected_hashes)
    inventory_names = set(inventory)
    if inventory_names != manifest_names:
        missing = sorted(manifest_names - inventory_names)
        extra = sorted(inventory_names - manifest_names)
        failures.extend(f"inventory missing: {name}" for name in missing)
        failures.extend(f"inventory unexpected: {name}" for name in extra)

    for name, expected in expected_hashes.items():
        path = raw_root / name
        if not path.is_file():
            failures.append(f"missing: {name}")
            continue
        if name not in inventory:
            continue
        if name not in EXPECTED_HEADERS:
            failures.append(f"no frozen schema for: {name}")
            continue

        row = inventory[name]
        actual = sha256(path)
        if actual != expected:
            failures.append(f"hash mismatch: {name}")
        if row["sha256"] != expected:
            failures.append(f"inventory/manifest hash mismatch: {name}")

        expected_size = int(row["bytes"])
        if path.stat().st_size != expected_size:
            failures.append(f"size mismatch: {name}")

        declared = row["has_header"].strip().lower()
        if declared not in {"true", "false"}:
            failures.append(f"invalid has_header value: {name}")
            continue
        declared_header = declared == "true"
        expected_header = EXPECTED_HEADERS[name]
        if declared_header != (expected_header is not None):
            failures.append(f"header declaration mismatch: {name}")

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            first_row = next(reader, None)
            if expected_header is not None:
                if first_row != expected_header:
                    failures.append(f"header content mismatch: {name}")
                data_rows = sum(1 for _ in reader)
                logical_rows = data_rows + (1 if first_row is not None else 0)
            else:
                data_rows = (1 if first_row is not None else 0) + sum(
                    1 for _ in reader
                )
                logical_rows = data_rows

        if data_rows != int(row["data_rows"]):
            failures.append(
                f"data row mismatch: {name} "
                f"(inventory={row['data_rows']}, actual={data_rows})"
            )
        if logical_rows != int(row["line_count"]):
            failures.append(
                f"logical row mismatch: {name} "
                f"(inventory={row['line_count']}, actual={logical_rows})"
            )

    unexpected = sorted(
        path.name
        for path in raw_root.glob("*.csv")
        if path.name not in expected_hashes and path.name != "inventory.csv"
    )
    failures.extend(f"unexpected CSV: {name}" for name in unexpected)

    if failures:
        print("OFFICIAL DATA CHECK FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"OFFICIAL DATA CHECK PASSED: {len(expected_hashes)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
