#!/usr/bin/env python3
"""Independently validate the delivered snapshot, canonical layer, and audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import sys

import nbformat
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SNAPSHOT = "d002_official_csv_snapshot"
LAYER = "d003_validated_data_layer_v2"
AUDIT = "d004_data_quality_audit_v2"
OFFICIAL_FILES = [
    "comp_predict_table.csv",
    "mfd_bank_shibor.csv",
    "mfd_day_share_interest.csv",
    "user_balance_table.csv",
    "user_profile_table.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw" / "official"
    snapshot = root / "data" / "derived" / SNAPSHOT
    layer = root / "data" / "derived" / LAYER
    audit = root / "data" / "derived" / AUDIT
    failures: list[str] = []

    for name in OFFICIAL_FILES:
        source = raw / name
        copied = snapshot / "csv" / name
        require(copied.is_file(), f"snapshot missing {name}", failures)
        if copied.is_file():
            require(
                sha256(source) == sha256(copied),
                f"snapshot hash differs from official source: {name}",
                failures,
            )
            require(
                not bool(copied.stat().st_mode & stat.S_IWRITE),
                f"snapshot copy is writable: {name}",
                failures,
            )

    balance_path = layer / "user_balance_daily.parquet"
    balance_schema = pq.read_schema(balance_path)
    for parquet_path in layer.glob("*.parquet"):
        require(
            not bool(parquet_path.stat().st_mode & stat.S_IWRITE),
            f"canonical parquet is writable: {parquet_path.name}",
            failures,
        )
    require(
        balance_schema.field("report_date").type == pa.date32(),
        "report_date is not Parquet date32",
        failures,
    )
    require(
        not any("extreme" in field.name for field in balance_schema),
        "canonical balance layer contains full-period extreme flags",
        failures,
    )
    require(
        pq.ParquetFile(balance_path).metadata.num_rows == 2_840_421,
        "canonical balance row count differs from official source",
        failures,
    )

    source_totals = pd.read_csv(
        raw / "user_balance_table.csv",
        usecols=["total_purchase_amt", "total_redeem_amt"],
    ).sum()
    layer_totals = pd.read_parquet(
        balance_path,
        columns=["total_purchase_amt", "total_redeem_amt"],
    ).sum()
    require(
        all(int(source_totals[column]) == int(layer_totals[column]) for column in source_totals.index),
        "canonical balance totals do not reconcile to official CSV",
        failures,
    )

    observed = pd.read_csv(
        raw / "mfd_bank_shibor.csv", parse_dates=["mfd_date"], date_format="%Y%m%d"
    ).sort_values("mfd_date")
    daily = pd.read_parquet(layer / "shibor_daily.parquet")
    daily["mfd_date"] = pd.to_datetime(daily["mfd_date"])
    rate_columns = [column for column in observed.columns if column != "mfd_date"]
    expected = pd.DataFrame(
        {"mfd_date": pd.date_range(daily["mfd_date"].min(), daily["mfd_date"].max())}
    ).merge(observed, on="mfd_date", how="left")
    expected[rate_columns] = expected[rate_columns].ffill()
    require(
        expected[rate_columns].reset_index(drop=True).equals(
            daily[rate_columns].astype(float).reset_index(drop=True)
        ),
        "derived SHIBOR differs from historical-only forward fill",
        failures,
    )

    summary = pd.read_csv(audit / "audit_summary.csv")
    require(
        not summary["status"].eq("FAIL").any(),
        "formal audit contains a failed hard check",
        failures,
    )
    require(
        int(summary.loc[summary["check_id"] == "additive_relationships", "metric_value"].iloc[0])
        == 1,
        "expected single official balance identity exception not recorded",
        failures,
    )

    notebook = nbformat.read(audit / "data_quality_audit.ipynb", as_version=4)
    error_outputs = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    require(not error_outputs, "executed audit notebook contains errors", failures)

    artifact = json.loads((audit / "report_artifact.json").read_text(encoding="utf-8"))
    require(
        artifact["snapshot"]["status"] == "ready",
        "report artifact is not ready",
        failures,
    )
    require(
        len(artifact["manifest"]["blocks"]) == 13,
        "report artifact does not contain 13 reviewed blocks",
        failures,
    )
    require(
        all(row.get("query", {}).get("sql") for row in artifact["sources"]),
        "report artifact has a source without a reproducible query",
        failures,
    )
    html = audit / "data_quality_report.html"
    require(html.is_file() and html.stat().st_size > 100_000, "portable HTML missing", failures)
    if html.is_file():
        html_text = html.read_text(encoding="utf-8")
        require("官方数据质量审计" in html_text, "HTML title missing", failures)
        require(
            html_text.count("data-artifact-block-id=") == 13,
            "HTML block count differs from artifact",
            failures,
        )

    manifest = json.loads((audit / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = audit / row["path"]
        require(path.is_file(), f"audit manifest missing file: {row['path']}", failures)
        if path.is_file():
            require(
                sha256(path) == row["sha256"],
                f"audit manifest hash mismatch: {row['path']}",
                failures,
            )

    if failures:
        print("DATA FOUNDATION VALIDATION FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("DATA FOUNDATION VALIDATION PASSED")
    print("- official snapshot: 5/5 byte-identical files")
    print("- canonical layer: rows, totals, date32, and historical SHIBOR verified")
    print("- formal audit: no hard failures; one retained balance exception")
    print("- notebook/report: executed notebook, 13-block HTML, and manifest hashes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
