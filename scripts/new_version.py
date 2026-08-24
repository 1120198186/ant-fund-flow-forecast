#!/usr/bin/env python3
"""Create a standardized, self-contained major-version folder."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


VERSION_RE = re.compile(r"^v\d{3}_[a-z0-9][a-z0-9_]*$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Version name such as v002_calendar_features")
    parser.add_argument("--title", required=True, help="Human-readable purpose")
    parser.add_argument("--parent", default="none", help="Parent version, if any")
    args = parser.parse_args()

    if not VERSION_RE.fullmatch(args.version):
        parser.error("version must match vNNN_lower_snake_case")

    project_root = Path(__file__).resolve().parents[1]
    version_root = project_root / "versions" / args.version
    if version_root.exists():
        parser.error(f"refusing to overwrite existing version: {version_root}")

    folders = [
        "src",
        "configs",
        "notebooks",
        "artifacts/models",
        "artifacts/metrics",
        "artifacts/predictions",
        "artifacts/submissions",
    ]
    for folder in folders:
        (version_root / folder).mkdir(parents=True, exist_ok=False)

    readme = f"""# {args.version}

Status: scaffolded

Purpose: {args.title}

Parent: {args.parent}

Keep all code, configuration, notebooks, models, metrics, predictions, and submissions for this major direction inside this folder.
"""
    (version_root / "README.md").write_text(readme, encoding="utf-8")
    for relative in [
        "src/.gitkeep",
        "configs/.gitkeep",
        "notebooks/.gitkeep",
        "artifacts/models/.gitkeep",
        "artifacts/predictions/.gitkeep",
        "artifacts/submissions/.gitkeep",
    ]:
        (version_root / relative).touch()

    print(version_root)
    print("Next: add this version and its lineage to versions/REGISTRY.md")


if __name__ == "__main__":
    main()
