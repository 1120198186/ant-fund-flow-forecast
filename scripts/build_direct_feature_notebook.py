#!/usr/bin/env python3
"""Create and execute the direct-feature analysis companion notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output_root = root / "data" / "derived" / "d005_direct_feature_analysis_v1"
    output = output_root / "direct_feature_analysis.ipynb"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite notebook: {output}")
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Direct feature and target associations\n\n"
            "Reviewed evidence at three grains: user demographics, daily rates, "
            "and daily/user transaction components."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display\n"
            "root = Path.cwd() / 'data' / 'derived' / 'd005_direct_feature_analysis_v1'\n"
            "demo = pd.read_csv(root / 'demographic_associations.csv')\n"
            "rates = pd.read_csv(root / 'rate_associations.csv')\n"
            "transactions = pd.read_csv(root / 'transaction_component_associations.csv')\n"
            "catalog = pd.read_csv(root / 'feature_combination_catalog.csv')"
        ),
        nbformat.v4.new_code_cell(
            "primary = ['total_purchase_amt','total_redeem_amt','direct_purchase_amt','purchase_bank_amt','transfer_amt']\n"
            "display(demo[(demo.target.isin(primary)) & (demo.metric == 'log1p_amount_per_exposure_day')]"
            ".sort_values('eta_squared', ascending=False))"
        ),
        nbformat.v4.new_code_cell(
            "cols = ['feature','target','spearman','diff1_corr_log_target','lagged_diff1_corr_log_target']\n"
            "display(rates[rates.target.isin(primary)].sort_values('short_run_abs_diff1', ascending=False)[cols].head(25))"
        ),
        nbformat.v4.new_code_cell(
            "display(transactions.head(25)[['left','right','relationship_type','daily_spearman','user_spearman','prediction_use']])"
        ),
        nbformat.v4.new_code_cell(
            "display(catalog.groupby(['priority','family']).size().rename('features').reset_index())"
        ),
    ]
    nbformat.write(notebook, output)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    client.execute()
    nbformat.write(notebook, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
