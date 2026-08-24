# Workspace setup ledger

Goal: start Tianchi competition `231573` with strict partitions for official data, self-built tests, and major model versions.

Acceptance basis:

- supplied data is preserved in one immutable project zone and can be integrity-checked;
- self-built test definitions, locked labels, and reports have a separate zone;
- every major model direction has one self-contained `vNNN_name` folder;
- the first version is scaffolded, without claiming a trained result;
- official competition and submission facts are recorded with evidence boundaries;
- the integrated workspace passes local checks and independent delivery review.

State:

- `competition_contract` worker: completed official-page and API verification; read-only.
- Root Main: project root and begin checkpoint created; workspace structure and competition contract implemented.

Material decisions:

- Project root: `/Users/zyy/Documents/tianchi_231573_purchase_redemption`, an independent Git repository outside the dirty pre-existing `movir-command` repository.
- Large CSV files and generated labels/artifacts are local and ignored by Git; tracked manifests preserve identity and reproducibility.
- Initial local holdout is time-based and exactly 30 days: train through 2014-08-01, validate on 2014-08-02 through 2014-08-31.

Validation and risks:

- Official contract verified from the live Tianchi pages and APIs. The current API shows the season ending 2027-05-06, but actual submit availability remains unverified until login.
- Official-data integrity passed for all five files; source and project copies are byte-identical.
- The 30-day holdout was generated with 30 data rows. A second run correctly refused to overwrite the locked labels.
- The submission validator rejected the supplied three-row fragment and accepted a generated 30-row no-header contract fixture via standard input.
- Python syntax compilation passed.
- First independent delivery review passed with no material findings.
- A final-surface reviewer then found that the original location still inherited the parent `movir-command` Git boundary. Root Main repaired this by moving the complete workspace to the independent project root above; final re-review is pending.
- Post-migration checks passed: the new `.git/config` has no inherited remote, the old project directory is empty, all five source copies remain byte-identical, Python compilation and official-data verification pass, the holdout remains 30 unique dates, overwrite protection holds, and submission-contract positive/negative checks pass.
- Post-migration review found two rate-table row counts were each understated by one because newline counts had been used for files without a trailing newline. The inventory was corrected without touching raw data; `verify_official_data.py` now parses logical CSV rows, validates frozen headers, row counts, sizes, and manifest/inventory hashes. The expanded integrity check passes for all five files; final re-review is pending.
- The supplied `comp_predict_table.csv` has only three rows and no header, while the verified contract requires 30 rows. It is preserved as raw evidence but excluded as a final template.

Current state:

- Partition setup, data, and scripts are complete. Independent-repository repair is complete; final delivery re-review and finish checkpoints are pending.
- No modeling or online submission was performed in this turn.
