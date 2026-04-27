# Non-invasive glucose estimation (ML prototype)

Engineering prototype for estimating blood glucose (mg/dL) from indirect physiological features—PPG-derived signals, heart rate, and demographics—using supervised **regression**. This repository is for **research and demonstration**, not medical diagnosis or a certified medical device.

## Repository contents

| File / folder | Purpose |
|----------------|---------|
| `clean-dataset.csv` | Raw dataset as supplied before project-specific cleaning |
| `clean-dataset-relevant.csv` | Cleaned, de-duplicated, conflict-resolved table with engineered features for modeling |
| `clean-dataset-relevant-report.json` | Machine-readable summary of the build (row counts, columns, steps) |
| `build_relevant_dataset.py` | Script that rebuilds `clean-dataset-relevant.csv` from `clean-dataset.csv` |
| `01_data_domain_analysis.ipynb` | Data and domain analysis: leakage checks, pipeline walkthrough, plots |
| `_gen_notebook.py` | Optional helper that regenerates `01_data_domain_analysis.ipynb` from the same folder |
| `PROGRESS_UPDATE_REPORT.md` | Client-facing progress narrative |

## Environment

Use Python 3.10+ (3.11+ recommended). Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbformat
```

`nbformat` is only needed if you run `_gen_notebook.py`.

## Rebuild the modeling dataset

From this directory:

```bash
python build_relevant_dataset.py
```

This overwrites `clean-dataset-relevant.csv` and `clean-dataset-relevant-report.json` according to the rules in the script.

## Run the analysis notebook

```bash
jupyter notebook 01_data_domain_analysis.ipynb
```

Or open the notebook in VS Code / Cursor and run all cells. Update `BASE` in the first setup cell if you move the project to another path.

## Modeling notes

- **Task:** regression on `Glucose_level` (continuous mg/dL).
- **Splits:** use **grouped** train/test or cross-validation by `Patient_Id` — do not rely on random row splits alone for reported performance.
- **Limitations:** small participant count and limited glucose range in the current sample; see the notebook and `PROGRESS_UPDATE_REPORT.md`.

## Disclaimer

All outputs are **prototype / educational** quality. Do not use them for clinical decisions without proper validation, regulatory compliance, and supervision by qualified health professionals.
