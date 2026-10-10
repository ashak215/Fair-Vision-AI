# FairVision AI

FairVision AI is a Streamlit dashboard for exploring gender representation, salary differences, and recorded promotion outcomes. It also includes a carefully labeled experimental machine-learning demonstration and downloadable PDF/CSV reports.

> **Data note:** `data/equalai_employee_data.csv` is synthetic demonstration data. It does not describe a real company or population and must not be presented as real-world evidence. Results are descriptive; differences do not by themselves prove discrimination.

## Features

- Five pages: Home, Dashboard, Gender Analytics, AI Insights, and Reports.
- Responsive landing page with persistent top navigation and local SVG equality artwork.
- Department and experience filters, interactive Plotly charts, and data-quality reporting.
- CSV upload with required-column validation, duplicate removal, safe numeric parsing, and conservative label normalization.
- Optional promotion prediction model only when both target classes have sufficient records. Gender is excluded from model features.
- PDF summary and ML evaluation reports, plus filtered/detailed CSV downloads.
- Synthetic provenance labels, report limitations, and human-review guidance.

## Requirements

- Windows 10 or 11
- Python 3.10 or later
- Internet access for the initial package installation

## Windows setup

Open PowerShell in the `FairVision AI` project folder.

### 1. Create a virtual environment

```powershell
py -m venv .venv
```

### 2. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current window, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the app

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your browser. Stop the app with `Ctrl+C`.

## Dataset

The app loads `data/equalai_employee_data.csv` relative to the project, so it does not depend on the PowerShell working directory. Expected columns:

`gender`, `age`, `department`, `experience_years`, `salary`, `promotion_status`

Use **Upload an alternative CSV** in the sidebar to inspect another dataset. Uploaded data is labeled as unverified. If the bundled CSV is missing, the app provides an optional generated synthetic demo dataset.

Promotion target normalization recognizes common exact yes/no values (`Yes`, `Y`, `True`, `1`, `Promoted`; and `No`, `N`, `False`, `0`, `Not promoted`). Other labels are treated as unknown rather than reinterpreted. Numeric values outside reasonable bounds or which cannot be parsed are set to missing; they are not used in calculations requiring those values.

## Model behavior

The optional model predicts only recorded `promotion_status` values and uses an approximately 75/25 stratified train/test split where the sample supports it. At least 30 usable outcome records and 8 records in each class are required. Numerical imputation, categorical imputation, and one-hot encoding are fitted as part of the training pipeline. `gender` and the target are excluded from model inputs. Metrics and feature importance are computed from the trained model; if evaluation is unsuitable, the app explains why and shows descriptive outcomes instead. In Gender Analytics, promotion-rate comparisons are displayed only when both Female and Male groups have at least five recorded Yes/No outcomes.

This demonstration is not a hiring or promotion recommendation and must not be used to make decisions about individuals. Holdout scores on small or synthetic data do not establish model validity, fairness, or generalization.

## Testing checklist

- Run the automated core tests with `python -m unittest discover -s tests -v`.
- [ ] Activate the virtual environment and install `requirements.txt`.
- [ ] Start the app with the command above and navigate through all five pages.
- [ ] On Dashboard, change department and experience filters; confirm metrics and charts recalculate.
- [ ] Confirm the bundled file is identified as synthetic and data-quality counts are displayed.
- [ ] On Gender Analytics, inspect salary, promotion, department representation, and experience summaries; download each CSV.
- [ ] On AI Insights, verify the metrics and confusion matrix are produced from a holdout split; try the experimental interface.
- [ ] Generate a report pack and download each available PDF/CSV; inspect the provenance, filters, generation date, and limitations.
- [ ] Upload a valid alternative CSV, then try a CSV missing a required column and confirm a clear validation message appears.
- [ ] For a dataset too small or without valid Yes/No outcomes, confirm no model scores are fabricated and descriptive insights remain available.
