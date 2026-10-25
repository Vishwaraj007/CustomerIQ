# CustomerIQ — Dynamic CSV Analytics

CustomerIQ is a single-page Streamlit dashboard that profiles CSV files and chooses analysis based on the columns actually present. It preserves the bundled retail demo while supporting general numerical/categorical data, customer profiles, sales tables, and date-based datasets.

## What it does

- Upload a CSV (up to 50 MB) or explore `data/sample_retail_transactions.csv`.
- Detect column roles with optional Gemini schema assistance; local heuristics keep the app useful without an API key.
- Review and correct column roles without renaming the original fields.
- Summarize records, columns, missing data, duplicates, constant fields, and potential numerical outliers.
- Choose distributions, category comparisons, numerical correlations, and time trends only when suitable columns exist.
- Show sales and customer/RFM analysis only when required fields are available.
- Generate evidence-based local recommendations from calculated dataset statistics, with optional Gemini interpretation of aggregate findings.
- Optionally run K-Means clustering on selected numerical features.
- Download analyzed CSV data, role mappings, clustered data, and a dataset-aware PDF report.

All calculations, charts, clustering, and PDF generation run locally. Gemini is optional and receives schema metadata, not raw customer rows. No data is silently deleted.

## Requirements

Python 3.10–3.12 is recommended.

## Install and launch

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Optional Gemini setup

1. Get a Gemini Developer API key from [Google AI Studio](https://aistudio.google.com/).
2. Copy `.env.example` to `.env`.
3. Replace `your_gemini_api_key_here` with your key. Keep `.env` private; it is ignored by Git.
4. Restart Streamlit after changing the file.

Example `.env`:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

The default model is configurable using `GEMINI_MODEL`. Model availability, free-tier quotas, and data-use terms can change; review the [Gemini API documentation](https://ai.google.dev/gemini-api/docs) and [pricing](https://ai.google.dev/gemini-api/docs/pricing) before use. A valid API key is not required to run the dashboard.

Alternatively, set `GEMINI_API_KEY` in your environment or Streamlit secrets. Never commit a real key.

## Privacy and limitations

- Gemini schema detection receives column names, inferred data types, missing percentages, unique counts, and synthetic value-shape descriptors. Optional insight interpretation receives only calculated aggregate findings. Neither feature receives full CSV rows or raw cell values.
- If Gemini is unavailable, local schema detection and calculated insights remain available.
- Correlation and clustering show patterns, not causation or ground-truth classes.
- Outliers are flagged for review and are not automatically removed.
- The PDF report describes locally calculated findings and includes methodology/limitations.
