# CustomerIQ Project Overview

CustomerIQ is a single-page CSV analytics dashboard built with Streamlit, Pandas, Plotly, NumPy, scikit-learn, and ReportLab.

## Main workflow
1. Upload a CSV or use the bundled retail demo.
2. Inspect the dataset overview and quality checks.
3. Review/correct semantic column roles.
4. Explore distributions, category comparisons, correlations, and time trends when supported by the data.
5. Review retail/customer analytics only when the required fields exist.
6. Read locally calculated insights and recommended next steps.
7. Optionally run K-Means and download CSV/PDF outputs.

Gemini Developer API support is optional and used for semantic schema detection. Without an API key, local heuristic detection and the core dashboard continue to work. Raw CSV records are not sent to Gemini.
