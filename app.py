from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib, io, json, os, re, warnings

# Load .env first so all os.getenv() calls below pick up the configured values.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

st.set_page_config(page_title="CustomerIQ | CSV Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "sample_retail_transactions.csv"
MAX_UPLOAD_MB = 50
ROLE_OPTIONS = [
    "unknown", "customer_id", "transaction_id", "order_id", "transaction_date",
    "signup_date", "date_time", "quantity", "unit_price", "revenue", "sales_amount",
    "product_name", "category", "brand", "region", "country", "age", "income",
    "gender", "location", "numeric_measure", "categorical", "text", "target"
]


# ── CSV helpers ──────────────────────────────────────────────────────────────

def read_csv_bytes(data: bytes) -> pd.DataFrame:
    if not data:
        raise ValueError("The uploaded CSV is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding, low_memory=False)
            if len(df.columns) == 0:
                raise ValueError("No columns were found in this CSV.")
            return make_unique_columns(df)
        except Exception:
            pass
    raise ValueError("Could not parse this CSV. Check its delimiter, quoting, and file contents.")


def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seen, cols = {}, []
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"column_{i+1}"
        count = seen.get(name, 0); seen[name] = count + 1
        cols.append(name if count == 0 else f"{name}.{count+1}")
    df.columns = cols
    return df


# ── Schema detection ─────────────────────────────────────────────────────────

def schema_metadata(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        sample = s.dropna().head(3).tolist()
        safe_samples = []
        for x in sample:
            if pd.api.types.is_numeric_dtype(s):
                safe_samples.append("numeric_value")
            elif pd.api.types.is_datetime64_any_dtype(s):
                safe_samples.append("date_value")
            else:
                safe_samples.append(f"text_length_{min(len(str(x)), 50)}")
        cols.append({
            "name": str(c), "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique_count": int(s.nunique(dropna=True)),
            "safe_examples": list(dict.fromkeys(safe_samples))[:3]
        })
    return {"rows": int(len(df)), "column_count": len(df.columns), "columns": cols}


def heuristic_schema(df: pd.DataFrame) -> dict:
    mapping = {}
    patterns = [
        ("customer_id",    r"customer.?id|client.?no|client.?id|user.?id|member.?id|account.?id"),
        ("transaction_id", r"transaction.?id|trans.?id"),
        ("order_id",       r"invoice.?no|order.?id|order.?no|receipt.?id"),
        ("transaction_date", r"invoice.?date|order.?date|order.?dt|transaction.?date|purchase.?date|sales.?date"),
        ("signup_date",    r"signup.?date|registration.?date|joined.?date"),
        ("date_time",      r"date|time|timestamp|created.?at"),
        ("quantity",       r"quantity|qty|units.?sold|units|count"),
        ("unit_price",     r"unit.?price|price.?per.?unit|item.?price|price"),
        ("revenue",        r"revenue|sales.?amount|total.?sales|turnover|amount"),
        ("sales_amount",   r"^sales$|total.?amount|monetary|spend"),
        ("product_name",   r"product|item.?name|description|sku|stock.?code"),
        ("category",       r"category|department|class|segment"),
        ("brand",          r"brand|manufacturer"),
        ("region",         r"region|state|province|territory"),
        ("country",        r"country|nation"),
        ("age",            r"^age$|customer.?age"),
        ("income",         r"income|salary|annual.?earnings"),
        ("gender",         r"gender|sex"),
        ("location",       r"location|city|address|postal|zip"),
        ("target",         r"target|label|outcome|class|churn|default|diagnosis|^y$"),
    ]
    for c in df.columns:
        norm = re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
        role = None
        for candidate, pat in patterns:
            if re.search(pat, norm):
                role = candidate; break
        s = df[c]
        if role is None and ("date" in norm or "time" in norm): role = "date_time"
        if role is None and pd.api.types.is_numeric_dtype(s): role = "numeric_measure"
        if role is None and (s.nunique(dropna=True) <= max(30, int(len(s) * .15))): role = "categorical"
        if role is None: role = "text"
        mapping[c] = role
    roles = list(mapping.values())
    dtype_family = (
        "retail_transactions" if "customer_id" in roles and "transaction_date" in roles
            and ("revenue" in roles or "sales_amount" in roles or ("quantity" in roles and "unit_price" in roles))
        else "customer_profiles" if "transaction_date" not in roles and any(r in roles for r in ("age","income","gender"))
        else "sales" if any(r in roles for r in ("revenue","sales_amount"))
        else "general_structured"
    )
    return {"dataset_type": dtype_family, "columns": mapping, "confidence": 0.55,
            "missing_roles": [], "warnings": ["Local heuristic detection used."]}


_PLACEHOLDER_KEY = "your_groq_api_key_here"

def get_api_key():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        try: key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception: pass
    if key == _PLACEHOLDER_KEY:
        return ""
    return key

def _groq_chat(api_key: str, model: str, system_prompt: str, user_content: str, timeout: int = 30) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

@st.cache_data(show_spinner=False, ttl=86400)
def _groq_schema_cached(fingerprint: str, metadata_json: str, model: str, api_key: str) -> dict:
    prompt = (
        "Classify the semantic role of each CSV column. "
        "Return ONLY valid JSON with keys dataset_type, columns, confidence, missing_roles, warnings. "
        "dataset_type must be one of retail_transactions, customer_profiles, sales, general_structured. "
        "columns must map every exact provided column name to one role from: "
        "customer_id, transaction_id, order_id, transaction_date, signup_date, date_time, quantity, "
        "unit_price, revenue, sales_amount, product_name, category, brand, region, country, age, income, "
        "gender, location, numeric_measure, categorical, text, target, unknown. "
        "Use only exact supplied column names. Do not infer facts from rows; no analysis. "
        "Values in safe_examples are synthetic shape descriptors, not real data. "
        "Keep confidence between 0 and 1. Do not add markdown."
    )
    text = _groq_chat(api_key, model, prompt, "SCHEMA METADATA:\n" + metadata_json, timeout=20)
    return json.loads(text)


def validate_schema(candidate, df):
    if not isinstance(candidate, dict): raise ValueError("Schema response is not an object")
    dataset_type = candidate.get("dataset_type")
    if dataset_type not in {"retail_transactions","customer_profiles","sales","general_structured"}:
        raise ValueError("Invalid dataset type")
    cols = candidate.get("columns")
    if not isinstance(cols, dict) or set(cols.keys()) != set(map(str, df.columns)):
        raise ValueError("Column mapping does not exactly match uploaded columns")
    if any(role not in ROLE_OPTIONS for role in cols.values()):
        raise ValueError("Unknown semantic role returned")
    confidence = float(candidate.get("confidence", 0))
    if not 0 <= confidence <= 1: raise ValueError("Invalid confidence")
    if confidence < 0.35: raise ValueError("Schema confidence too low")
    roles = set(cols.values())
    if dataset_type == "retail_transactions" and not (
        "customer_id" in roles and "transaction_date" in roles and
        ("revenue" in roles or "sales_amount" in roles or ("quantity" in roles and "unit_price" in roles))
    ):
        raise ValueError("Retail classification incompatible with mapped columns")
    candidate["confidence"] = confidence
    candidate["missing_roles"]  = candidate.get("missing_roles", [])  if isinstance(candidate.get("missing_roles", []),  list) else []
    candidate["warnings"]       = candidate.get("warnings", [])       if isinstance(candidate.get("warnings", []),       list) else []
    if not all(isinstance(x, str) for x in candidate["missing_roles"] + candidate["warnings"]):
        raise ValueError("Invalid schema metadata fields")
    return candidate


def detect_schema(df):
    metadata    = schema_metadata(df)
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    model       = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip() or "qwen/qwen3.8-27b"
    key         = get_api_key()
    if key:
        try:
            result = validate_schema(_groq_schema_cached(fingerprint, json.dumps(metadata, sort_keys=True), model, key), df)
            return result, f"Groq ({model})", None
        except Exception:
            fallback_reason = "Groq request failed or returned an invalid schema; local detection is active."
    else:
        fallback_reason = "GROQ_API_KEY is not configured; local detection is active."
    return validate_schema(heuristic_schema(df), df), "Local heuristic fallback", fallback_reason


# ── Role helpers ──────────────────────────────────────────────────────────────

def role_columns(mapping, role): return [c for c, r in mapping.items() if r == role]
def first_role(mapping, *roles):
    for role in roles:
        found = role_columns(mapping, role)
        if found: return found[0]
    return None


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_analysis(df, mapping):
    data = df.copy()
    date_col = first_role(mapping, "transaction_date", "signup_date", "date_time")
    numeric_cols = [c for c in data.columns if mapping.get(c) in {
        "quantity","unit_price","revenue","sales_amount","numeric_measure","age","income"
    }]
    for c in numeric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    parsed_date = None
    if date_col:
        parsed = pd.to_datetime(data[date_col], errors="coerce", format="mixed")
        if parsed.notna().sum() >= max(2, int(len(data) * .25)):
            data[date_col] = parsed; parsed_date = date_col
    qty     = first_role(mapping, "quantity")
    price   = first_role(mapping, "unit_price")
    revenue = first_role(mapping, "revenue", "sales_amount")
    if revenue is None and qty and price:
        data["CalculatedRevenue"] = data[qty] * data[price]
        revenue = "CalculatedRevenue"
    return data, parsed_date, revenue


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_frame(df, cols, k):
    if len(cols) < 2: raise ValueError("Select at least two numeric features for clustering.")
    x = df[cols].replace([np.inf, -np.inf], np.nan)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0)
    keep = x.nunique(dropna=False) > 1
    x = x.loc[:, keep]
    if x.shape[1] < 2: raise ValueError("At least two non-constant numeric features are required.")
    if len(x) < 3:     raise ValueError("At least three rows are required for clustering.")
    k = min(max(2, int(k)), len(x) - 1)
    source_index = x.index
    if x.shape[0] > 50000:
        x = x.sample(50000, random_state=42)
        source_index = x.index
    scaled = StandardScaler().fit_transform(x)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    score  = (silhouette_score(scaled, labels, sample_size=min(len(labels), 5000), random_state=42)
              if len(np.unique(labels)) > 1 and len(labels) > k else np.nan)
    result = df.loc[source_index].copy()
    result["Cluster"] = labels
    return result, score, list(x.columns)


# ── Per-column statistics (universal) ────────────────────────────────────────

def column_stats(df: pd.DataFrame) -> list[dict]:
    """Compute descriptive stats for every column."""
    stats = []
    for c in df.columns:
        s = df[c]
        row: dict = {
            "Column": c,
            "Type":   str(s.dtype),
            "Non-null": int(s.notna().sum()),
            "Missing (%)": round(float(s.isna().mean() * 100), 2),
            "Unique": int(s.nunique(dropna=True)),
        }
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() >= 4:
            row.update({
                "Min":    round(float(num.min()), 4),
                "Max":    round(float(num.max()), 4),
                "Mean":   round(float(num.mean()), 4),
                "Median": round(float(num.median()), 4),
                "Std":    round(float(num.std()), 4),
            })
            q25, q75 = num.quantile([.25, .75])
            iqr = q75 - q25
            if iqr > 0:
                outliers = int(((num < q25 - 1.5 * iqr) | (num > q75 + 1.5 * iqr)).sum())
                row["Outliers (IQR)"] = outliers
                row["Outlier (%)"] = round(outliers / max(1, num.notna().sum()) * 100, 2)
        else:
            row.update({"Min": "—", "Max": "—", "Mean": "—", "Median": "—", "Std": "—"})
        stats.append(row)
    return stats


def correlation_pairs(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Return all unique column pairs with Pearson correlation, sorted by |r|."""
    if len(num_cols) < 2:
        return pd.DataFrame()
    corr = df[num_cols].corr(numeric_only=True)
    rows = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if pd.notna(val):
                rows.append({
                    "Column A": corr.columns[i],
                    "Column B": corr.columns[j],
                    "Pearson r": round(float(val), 4),
                    "|r|": round(abs(float(val)), 4),
                    "Strength": (
                        "Strong" if abs(val) >= .7
                        else "Moderate" if abs(val) >= .4
                        else "Weak"
                    ),
                    "Direction": "Positive" if val > 0 else "Negative",
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("|r|", ascending=False).drop(columns=["|r|"])


def outlier_summary(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    rows = []
    for c in num_cols[:60]:
        s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 4 or s.nunique() <= 1:
            continue
        q25, q75 = s.quantile([.25, .75])
        iqr = q75 - q25
        if iqr <= 0:
            continue
        lo, hi    = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        mask      = (s < lo) | (s > hi)
        count     = int(mask.sum())
        if count:
            rows.append({
                "Column":          c,
                "Outlier count":   count,
                "Outlier (%)":     round(count / len(s) * 100, 2),
                "Lower fence":     round(float(lo), 4),
                "Upper fence":     round(float(hi), 4),
                "Min outlier val": round(float(s[mask].min()), 4),
                "Max outlier val": round(float(s[mask].max()), 4),
            })
    return sorted(rows, key=lambda x: x["Outlier count"], reverse=True)


# ── Groq AI insights ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _groq_insights_cached(payload_json: str, model: str, api_key: str) -> list:
    system_prompt = (
        "You are a cautious data analyst. Interpret only the supplied aggregate findings; "
        "do not invent statistics or causes. "
        "Return ONLY JSON: {\"insights\":[{\"finding\":\"...\",\"evidence\":\"...\","
        "\"why_it_matters\":\"...\",\"action\":\"...\",\"priority\":\"Low|Medium|High\"}]}. "
        "Provide at most 5 concise items covering column distributions, correlations, outliers, "
        "and any time trends present. Every evidence statement must be directly supported by "
        "the supplied findings. Treat correlation as association, not causation. "
        "Do not request or infer raw records."
    )
    text   = _groq_chat(api_key, model, system_prompt, "AGGREGATE FINDINGS:\n" + payload_json, timeout=40)
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("insights"), list):
        raise ValueError("Invalid insight response")
    safe = []
    for item in parsed["insights"][:5]:
        if not isinstance(item, dict): continue
        fields = {k: item.get(k, "") for k in ("finding","evidence","why_it_matters","action","priority")}
        if not all(isinstance(v, str) for v in fields.values()): continue
        if fields["priority"] not in {"Low","Medium","High"}: fields["priority"] = "Low"
        if not fields["finding"] or not fields["evidence"] or not fields["action"]: continue
        safe.append(fields)
    return safe


# ── PDF report ────────────────────────────────────────────────────────────────

def _tbl(data_rows, col_widths, header_color="#174b65"):
    t = Table(data_rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor(header_color)),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
        ("GRID",          (0, 0), (-1, -1), .3, colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]),
        ("PADDING",       (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def create_pdf_report(
    df, mapping, dataset_type, source_name,
    quality, summary_lines,
    col_stats: list[dict],
    corr_df: pd.DataFrame,
    outliers: list[dict],
    ai_insights: list[dict] | None = None,
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("CIQTitle",   parent=styles["Title"],   textColor=colors.HexColor("#174b65"), fontSize=22, leading=26, spaceAfter=8))
    styles.add(ParagraphStyle("CIQSection", parent=styles["Heading2"],textColor=colors.HexColor("#174b65"), spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle("CIQSub",     parent=styles["Heading3"],textColor=colors.HexColor("#187f83"), spaceBefore=6, spaceAfter=3))
    B  = styles["BodyText"]
    IT = styles["Italic"]

    story = []

    # ── Cover ──
    story += [
        Paragraph("CustomerIQ", styles["CIQTitle"]),
        Paragraph("Dataset Analytics Report", styles["Heading2"]),
        Paragraph(f"Source: <b>{source_name}</b> · Generated: {datetime.now():%d %b %Y %H:%M}", B),
        Spacer(1, 10),
    ]

    # ── 1. Dataset profile ──
    story.append(Paragraph("1. Dataset Profile", styles["CIQSection"]))
    profile_rows = [
        ["Property", "Value"],
        ["Dataset type",                  dataset_type],
        ["Rows",                          f"{len(df):,}"],
        ["Columns",                       str(len(df.columns))],
        ["Duplicate rows",                f"{int(df.duplicated().sum()):,}"],
        ["Columns with missing values",   str(int(df.isna().any().sum()))],
        ["Total missing cells",           f"{int(df.isna().sum().sum()):,}"],
    ]
    story.append(_tbl(profile_rows, [65*mm, 105*mm]))
    story.append(Spacer(1, 6))

    # ── 2. Data quality ──
    story.append(Paragraph("2. Data Quality", styles["CIQSection"]))
    q_rows = [["Check", "Result"]] + [[str(r[0]), str(r[1])] for r in quality]
    story.append(_tbl(q_rows, [95*mm, 75*mm], header_color="#187f83"))
    story.append(Spacer(1, 6))

    # ── 3. Key findings ──
    story.append(Paragraph("3. Key Findings", styles["CIQSection"]))
    for line in summary_lines[:30]:
        story.append(Paragraph("• " + str(line), B))
        story.append(Spacer(1, 2))
    story.append(Spacer(1, 6))

    # ── 4. Per-column statistics ──
    story.append(PageBreak())
    story.append(Paragraph("4. Per-Column Statistics", styles["CIQSection"]))
    story.append(Paragraph(
        "Descriptive statistics for every column in the dataset. "
        "Outlier counts use the IQR rule (1.5×IQR beyond Q1/Q3).", IT
    ))
    story.append(Spacer(1, 4))

    stat_keys = ["Column","Type","Non-null","Missing (%)","Unique","Min","Max","Mean","Median","Std","Outliers (IQR)","Outlier (%)"]
    # Only include keys that actually appear
    present_keys = [k for k in stat_keys if any(k in row for row in col_stats)]
    header = [present_keys]
    col_w_map = {
        "Column": 35*mm, "Type": 18*mm, "Non-null": 16*mm, "Missing (%)": 16*mm,
        "Unique": 14*mm, "Min": 18*mm, "Max": 18*mm, "Mean": 18*mm,
        "Median": 18*mm, "Std": 18*mm, "Outliers (IQR)": 18*mm, "Outlier (%)": 14*mm,
    }
    col_widths_stat = [col_w_map.get(k, 18*mm) for k in present_keys]

    stat_data = header + [
        [str(row.get(k, "—"))[:30] for k in present_keys]
        for row in col_stats
    ]
    story.append(_tbl(stat_data, col_widths_stat))
    story.append(Spacer(1, 6))

    # ── 5. Correlation analysis ──
    story.append(PageBreak())
    story.append(Paragraph("5. Correlation Analysis", styles["CIQSection"]))
    if corr_df.empty:
        story.append(Paragraph("Fewer than two numeric columns — correlation not applicable.", B))
    else:
        story.append(Paragraph(
            "Pearson correlations between all numeric column pairs, sorted by absolute strength. "
            "Correlation describes linear association; it does not establish causation.", IT
        ))
        story.append(Spacer(1, 4))
        strong   = corr_df[corr_df["Strength"] == "Strong"]
        moderate = corr_df[corr_df["Strength"] == "Moderate"]
        if not strong.empty:
            story.append(Paragraph(f"Strong associations (|r| ≥ 0.70): {len(strong)}", styles["CIQSub"]))
        if not moderate.empty:
            story.append(Paragraph(f"Moderate associations (0.40 ≤ |r| < 0.70): {len(moderate)}", styles["CIQSub"]))

        corr_show = corr_df.head(40)
        c_header  = [list(corr_show.columns)]
        c_data    = c_header + [
            [str(v)[:25] for v in row] for row in corr_show.values
        ]
        cw = [45*mm, 45*mm, 22*mm, 22*mm, 22*mm]
        story.append(_tbl(c_data, cw))
    story.append(Spacer(1, 6))

    # ── 6. Outlier summary ──
    story.append(Paragraph("6. Outlier Summary (IQR Rule)", styles["CIQSection"]))
    if not outliers:
        story.append(Paragraph("No outliers detected across any numeric column.", B))
    else:
        story.append(Paragraph(
            "Potential outliers detected using the 1.5×IQR rule. "
            "These are candidates for investigation, not confirmed errors.", IT
        ))
        story.append(Spacer(1, 4))
        o_keys   = ["Column","Outlier count","Outlier (%)","Lower fence","Upper fence","Min outlier val","Max outlier val"]
        o_header = [o_keys]
        o_data   = o_header + [[str(row.get(k,"—")) for k in o_keys] for row in outliers]
        o_widths = [35*mm, 20*mm, 18*mm, 22*mm, 22*mm, 22*mm, 22*mm]
        story.append(_tbl(o_data, o_widths))
    story.append(Spacer(1, 6))

    # ── 7. Column roles ──
    story.append(PageBreak())
    story.append(Paragraph("7. Detected Column Roles", styles["CIQSection"]))
    role_rows = [["Column", "Assigned Role", "Data Type"]] + [
        [str(c)[:60], str(mapping.get(c, "unknown")), str(df[c].dtype)]
        for c in df.columns
    ]
    story.append(_tbl(role_rows, [80*mm, 50*mm, 40*mm]))
    story.append(Spacer(1, 6))

    # ── 8. AI insights (if available) ──
    if ai_insights:
        story.append(Paragraph("8. AI-Generated Insights (Groq)", styles["CIQSection"]))
        story.append(Paragraph(
            "Interpretations generated from aggregate statistics only — no raw data was sent to the AI.", IT
        ))
        story.append(Spacer(1, 4))
        for item in ai_insights:
            story.append(Paragraph(f"<b>{item.get('finding','')}</b> · Priority: {item.get('priority','')}", B))
            story.append(Paragraph(f"Evidence: {item.get('evidence','')}", B))
            story.append(Paragraph(f"Why it matters: {item.get('why_it_matters','')}", B))
            story.append(Paragraph(f"Suggested action: {item.get('action','')}", B))
            story.append(Spacer(1, 5))

    # ── Methodology ──
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Methodology: all metrics are calculated locally from the provided CSV. "
        "Semantic column roles may be heuristic or AI-assisted. "
        "Correlation and clustering indicate patterns, not causation. "
        "Outlier detection uses the 1.5×IQR rule as a screening heuristic. "
        "Analyses are omitted when required data is unavailable.", IT
    ))

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.block-container{padding-top:1.25rem;padding-bottom:3rem;max-width:1440px}
.hero{padding:1.35rem 1.55rem;border:1px solid rgba(100,116,139,.22);border-radius:16px;
  background:linear-gradient(115deg,#10233f,#174b65 58%,#187f83);color:white;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem;color:white;letter-spacing:-.03em}
.hero p{margin:.4rem 0 0;color:#d8eef2;font-size:1rem}
.section-label{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;
  color:#187f83;font-weight:700;margin-bottom:.15rem}
div[data-testid="stMetric"]{border:1px solid rgba(128,145,160,.25);padding:.8rem 1rem;
  border-radius:12px;background:rgba(128,145,160,.045)}
div[data-testid="stMetricLabel"]{font-size:.85rem}
hr{margin:1.25rem 0}
</style>
<div class="hero">
  <h1>CustomerIQ</h1>
  <p>Understand your data. Find meaningful patterns. Decide what to do next.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your dataset")
    uploaded  = st.file_uploader("Upload a CSV file", type=["csv"], help=f"Maximum file size: {MAX_UPLOAD_MB} MB")
    use_sample = st.checkbox("Use bundled retail demo", value=uploaded is None)
    st.divider()
    st.caption("Analysis settings")
    n_clusters = st.slider("Number of clusters", 2, 8, 4)
    st.caption("Groq AI is optional. Without a key, analysis uses local Python rules.")

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    if uploaded is not None:
        raw_df      = read_csv_bytes(uploaded.getvalue())
        source_name = uploaded.name
    elif use_sample:
        raw_df      = read_csv_bytes(DATA_PATH.read_bytes())
        source_name = "sample_retail_transactions.csv"
    else:
        st.info("Upload a CSV in the sidebar, or enable the bundled retail demo to explore CustomerIQ.")
        st.stop()
except Exception as exc:
    st.error(str(exc)); st.stop()

if raw_df.empty:
    st.error("This CSV contains headers but no data rows."); st.stop()
if len(raw_df.columns) > 500:
    st.error("Dataset has more than 500 columns. Reduce width before analysis."); st.stop()

# ── Schema detection ──────────────────────────────────────────────────────────
with st.spinner("Understanding your dataset…"):
    schema, detection_source, fallback_message = detect_schema(raw_df)
if fallback_message:
    st.caption(f"AI status: {fallback_message}")
else:
    st.caption(f"AI-assisted schema detection active: {detection_source}")
st.caption(f"**{source_name}** · {len(raw_df):,} rows · {len(raw_df.columns):,} columns · Dataset family: `{schema['dataset_type']}`")

with st.expander("Review and correct column meanings", expanded=False):
    st.write("Change any role that was detected incorrectly.")
    edited = {}
    for i, col in enumerate(raw_df.columns):
        current = schema["columns"].get(col, "unknown")
        edited[col] = st.selectbox(
            str(col), ROLE_OPTIONS,
            index=ROLE_OPTIONS.index(current) if current in ROLE_OPTIONS else 0,
            key=f"role_{i}_{hashlib.md5(str(col).encode()).hexdigest()[:8]}"
        )
    schema["columns"] = edited
    if st.button("Reclassify dataset from these roles", key="apply_roles"):
        st.rerun()

mapping = schema["columns"]
data, date_col, revenue_col = prepare_analysis(raw_df, mapping)

# Derive column lists
num_cols      = data.select_dtypes(include=np.number).columns.tolist()
date_cols_all = [c for c in data.columns if pd.api.types.is_datetime64_any_dtype(data[c])]
cat_cols      = [c for c in data.columns if c not in num_cols and c not in date_cols_all]

customer_col  = first_role(mapping, "customer_id")
order_col     = first_role(mapping, "transaction_id", "order_id")
product_col   = first_role(mapping, "product_name", "category", "brand")
location_col  = first_role(mapping, "country", "region", "location")

missing_count  = int(data.isna().sum().sum())
missing_pct    = (data.isna().mean() * 100).sort_values(ascending=False)
duplicate_count = int(data.duplicated().sum())
incomplete_rows = int(data.isna().any(axis=1).sum())
constant_cols   = [c for c in data.columns if data[c].nunique(dropna=False) <= 1]

quality = [
    ("Missing cells",           f"{missing_count:,}"),
    ("Duplicate rows",          f"{duplicate_count:,}"),
    ("Rows with missing values", f"{incomplete_rows:,}"),
    ("Constant columns",        f"{len(constant_cols):,}"),
]

# Build summary_lines for PDF
summary_lines = [
    f"Dataset contains {len(data):,} rows and {len(data.columns):,} columns.",
    f"{missing_count:,} missing cells ({missing_count / max(1, data.size) * 100:.2f}% of all cells).",
    f"{duplicate_count:,} duplicate rows.",
    f"{incomplete_rows:,} rows have at least one missing value.",
]
if revenue_col:
    total_revenue = pd.to_numeric(data[revenue_col], errors="coerce").sum(min_count=1)
    if pd.notna(total_revenue):
        summary_lines.append(f"Total {revenue_col}: {total_revenue:,.2f}.")
if date_col and data[date_col].notna().any():
    summary_lines.append(f"Date coverage: {data[date_col].min().date()} to {data[date_col].max().date()}.")
if customer_col:
    summary_lines.append(f"Unique customers: {data[customer_col].nunique(dropna=True):,}.")
if num_cols:
    summary_lines.append(f"Numeric columns: {', '.join(num_cols[:15])}{'…' if len(num_cols) > 15 else ''}.")
if cat_cols:
    summary_lines.append(f"Categorical columns: {', '.join(cat_cols[:15])}{'…' if len(cat_cols) > 15 else ''}.")

# Pre-compute universal analytics
col_stats_data  = column_stats(data)
corr_pairs_df   = correlation_pairs(data, num_cols)
outlier_data    = outlier_summary(data, num_cols)

# Add correlation highlights to summary_lines
if not corr_pairs_df.empty:
    strong = corr_pairs_df[corr_pairs_df["Strength"] == "Strong"]
    if not strong.empty:
        top = strong.iloc[0]
        summary_lines.append(
            f"Strongest correlation: {top['Column A']} ↔ {top['Column B']} (r = {top['Pearson r']})."
        )
    summary_lines.append(
        f"Correlation matrix covers {len(num_cols)} numeric columns; "
        f"{len(corr_pairs_df)} unique pairs computed."
    )

# Add outlier highlights
if outlier_data:
    top_o = outlier_data[0]
    summary_lines.append(
        f"Most outliers: {top_o['Column']} — {top_o['Outlier count']:,} values "
        f"({top_o['Outlier (%)']:.1f}% of non-null)."
    )

# ── Overview metrics ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Dataset overview</div>', unsafe_allow_html=True)
n_metric_cols = 5 if (revenue_col and customer_col) else 4
mc = st.columns(n_metric_cols)
mc[0].metric("Records",         f"{len(data):,}")
mc[1].metric("Columns",         f"{len(data.columns):,}")
mc[2].metric("Numeric fields",  f"{len(num_cols):,}")
mc[3].metric("Missing cells",   f"{missing_count:,}", f"{missing_count / max(1, data.size) * 100:.1f}%")
if revenue_col and customer_col:
    mc[4].metric("Unique customers", f"{data[customer_col].nunique(dropna=True):,}")

with st.expander("Preview the uploaded data", expanded=False):
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════
# 01 · DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-label">01 · Data quality</div>', unsafe_allow_html=True)
st.subheader("How complete and consistent is this dataset?")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Missing cells",    f"{missing_count:,}")
q2.metric("Duplicate rows",   f"{duplicate_count:,}")
q3.metric("Incomplete rows",  f"{incomplete_rows:,}")
q4.metric("Constant columns", f"{len(constant_cols):,}")

if missing_pct.max() > 0:
    mp = missing_pct[missing_pct > 0].head(20)
    st.plotly_chart(
        px.bar(pd.DataFrame({"Column": mp.index, "Missing (%)": mp.values}).sort_values("Missing (%)"),
               x="Missing (%)", y="Column", orientation="h",
               title="Columns with missing values", template="plotly_white",
               color="Missing (%)", color_continuous_scale=["#8bc6c9","#c75b5b"]
               ).update_layout(coloraxis_showscale=False),
        use_container_width=True
    )
else:
    st.success("No missing values detected.")

quality_details = []
for c in data.columns:
    s = data[c]
    if s.isna().any():
        quality_details.append({"Finding":"Missing values","Column":c,
            "Details":f"{int(s.isna().sum()):,} missing ({s.isna().mean()*100:.1f}%)"})
    if s.nunique(dropna=False) <= 1:
        quality_details.append({"Finding":"Constant column","Column":c,
            "Details":"Only one distinct value including missing values"})
if quality_details:
    st.dataframe(pd.DataFrame(quality_details), use_container_width=True, hide_index=True)
if date_col:
    invalid_dates = int(raw_df[date_col].notna().sum() - data[date_col].notna().sum())
    st.caption(f"Date parsing: {int(data[date_col].notna().sum()):,} valid; {max(0,invalid_dates):,} could not be parsed.")

# ═══════════════════════════════════════════════════════════════════════
# 02 · PER-COLUMN STATISTICS (universal — every dataset)
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-label">02 · Column-by-column analysis</div>', unsafe_allow_html=True)
st.subheader("Detailed statistics for every column")

stats_df = pd.DataFrame(col_stats_data)
st.dataframe(stats_df, use_container_width=True, hide_index=True)

# Auto-charts for every numeric column
if num_cols:
    st.markdown("#### Distributions — all numeric columns")
    n_per_row = 3
    chunks = [num_cols[i:i+n_per_row] for i in range(0, min(len(num_cols), 30), n_per_row)]
    for chunk in chunks:
        cols_ui = st.columns(len(chunk))
        for ui_col, c in zip(cols_ui, chunk):
            s = data[c].replace([np.inf,-np.inf], np.nan).dropna()
            if len(s) < 2: continue
            fig = px.histogram(s.to_frame(), x=c, nbins=30, marginal="box",
                               template="plotly_white", color_discrete_sequence=["#187f83"],
                               title=f"{c}")
            fig.update_layout(showlegend=False, margin=dict(t=35,b=20,l=10,r=10), height=280)
            ui_col.plotly_chart(fig, use_container_width=True)

# Auto-charts for every categorical column
if cat_cols:
    st.markdown("#### Frequency — all categorical columns")
    chunks_cat = [cat_cols[i:i+n_per_row] for i in range(0, min(len(cat_cols), 21), n_per_row)]
    for chunk in chunks_cat:
        cols_ui = st.columns(len(chunk))
        for ui_col, c in zip(cols_ui, chunk):
            counts = (data[c].fillna("(Missing)").astype(str)
                      .value_counts().head(10)
                      .rename_axis(c).reset_index(name="Count"))
            fig = px.bar(counts.sort_values("Count"), x="Count", y=c, orientation="h",
                         template="plotly_white", color_discrete_sequence=["#174b65"],
                         title=f"{c}")
            fig.update_layout(showlegend=False, margin=dict(t=35,b=20,l=10,r=10), height=280)
            ui_col.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# 03 · CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-label">03 · Correlations</div>', unsafe_allow_html=True)
st.subheader("Which numeric columns move together?")

if len(num_cols) < 2:
    st.info("Need at least two numeric columns for correlation analysis.")
else:
    corr_matrix = data[num_cols[:25]].corr(numeric_only=True)
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation matrix — all numeric columns", template="plotly_white"
    )
    fig_corr.update_layout(height=max(350, len(num_cols[:25]) * 28))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Pearson correlation. Values close to ±1 indicate a strong linear relationship.")

    if not corr_pairs_df.empty:
        st.markdown("#### All correlation pairs ranked by strength")
        st.dataframe(corr_pairs_df, use_container_width=True, hide_index=True)
        strong_pairs = corr_pairs_df[corr_pairs_df["Strength"] == "Strong"]
        if not strong_pairs.empty:
            st.markdown("#### Scatter plots for strongly correlated pairs")
            for _, row in strong_pairs.head(6).iterrows():
                ca, cb = row["Column A"], row["Column B"]
                plot_df = data[[ca, cb]].replace([np.inf, -np.inf], np.nan).dropna()
                fig_sc = px.scatter(
                    plot_df, x=ca, y=cb,
                    title=f"{ca} vs {cb}  (r = {row['Pearson r']})",
                    template="plotly_white", color_discrete_sequence=["#187f83"],
                    opacity=0.6,
                )
                # Add numpy polyfit trendline (no statsmodels dependency)
                if len(plot_df) >= 2:
                    m, b = np.polyfit(plot_df[ca].astype(float), plot_df[cb].astype(float), 1)
                    x_range = np.linspace(plot_df[ca].min(), plot_df[ca].max(), 100)
                    import plotly.graph_objects as go
                    fig_sc.add_trace(go.Scatter(
                        x=x_range, y=m * x_range + b,
                        mode="lines", line=dict(color="#c75b5b", width=2),
                        name="Trend",
                    ))
                st.plotly_chart(fig_sc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# 04 · OUTLIER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-label">04 · Outlier analysis</div>', unsafe_allow_html=True)
st.subheader("Which columns contain unusual values?")

if not outlier_data:
    st.success("No outliers detected in any numeric column using the IQR rule.")
else:
    st.dataframe(pd.DataFrame(outlier_data), use_container_width=True, hide_index=True)
    st.caption("IQR rule: values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR. These are candidates for review, not confirmed errors.")
    st.markdown("#### Box plots — columns with outliers")
    outlier_cols = [o["Column"] for o in outlier_data[:12]]
    n_per_row_o  = 3
    for chunk in [outlier_cols[i:i+n_per_row_o] for i in range(0, len(outlier_cols), n_per_row_o)]:
        cols_ui = st.columns(len(chunk))
        for ui_col, c in zip(cols_ui, chunk):
            s = data[c].replace([np.inf,-np.inf], np.nan).dropna()
            fig = px.box(s.to_frame(), y=c, points="outliers",
                         template="plotly_white", color_discrete_sequence=["#187f83"],
                         title=f"{c}")
            fig.update_layout(showlegend=False, margin=dict(t=35,b=20,l=10,r=10), height=280)
            ui_col.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# 05 · EXPLORE PATTERNS (group comparisons)
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-label">05 · Explore patterns</div>', unsafe_allow_html=True)
st.subheader("How do groups differ?")

if cat_cols and num_cols:
    cmp_l, cmp_r = st.columns(2)
    with cmp_l:
        compare_cat = st.selectbox("Group by (categorical)", cat_cols, key="group_compare")
    with cmp_r:
        compare_metric = st.selectbox("Measure (numeric)", num_cols, key="metric_compare")
    agg_choice = st.selectbox("Aggregation", ["Mean","Median","Sum","Record count"], key="aggregation")

    gd = data[[compare_cat, compare_metric]].copy()
    gd[compare_cat] = gd[compare_cat].fillna("(Missing)").astype(str)
    grp = gd.groupby(compare_cat)[compare_metric]
    if agg_choice == "Mean":
        grouped = grp.agg(["mean","count"]).reset_index().rename(columns={"mean":"Value","count":"Sample size"})
    elif agg_choice == "Median":
        grouped = grp.agg(["median","count"]).reset_index().rename(columns={"median":"Value","count":"Sample size"})
    elif agg_choice == "Sum":
        grouped = grp.agg(["sum","count"]).reset_index().rename(columns={"sum":"Value","count":"Sample size"})
    else:
        grouped = grp.size().reset_index(name="Value"); grouped["Sample size"] = grouped["Value"]
    grouped = grouped.sort_values("Value", ascending=False).head(25)
    st.plotly_chart(
        px.bar(grouped.sort_values("Value"), x="Value", y=compare_cat, orientation="h",
               title=f"{agg_choice} of {compare_metric} by {compare_cat}",
               hover_data=["Sample size"] if "Sample size" in grouped else None,
               template="plotly_white"),
        use_container_width=True
    )
    st.dataframe(grouped, use_container_width=True, hide_index=True)
elif num_cols:
    st.info("Add categorical columns to compare groups.")
else:
    st.info("No numeric or categorical columns found for group comparison.")

# ═══════════════════════════════════════════════════════════════════════
# 06 · TIME TRENDS
# ═══════════════════════════════════════════════════════════════════════
if date_col and data[date_col].notna().sum() >= 2:
    st.divider()
    st.markdown('<div class="section-label">06 · Time trends</div>', unsafe_allow_html=True)
    st.subheader("How does the dataset change over time?")

    date_valid = data.dropna(subset=[date_col]).copy()
    date_valid["_period"] = date_valid[date_col].dt.to_period("M").dt.to_timestamp()

    if revenue_col:
        time_agg = date_valid.groupby("_period")[revenue_col].sum(min_count=1).reset_index()
        y_axis, title = revenue_col, f"Monthly {revenue_col}"
    elif num_cols:
        measure  = st.selectbox("Numeric measure over time", num_cols, key="time_measure")
        time_agg = date_valid.groupby("_period")[measure].mean().reset_index()
        y_axis, title = measure, f"Monthly average {measure}"
    else:
        time_agg = date_valid.groupby("_period").size().reset_index(name="Records")
        y_axis, title = "Records", "Records by month"

    time_agg.columns = ["Month", y_axis]
    if len(time_agg):
        st.plotly_chart(
            px.line(time_agg, x="Month", y=y_axis, markers=True, title=title,
                    template="plotly_white", color_discrete_sequence=["#187f83"]),
            use_container_width=True
        )
        st.caption(f"Monthly aggregation: {time_agg['Month'].min().date()} to {time_agg['Month'].max().date()}.")
        if len(time_agg) >= 2:
            fv, lv = time_agg[y_axis].iloc[0], time_agg[y_axis].iloc[-1]
            if pd.notna(fv) and pd.notna(lv) and fv != 0:
                st.caption(f"First to last observed month: {(lv-fv)/abs(fv)*100:+.1f}% change.")

    # Additional time breakdowns for every numeric column
    if num_cols:
        with st.expander("Monthly trends for all numeric columns"):
            trend_chunks = [num_cols[i:i+3] for i in range(0, min(len(num_cols), 12), 3)]
            for chunk in trend_chunks:
                cols_ui = st.columns(len(chunk))
                for ui_col, c in zip(cols_ui, chunk):
                    agg = date_valid.groupby("_period")[c].mean().reset_index()
                    agg.columns = ["Month", c]
                    if len(agg) < 2: continue
                    fig = px.line(agg, x="Month", y=c, title=f"Monthly avg {c}",
                                  template="plotly_white", color_discrete_sequence=["#174b65"])
                    fig.update_layout(height=220, margin=dict(t=35,b=20,l=10,r=10))
                    ui_col.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# 07 · BUSINESS ANALYSIS (only when relevant roles exist)
# ═══════════════════════════════════════════════════════════════════════
can_rfm = bool(customer_col and date_col and order_col and revenue_col)
if revenue_col or customer_col:
    st.divider()
    st.markdown('<div class="section-label">07 · Business analysis</div>', unsafe_allow_html=True)
    st.subheader("Business and customer performance")

    if revenue_col:
        total = pd.to_numeric(data[revenue_col], errors="coerce").sum(min_count=1)
        ba1, ba2 = st.columns(2)
        ba1.metric(f"Total {revenue_col}", f"{total:,.2f}" if pd.notna(total) else "—")
        if order_col:
            ba2.metric("Unique orders", f"{data[order_col].nunique(dropna=True):,}")
        elif customer_col:
            ba2.metric("Unique customers", f"{data[customer_col].nunique(dropna=True):,}")
        if product_col:
            top_prod = (data.groupby(product_col, dropna=False)[revenue_col]
                        .sum(min_count=1).sort_values(ascending=False).head(15).reset_index())
            st.plotly_chart(
                px.bar(top_prod.sort_values(revenue_col), x=revenue_col, y=product_col,
                       orientation="h",
                       title=f"Which {product_col} drives most {revenue_col}?",
                       template="plotly_white"),
                use_container_width=True
            )
        if location_col:
            by_loc = (data.groupby(location_col, dropna=False)[revenue_col]
                      .sum(min_count=1).sort_values(ascending=False).head(15).reset_index())
            st.plotly_chart(
                px.bar(by_loc.sort_values(revenue_col), x=revenue_col, y=location_col,
                       orientation="h", title=f"{revenue_col} by {location_col}",
                       template="plotly_white"),
                use_container_width=True
            )

    if can_rfm:
        st.markdown("#### RFM customer analysis")
        tx  = data.dropna(subset=[customer_col, date_col, revenue_col]).copy()
        rfm = tx.groupby(customer_col, dropna=True).agg(
            LastPurchase=(date_col, "max"),
            Frequency=(order_col, "nunique"),
            Monetary=(revenue_col, "sum"),
        ).reset_index()
        if len(rfm):
            snapshot = tx[date_col].max().normalize() + pd.Timedelta(days=1)
            rfm["RecencyDays"]       = (snapshot - rfm["LastPurchase"].dt.normalize()).dt.days
            rfm["AverageOrderValue"] = rfm["Monetary"] / rfm["Frequency"].replace(0, np.nan)
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Customers",           f"{len(rfm):,}")
            rc2.metric("Median recency",      f"{rfm['RecencyDays'].median():.0f} days")
            rc3.metric("Median customer spend", f"{rfm['Monetary'].median():,.2f}")
            st.plotly_chart(
                px.scatter(rfm, x="Frequency", y="Monetary", color="RecencyDays",
                           hover_data=[customer_col],
                           title="Customer frequency vs. monetary value",
                           color_continuous_scale="Teal", template="plotly_white"),
                use_container_width=True
            )
            st.dataframe(rfm.sort_values("Monetary", ascending=False).head(100),
                         use_container_width=True, hide_index=True)
            st.download_button("Download RFM table", rfm.to_csv(index=False).encode("utf-8"),
                               "customeriq_rfm.csv", "text/csv")
    elif customer_col:
        st.info("Customer IDs present. RFM also needs transaction date, order ID, and a monetary column.")

# ═══════════════════════════════════════════════════════════════════════
# 08 · RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-label">08 · Recommendations</div>', unsafe_allow_html=True)
st.subheader("Insights & what to do next")

insights = []
# Missing data
if missing_pct.max() > 20:
    col = str(missing_pct.idxmax()); pct = float(missing_pct.max())
    insights.append(("High", f"Review missing data in '{col}'",
                     f"{pct:.1f}% of values in this column are missing.",
                     "Check collection process; decide whether to impute or exclude before analysis."))
# Duplicates
if duplicate_count:
    insights.append(("Medium", "Review duplicate records",
                     f"{duplicate_count:,} exact duplicate rows detected.",
                     "Confirm whether duplicates are valid repeats or data entry errors before removing."))
# Skewed distributions
if num_cols:
    skewed = []
    for c in num_cols[:40]:
        s = data[c].replace([np.inf,-np.inf], np.nan).dropna()
        if len(s) >= 5 and s.nunique() > 1 and s.median() != 0 and abs(s.mean()/s.median()) >= 1.5:
            skewed.append((c, float(s.mean()), float(s.median()), len(s)))
    if skewed:
        c, mean, med, n = sorted(skewed, key=lambda x: abs(x[1]/x[2]), reverse=True)[0]
        insights.append(("Low", f"Inspect distribution of '{c}'",
                         f"Mean {mean:,.3g}; median {med:,.3g}; {n:,} valid values — heavy skew detected.",
                         "Review histogram and outliers; prefer median over mean for summary reporting."))
# Time trend
if date_col and data[date_col].notna().sum() >= 2:
    dated = data.dropna(subset=[date_col]).copy()
    dated["_month"] = dated[date_col].dt.to_period("M")
    counts = dated.groupby("_month").size()
    if len(counts) >= 2 and counts.iloc[0] > 0:
        change = (counts.iloc[-1] - counts.iloc[0]) / counts.iloc[0] * 100
        insights.append(("Medium", "Activity change over time",
                         f"Record count changed {change:+.1f}% from first to last observed month "
                         f"({int(counts.iloc[0]):,} → {int(counts.iloc[-1]):,} records).",
                         "Investigate whether this reflects real trends, partial periods, or collection gaps."))
# Product concentration
if revenue_col and product_col:
    totals = data.groupby(product_col, dropna=False)[revenue_col].sum(min_count=1).dropna().sort_values(ascending=False)
    grand  = totals.sum()
    if len(totals) >= 2 and grand > 0:
        share = float(totals.iloc[0] / grand * 100)
        insights.append(("Medium" if share >= 50 else "Low", f"Revenue concentration in '{product_col}'",
                         f"Top category '{str(totals.index[0])[:60]}' = {share:.1f}% of total {revenue_col}.",
                         "Assess whether concentration on one category is a business risk."))
# Outliers
if outlier_data:
    top_o = outlier_data[0]
    insights.append(("Medium" if top_o["Outlier (%)"] > 5 else "Low",
                     f"Outliers detected in '{top_o['Column']}'",
                     f"{top_o['Outlier count']:,} outlier values ({top_o['Outlier (%)']:.1f}%) "
                     f"outside [{top_o['Lower fence']}, {top_o['Upper fence']}].",
                     "Review box plot; confirm whether extremes are valid or data errors."))
# Strong correlations
if not corr_pairs_df.empty:
    strong_c = corr_pairs_df[corr_pairs_df["Strength"] == "Strong"]
    if not strong_c.empty:
        row = strong_c.iloc[0]
        insights.append(("Low", f"Strong association: '{row['Column A']}' ↔ '{row['Column B']}'",
                         f"Pearson r = {row['Pearson r']} — {row['Direction'].lower()} linear relationship.",
                         "Investigate with a scatter plot; correlation does not imply causation."))

if not insights:
    st.info("No strong automatic recommendation. Review charts and column roles above for further analysis.")
else:
    for priority, finding, evidence, action in insights:
        with st.container(border=True):
            st.markdown(f"**{finding}**")
            st.caption(f"Priority: {priority}")
            st.write(f"**Evidence:** {evidence}")
            st.write(f"**What to do:** {action}")
st.caption("Recommendations are generated from calculated dataset statistics — investigation opportunities, not proven causes.")

# ── AI interpretation ──────────────────────────────────────────────────────────
stored_ai_insights: list[dict] = []
if get_api_key() and insights:
    with st.expander("Optional AI interpretation of these findings"):
        if st.button("Generate AI interpretation", key="groq_interpret"):
            try:
                model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip() or "qwen/qwen3.8-27b"
                # Build a richer payload: include per-column stats + correlations + outliers
                corr_top = (corr_pairs_df.head(10).to_dict(orient="records")
                            if not corr_pairs_df.empty else [])
                aggregate_payload = {
                    "dataset_shape":       {"rows": int(len(data)), "columns": int(len(data.columns))},
                    "quality":             {"missing_cells": missing_count, "duplicate_rows": duplicate_count,
                                            "incomplete_rows": incomplete_rows},
                    "numeric_columns":     num_cols[:20],
                    "categorical_columns": cat_cols[:20],
                    "top_correlations":    corr_top,
                    "outlier_columns":     [o["Column"] for o in outlier_data[:10]],
                    "calculated_findings": [
                        {"priority": p, "finding": f, "evidence": e, "action": a}
                        for p, f, e, a in insights
                    ],
                }
                ai_items = _groq_insights_cached(
                    json.dumps(aggregate_payload, ensure_ascii=False, sort_keys=True),
                    model_name, get_api_key()
                )
                if not ai_items:
                    st.info("AI did not find an additional evidence-supported interpretation.")
                else:
                    stored_ai_insights = ai_items
                    for item in ai_items:
                        st.markdown(f"**{item['finding']}** · {item['priority']} priority")
                        st.write(f"**Evidence:** {item['evidence']}")
                        st.write(f"**Why it matters:** {item['why_it_matters']}")
                        st.write(f"**Suggested action:** {item['action']}")
            except Exception as _ai_exc:
                st.warning(f"AI request failed: {type(_ai_exc).__name__}: {_ai_exc}")

# ═══════════════════════════════════════════════════════════════════════
# 09 · K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-label">09 · Optional machine learning</div>', unsafe_allow_html=True)
with st.expander("Explore K-Means clusters"):
    st.write("Clustering explores groups in numerical features. It is exploratory and does not predict outcomes.")
    eligible_num = [c for c in num_cols if data[c].nunique(dropna=True) > 1]
    selected_features = st.multiselect("Select at least two numeric features", eligible_num,
                                       default=eligible_num[:min(3, len(eligible_num))],
                                       key="cluster_features")
    if st.button("Run K-Means clustering", key="run_clusters"):
        try:
            clustered, score, used_features = cluster_frame(data, selected_features, n_clusters)
            st.success(f"Clustered {len(clustered):,} rows using {', '.join(used_features)}.")
            if np.isfinite(score):
                st.metric("Silhouette score", f"{score:.3f}")
            else:
                st.caption("Silhouette score not meaningful for this result.")
            st.dataframe(clustered.head(1000), use_container_width=True, hide_index=True)
            if len(used_features) >= 2:
                st.plotly_chart(
                    px.scatter(clustered[used_features[:2]+["Cluster"]], x=used_features[0],
                               y=used_features[1], color="Cluster",
                               title="K-Means groups across first two selected features",
                               template="plotly_white"),
                    use_container_width=True
                )
            st.download_button("Download clustered dataset",
                               clustered.to_csv(index=False).encode("utf-8"),
                               "customeriq_clustered.csv", "text/csv")
        except Exception as exc:
            st.warning(str(exc))

# ═══════════════════════════════════════════════════════════════════════
# 10 · EXPORT
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-label">10 · Export</div>', unsafe_allow_html=True)
st.subheader("Take your analysis with you")
exp_l, exp_r = st.columns(2)

with exp_l:
    st.download_button("Download analyzed dataset (CSV)",
                       data.to_csv(index=False).encode("utf-8"),
                       "customeriq_dataset.csv", "text/csv", use_container_width=True)
    roles_df = pd.DataFrame({
        "Column":    list(mapping.keys()),
        "Role":      [mapping[c] for c in mapping],
        "Data type": [str(data[c].dtype) for c in mapping],
    })
    st.download_button("Download column-role mapping",
                       roles_df.to_csv(index=False).encode("utf-8"),
                       "customeriq_column_roles.csv", "text/csv", use_container_width=True)
    st.download_button("Download full column statistics",
                       stats_df.to_csv(index=False).encode("utf-8"),
                       "customeriq_column_stats.csv", "text/csv", use_container_width=True)
    if not corr_pairs_df.empty:
        st.download_button("Download correlation pairs",
                           corr_pairs_df.to_csv(index=False).encode("utf-8"),
                           "customeriq_correlations.csv", "text/csv", use_container_width=True)

with exp_r:
    try:
        pdf_bytes = create_pdf_report(
            data, mapping, schema["dataset_type"], source_name,
            quality, summary_lines,
            col_stats_data, corr_pairs_df, outlier_data,
            ai_insights=stored_ai_insights if stored_ai_insights else None,
        )
        st.download_button("Download PDF analytics report",
                           pdf_bytes, "CustomerIQ_Analytics_Report.pdf",
                           "application/pdf", use_container_width=True)
        st.caption(
            "PDF includes: dataset profile, data quality, per-column stats, "
            "full correlation table, outlier summary, column roles, and AI insights (if generated)."
        )
    except Exception as exc:
        st.warning(f"PDF could not be generated: {type(exc).__name__}: {exc}. CSV exports remain available.")

st.caption("CustomerIQ · Dynamic CSV analytics · All calculations run locally in Python.")
