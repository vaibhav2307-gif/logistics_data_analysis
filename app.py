from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")

DATA_PATH = Path(__file__).parent / "archive" / "global_supply_chain_risk_2026.csv"
TARGET = "Disruption_Occurred"
REQUIRED_COLUMNS = {
    "Shipment_ID",
    "Date",
    "Origin_Port",
    "Destination_Port",
    "Transport_Mode",
    "Product_Category",
    "Distance_km",
    "Weight_MT",
    "Fuel_Price_Index",
    "Geopolitical_Risk_Score",
    "Weather_Condition",
    "Carrier_Reliability_Score",
    "Lead_Time_Days",
    TARGET,
}


@st.cache_data
def load_and_clean(path: str):
    raw = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required observed columns: {sorted(missing_columns)}")

    audit = {
        "raw_rows": len(raw),
        "raw_columns": len(raw.columns),
        "duplicate_rows": int(raw.duplicated().sum()),
    }

    df = raw.drop_duplicates().copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_columns = [
        "Distance_km",
        "Weight_MT",
        "Fuel_Price_Index",
        "Geopolitical_Risk_Score",
        "Carrier_Reliability_Score",
        "Lead_Time_Days",
        TARGET,
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    categorical_columns = [
        "Shipment_ID",
        "Origin_Port",
        "Destination_Port",
        "Transport_Mode",
        "Product_Category",
        "Weather_Condition",
    ]
    for column in categorical_columns:
        df[column] = df[column].astype("string").str.strip()

    bounds = {
        "Distance_km": (0, None),
        "Weight_MT": (0, None),
        "Fuel_Price_Index": (0, None),
        "Geopolitical_Risk_Score": (0, 10),
        "Carrier_Reliability_Score": (0, 1),
        "Lead_Time_Days": (0, None),
        TARGET: (0, 1),
    }
    anomalies = 0
    for column, (lower, upper) in bounds.items():
        invalid = df[column].isna() | (df[column] < lower)
        if upper is not None:
            invalid |= df[column] > upper
        anomalies += int(invalid.sum())
        df.loc[invalid, column] = np.nan

    before_impute = int(df.isna().sum().sum())
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())
    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")
    df["Date"] = df["Date"].fillna(df["Date"].median())

    df[TARGET] = df[TARGET].round().astype(int)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Risk_Band"] = pd.cut(
        df["Geopolitical_Risk_Score"],
        bins=[-0.01, 3.33, 6.66, 10.01],
        labels=["Low", "Medium", "High"],
    )

    audit.update(
        {
            "clean_rows": len(df),
            "invalid_values": anomalies,
            "cells_imputed": before_impute,
            "remaining_missing": int(df.isna().sum().sum()),
        }
    )
    return df, audit


@st.cache_resource
def train_model(df: pd.DataFrame):
    feature_columns = [
        "Origin_Port",
        "Destination_Port",
        "Transport_Mode",
        "Product_Category",
        "Distance_km",
        "Weight_MT",
        "Fuel_Price_Index",
        "Geopolitical_Risk_Score",
        "Weather_Condition",
        "Carrier_Reliability_Score",
        "Lead_Time_Days",
    ]
    X = df[feature_columns]
    y = df[TARGET]

    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]

    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "matrix": confusion_matrix(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    scored = df.copy()
    scored["Predicted_Disruption_Probability"] = model.predict_proba(X)[:, 1]
    return model, metrics, scored, X_test, y_test


def observation(text: str):
    st.caption(
        "Observation: "
        + text
        + " Correlation or association does not establish causation; operational validation is required before changing policy."
    )


st.set_page_config(page_title="Global Supply Chain Risk", page_icon="📦", layout="wide")

st.sidebar.markdown("### Navigation")
selected_view = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Modeling", "Strategy"],
    index=0,
    label_visibility="collapsed",
)

theme_mode = st.sidebar.radio("Theme", ["Light", "Executive Dark"], horizontal=True)

if theme_mode == "Executive Dark":
    theme_css = """
    <style>
    .stApp { background: linear-gradient(180deg, #020817 0%, #0b1120 30%, #111827 100%); color: #e5eefc; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stSidebar"] { background: rgba(2, 8, 23, 0.96); border-right: 1px solid rgba(148, 163, 184, 0.18); }
    .hero { background: linear-gradient(135deg, #0f172a 0%, #111827 35%, #1d4ed8 100%); border: 1px solid rgba(147, 197, 253, 0.25); border-radius: 20px; padding: 1.5rem 1.5rem 1.2rem 1.5rem; box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25); margin-bottom: 1rem; animation: fadeInUp 0.8s ease; }
    .hero h1 { color: #f8fbff !important; font-size: 2.3rem !important; font-weight: 800; margin: 0; }
    .hero p { color: rgba(226, 232, 240, 0.88); font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0; }
    [data-testid="stMetricContainer"] { background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(96, 165, 250, 0.18); border-radius: 18px; box-shadow: 0 10px 22px rgba(37, 99, 235, 0.12); padding: 0.8rem 0.9rem; }
    .metric-box { background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.96)); border: 1px solid rgba(147, 197, 253, 0.15); border-radius: 18px; padding: 0.9rem 1rem; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.25); animation: fadeInUp 0.8s ease; }
    .metric-label { color: #cbd5e1; font-size: 0.8rem; }
    .metric-value { color: #f8fbff; font-size: 2rem; font-weight: 800; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """
else:
    theme_css = """
    <style>
    .stApp { background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 35%, #f7f9fc 100%); color: #0f172a; }
    .hero { background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #60a5fa 100%); border-radius: 22px; padding: 1.5rem 1.5rem 1.2rem 1.5rem; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.15); margin-bottom: 1.2rem; animation: fadeInUp 0.8s ease; }
    .hero h1 { color: white !important; font-size: 2.3rem !important; margin: 0; font-weight: 800; }
    .hero p { color: rgba(255,255,255,0.85); font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0; }
    [data-testid="stMetricContainer"] { background: rgba(255,255,255,0.94); border: 1px solid #d9e2f3; border-radius: 16px; padding: 0.8rem 0.9rem; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08); }
    .metric-box { background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%); border-radius: 18px; padding: 0.9rem 1rem; border: 1px solid #dfe8ff; box-shadow: 0 6px 18px rgba(37, 99, 235, 0.08); animation: fadeInUp 0.8s ease; }
    .metric-label { color: #475569; font-size: 0.8rem; }
    .metric-value { color: #0f172a; font-size: 2rem; font-weight: 800; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """
st.markdown(theme_css, unsafe_allow_html=True)
st.sidebar.caption("Logistics risk intelligence console")

st.markdown(
    """
    <div class="hero">
        <h1>Global Supply Chain Risk | 2026</h1>
        <p>Shipment-level analytics for disruption prevention, from data hygiene through risk prioritization and operational actions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df, audit = load_and_clean(str(DATA_PATH))
except Exception as error:
    st.error(str(error))
    st.stop()

model, metrics, scored, X_test, y_test = train_model(df)
positive_rate = df[TARGET].mean()
high_risk_rate = (scored["Predicted_Disruption_Probability"] >= 0.5).mean()

if selected_view == "Overview":
    st.subheader("Executive KPIs")
    kpis = st.columns(5)
    with kpis[0]:
        st.markdown(
            '<div class="metric-box"><div class="metric-label">Shipments</div><div class="metric-value">{}</div></div>'.format(
                f"{len(df):,}"
            ),
            unsafe_allow_html=True,
        )
    with kpis[1]:
        st.markdown(
            '<div class="metric-box"><div class="metric-label">Observed disruptions</div><div class="metric-value">{}</div></div>'.format(
                f"{positive_rate:.1%}"
            ),
            unsafe_allow_html=True,
        )
    with kpis[2]:
        st.markdown(
            '<div class="metric-box"><div class="metric-label">Average lead time</div><div class="metric-value">{}</div></div>'.format(
                f"{df['Lead_Time_Days'].mean():.1f} days"
            ),
            unsafe_allow_html=True,
        )
    with kpis[3]:
        st.markdown(
            '<div class="metric-box"><div class="metric-label">High-risk predictions</div><div class="metric-value">{}</div></div>'.format(
                f"{high_risk_rate:.1%}"
            ),
            unsafe_allow_html=True,
        )
    with kpis[4]:
        st.markdown(
            '<div class="metric-box"><div class="metric-label">ROC-AUC</div><div class="metric-value">{}</div></div>'.format(
                f"{metrics['roc_auc']:.3f}"
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Data quality audit", expanded=False):
        st.json(audit)
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

elif selected_view == "EDA":
    st.header("Level 1 & 2 | Exploratory analytics")

    fig, ax = plt.subplots(figsize=(10, 4))
    monthly = df.groupby("Month", as_index=False)[TARGET].mean()
    sns.lineplot(data=monthly, x="Month", y=TARGET, marker="o", ax=ax)
    ax.set(xlabel="Shipment month", ylabel="Disruption rate", title="1. Monthly disruption trend")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig, clear_figure=True)
    peak = monthly.loc[monthly[TARGET].idxmax()]
    observation(
        f"The highest monthly observed disruption rate was {peak[TARGET]:.1%} in {peak['Month']}; the overall rate was {positive_rate:.1%}."
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    mode_rate = df.groupby("Transport_Mode", as_index=False)[TARGET].mean().sort_values(TARGET, ascending=False)
    sns.barplot(data=mode_rate, x="Transport_Mode", y=TARGET, ax=ax)
    ax.set(xlabel="Transport mode", ylabel="Disruption rate", title="2. Disruption rate by transport mode")
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    st.pyplot(fig, clear_figure=True)
    observation(
        f"{mode_rate.iloc[0]['Transport_Mode']} had the highest observed mode-level rate ({mode_rate.iloc[0][TARGET]:.1%}); this is an association, not evidence that mode choice caused disruption."
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(data=df, x="Lead_Time_Days", hue=TARGET, bins=30, stat="density", common_norm=False, element="step", ax=ax)
    ax.set(title="3. Lead-time distributions by disruption outcome", xlabel="Lead time (days)", ylabel="Density")
    st.pyplot(fig, clear_figure=True)
    median_by_target = df.groupby(TARGET)["Lead_Time_Days"].median()
    observation(
        f"Median lead time was {median_by_target.get(1, np.nan):.1f} days for disrupted shipments versus {median_by_target.get(0, np.nan):.1f} days for non-disrupted shipments."
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    cohort = pd.crosstab(df["Origin_Port"], df["Destination_Port"], values=df[TARGET], aggfunc="mean")
    sns.heatmap(cohort, annot=True, fmt=".0%", cmap="YlOrRd", ax=ax)
    ax.set(title="4. Origin-destination cohort disruption rates", xlabel="Destination port", ylabel="Origin port")
    st.pyplot(fig, clear_figure=True)
    observation(
        f"The origin-destination matrix shows cohort differences across {len(cohort)} origins and {cohort.shape[1]} destinations; sparse lanes should be validated with shipment counts before intervention."
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    reliability_bins = pd.cut(df["Carrier_Reliability_Score"], bins=5)
    reliability_rate = df.groupby(reliability_bins, observed=False)[TARGET].mean().reset_index()
    reliability_rate["Carrier_Reliability_Score"] = reliability_rate["Carrier_Reliability_Score"].astype(str)
    sns.barplot(data=reliability_rate, x="Carrier_Reliability_Score", y=TARGET, ax=ax)
    ax.set(title="5. Disruption rate across carrier-reliability cohorts", xlabel="Carrier reliability band", ylabel="Disruption rate")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig, clear_figure=True)
    observation(
        "Observed disruption rates vary across reliability bands. This pattern can support prioritization, but it does not prove that changing a carrier would independently reduce disruption."
    )

elif selected_view == "Modeling":
    st.header("Level 3 | Predictive modeling")
    st.write(
        "Target: `Disruption_Occurred`. The model uses an 80/20 stratified split. Shipment IDs, raw dates, target derivatives, and any unknown columns are excluded from predictors. All categorical encoding is fit on the training partition only through the pipeline."
    )

    metric_cols = st.columns(4)
    for i, (column, label) in enumerate(
        [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("roc_auc", "ROC-AUC")]
    ):
        metric_cols[i].metric(label, f"{metrics[column]:.3f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(metrics["matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test set)")
    st.pyplot(fig, clear_figure=True)

    st.dataframe(pd.DataFrame(metrics["report"]).transpose().round(3), use_container_width=True)
    st.info(
        "Commercial trade-off: false negatives are disrupted shipments missed by the intervention queue, risking service failures, expedite cost, and customer impact. False positives consume scarce review or buffer capacity and can unnecessarily divert attention from routine flows. In this setting, rank-based review of the highest probabilities can protect recall while keeping intervention volume bounded."
    )

    st.header("Data quality audit & intervention threshold")
    threshold_values = np.linspace(0.15, 0.9, 61)
    threshold_summary = []
    for threshold in threshold_values:
        flagged = scored["Predicted_Disruption_Probability"] >= threshold

        if flagged.sum() == 0:
            actual_rate = 0.0
            precision_value = 0.0
            recall_value = 0.0
        else:
            detected = scored.loc[flagged, TARGET]
            tp = int(((scored["Predicted_Disruption_Probability"] >= threshold) & (scored[TARGET] == 1)).sum())
            actual_positives = int((scored[TARGET] == 1).sum())
            actual_rate = float(detected.mean()) if len(detected) else 0.0
            precision_value = tp / int(flagged.sum()) if int(flagged.sum()) else 0.0
            recall_value = tp / actual_positives if actual_positives else 0.0

        threshold_summary.append(
            {
                "Threshold": threshold,
                "Flagged_Share": float(flagged.mean()),
                "Actual_Disruption_Rate_in_Queue": actual_rate,
                "Precision": precision_value,
                "Recall": recall_value,
                "Flagged_Count": int(flagged.sum()),
            }
        )
    threshold_summary = pd.DataFrame(threshold_summary)

    intervention_threshold = st.slider(
        "Intervention probability threshold",
        min_value=0.15,
        max_value=0.90,
        value=0.50,
        step=0.01,
    )
    selected_flag = scored["Predicted_Disruption_Probability"] >= intervention_threshold
    selected_queue = scored[selected_flag].sort_values("Predicted_Disruption_Probability", ascending=False)
    selected_precision = (selected_queue[TARGET].mean()) if len(selected_queue) else 0.0
    current_alert_rate = selected_flag.mean()
    current_recall = (
        (selected_queue[TARGET] == 1).sum() / (scored[TARGET] == 1).sum()
    ) if (scored[TARGET] == 1).sum() else 0.0

    st.metric("Flagged shipments", f"{int(selected_flag.sum()):,}", f"{current_alert_rate:.1%} of all shipments")
    st.metric("Observed disruption in queue", f"{selected_precision:.1%}", f"Recall {current_recall:.1%}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(threshold_summary["Threshold"], threshold_summary["Flagged_Share"], label="Flagged share", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(
        threshold_summary["Threshold"],
        threshold_summary["Actual_Disruption_Rate_in_Queue"],
        label="Actual disruption rate in queue",
        color="tab:orange",
    )
    ax.axvline(intervention_threshold, color="red", linestyle="--", linewidth=1.5, label=f"Selected threshold={intervention_threshold:.2f}")
    ax.set_title("Threshold sensitivity: flagged volume vs disruption rate")
    ax.set_xlabel("Intervention threshold")
    ax.set_ylabel("Flagged share")
    ax2.set_ylabel("Actual disruption rate in queue")
    fig.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)

    st.write(
        "Data-quality view: as the intervention threshold rises, fewer shipments are flagged and the queue becomes more selective. This changes the precision/recall trade-off but does not mean the observed rate is causal; it reflects model-driven prioritization based on the current data and feature set."
    )

elif selected_view == "Strategy":
    st.header("Level 4 | Prescriptive strategy")
    intervention_threshold = st.slider(
        "Intervention probability threshold",
        min_value=0.15,
        max_value=0.90,
        value=0.50,
        step=0.01,
    )
    selected_flag = scored["Predicted_Disruption_Probability"] >= intervention_threshold
    priority = scored[selected_flag].sort_values("Predicted_Disruption_Probability", ascending=False)

    st.metric("Intervention queue", f"{len(priority):,} shipments", f"Threshold {intervention_threshold:.2f}")
    st.write(
        "Operational rule: assign the limited review team to the top-risk shipments by predicted probability. For each queued shipment, verify carrier booking, weather exposure, port congestion, and geopolitical escalation; then select a concrete mitigation rather than applying a blanket mode change."
    )

    recommendations = pd.DataFrame(
        {
            "Trigger": [
                "High geopolitical risk",
                "Low carrier reliability",
                "Long lead time",
                "Weather risk / exposed lane",
            ],
            "Action": [
                "Pre-book alternate port and customs broker; confirm a second routing window.",
                "Request carrier recovery plan and reserve a vetted alternate service.",
                "Add milestone alerts and pre-authorize expedite only for customer-critical loads.",
                "Move departure window or stage inventory upstream; validate insurance and contingency routing.",
            ],
            "Control": [
                "Review only queued shipments; track avoided disruptions by lane.",
                "Compare incremental cost with predicted-risk reduction.",
                "Cap expedite approvals within the review budget.",
                "Require planner confirmation and record the selected mitigation.",
            ],
        }
    )
    st.dataframe(recommendations, hide_index=True, use_container_width=True)

    st.subheader("Prioritized shipment queue")
    st.dataframe(
        priority[
            [
                "Shipment_ID",
                "Origin_Port",
                "Destination_Port",
                "Transport_Mode",
                "Weather_Condition",
                "Geopolitical_Risk_Score",
                "Carrier_Reliability_Score",
                "Lead_Time_Days",
                "Predicted_Disruption_Probability",
            ]
        ]
        .head(100)
        .round(3),
        use_container_width=True,
    )

    st.download_button(
        "Download scored shipments",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_supply_chain_shipments.csv",
        mime="text/csv",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard summary")
st.sidebar.write(f"Shipments: {len(df):,}")
st.sidebar.write(f"Current disruption rate: {positive_rate:.1%}")
st.sidebar.write(f"Model AUC: {metrics['roc_auc']:.3f}")
