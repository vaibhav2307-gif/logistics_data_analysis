<<<<<<< HEAD
# Global Supply Chain Risk 2026

## Problem statement

This project identifies shipments with elevated disruption risk so logistics planners can spend limited review, alternate-routing, and contingency capacity where it is most likely to reduce service impact. The source is `archive/global_supply_chain_risk_2026.csv`.

## Four-tier methodology

1. **Data hygiene and architecture:** `app.py` loads the real CSV with pandas, validates the observed schema, removes duplicate rows, parses dates, coerces numeric fields, flags values outside business bounds, imputes numeric medians and categorical `Unknown`, and reports the audit. The grain remains one row per unique `Shipment_ID` because IDs are unique in the supplied data.
2. **Descriptive and diagnostic analytics:** five charts cover monthly trend, transport-mode distribution, lead-time outcome distributions, origin-destination cohorts, and carrier-reliability cohorts. Each chart prints an empirical observation and explicitly avoids causal claims.
3. **Predictive analytics:** the target is `Disruption_Occurred`. A stratified 80/20 split feeds a preprocessing and Random Forest pipeline. `Shipment_ID`, raw `Date`, `Month`, `Risk_Band`, and the target are excluded. Categorical encoding is fit inside the training pipeline; unknown categories are handled safely.
4. **Prescriptive analytics:** all shipments receive a disruption probability. The dashboard ranks the top configurable percentage into a bounded intervention queue and maps observed risk signals to concrete logistics actions.

## Model metrics

Metrics are computed at runtime from the actual cleaned dataset and displayed in the dashboard: accuracy, precision, recall, ROC-AUC, a confusion matrix, and a classification report. This README intentionally does not hard-code results that could become stale after data changes.

## Business recommendations

- Review the highest predicted-risk shipments first, with capacity set by the slider rather than an unlimited alert stream.
- For high geopolitical risk, pre-book an alternate port and customs broker and confirm a second routing window.
- For low carrier reliability, require a carrier recovery plan and compare an alternate service against predicted risk.
- For long lead times, add milestone alerts and reserve expedite approvals for customer-critical shipments.
- For weather-exposed lanes, adjust departure or upstream staging and verify contingency routing and insurance.

False negatives can create missed disruption, expedite cost, and customer service exposure. False positives consume scarce review and buffer capacity. The ranked, capacity-constrained queue is designed to make that trade-off explicit; it is not a causal guarantee.

## Run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard expects the CSV at `archive/global_supply_chain_risk_2026.csv` relative to `app.py`.
=======
# logistics_data_analysis
An interactive logistics risk analytics dashboard that analyzes global supply chain disruptions, predicts shipment risk using Random Forest, and prioritizes high-risk shipments with actionable mitigation strategies.
>>>>>>> 749283a1342af6b53af783de76bef1dbeddbb5e1
