import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="BankOps AI Dashboard", page_icon="🏦", layout="wide")
st.title("🏦 BankOps AI Operations Dashboard")
st.caption("Infrastructure monitoring, model insights, and AI-assisted incident investigation")

def get_json(path):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Backend request failed: {exc}")
        return None

health = get_json("/health")
if health:
    st.success(f"Backend status: {health['status']}")

st.header("1. Infrastructure KPIs")
kpis = get_json("/api/kpis")
if kpis:
    timestamp = kpis.pop("timestamp", "Unknown")
    st.caption(f"Latest available dataset timestamp: {timestamp}")
    preferred = [
        "cpu_utilization", "memory_usage", "db_connection_pool_usage",
        "application_error_rate", "neft_success_rate", "imps_success_rate",
        "rtgs_success_rate", "avg_transaction_processing_time_ms",
        "auth_success_rate", "service_uptime",
    ]
    available = [key for key in preferred if key in kpis and kpis[key] is not None]
    cols = st.columns(4)
    for i, key in enumerate(available):
        cols[i % 4].metric(key.replace("_", " ").title(), f"{kpis[key]:,.2f}")

st.info("These are the latest recorded KPIs in infrastructure_metrics.csv, not live telemetry. Connect a streaming/monitoring source for true real-time values.")

st.header("2. Forecasted KPIs")
forecast = get_json("/api/forecast")
if forecast and forecast.get("status") == "not_configured":
    st.warning(forecast["message"])
    st.json(forecast.get("artifact", {}))
elif forecast:
    st.json(forecast)

st.header("3. Log insights and classification")
if st.button("Load recent application logs"):
    logs = get_json("/api/logs/insights?limit=100")
    if logs:
        st.write(f"Rows returned: {logs['rows']}")
        st.warning(logs["message"])
        st.dataframe(pd.DataFrame(logs["records"]), use_container_width=True)
        st.json(logs["artifact"])

st.header("4. AI-generated incident analysis")
incident = st.text_area(
    "Describe the incident and include available KPI values",
    placeholder="Example: CPU is 92%, memory is 88%, database connection pool is 96%, and NEFT success rate has dropped to 92%.",
    height=150,
)
if st.button("Generate root cause analysis", type="primary"):
    if not incident.strip():
        st.warning("Enter an incident description first.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/api/incidents/analyze",
                json={"description": incident},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            st.markdown(result["answer"])
            with st.expander("Retrieved knowledge-base sources"):
                st.json(result["sources"])
        except Exception as exc:
            st.error(f"Incident analysis failed: {exc}")
