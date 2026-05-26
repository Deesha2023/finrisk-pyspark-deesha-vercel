import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="FinRisk AI Dashboard", layout="wide")

# Title and description
st.title("🚀 Distributed Financial Risk Analytics Platform")
st.markdown("Powered by **PySpark** (distributed processing) & **AI risk scoring**")

# Load the risk results (CSV format)
try:
    df = pd.read_csv("risk_results.csv")
    st.success("✅ Data loaded successfully")
except FileNotFoundError:
    st.error("❌ risk_results.csv not found. Please run `python risk_score_pyspark.py` first.")
    st.stop()

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Customers", df['customer_id'].nunique())
col2.metric("💸 Total Transactions", len(df))
col3.metric("⚠️ High Risk (>70)", len(df[df['risk_score'] > 70]))
col4.metric("📊 Avg Risk Score", round(df['risk_score'].mean(), 2))

# ---------- Risk Score Distribution (Histogram) ----------
st.subheader("📈 Risk Score Distribution")
fig = px.histogram(
    df, 
    x='risk_score', 
    nbins=30, 
    title='Distribution of AI‑Computed Risk Scores',
    color_discrete_sequence=['#FF4B4B'],
    labels={'risk_score': 'Risk Score (0–100)'}
)
fig.update_layout(bargap=0.05)
st.plotly_chart(fig, width='stretch')   # ✅ updated from use_container_width=True

# ---------- Top 10 Riskiest Customers ----------
st.subheader("🔥 Top 10 Highest‑Risk Customers")
top_risk = df.groupby('customer_id')['risk_score'].max().reset_index()
top_risk = top_risk.sort_values('risk_score', ascending=False).head(10)
st.dataframe(top_risk, use_container_width=True)   # This one still works (no deprecation warning for dataframe)

# ---------- Real‑time Fraud Alerts (risk > 80) ----------
st.subheader("🚨 Real‑time Alerts (Risk Score > 80)")
alerts = df[df['risk_score'] > 80].sort_values('risk_score', ascending=False).head(20)
if len(alerts) > 0:
    st.dataframe(
        alerts[['txn_id', 'customer_id', 'amount', 'risk_score', 'merchant', 'timestamp']],
        use_container_width=True
    )
else:
    st.info("No high‑risk transactions at the moment.")

# ---------- Optional: Risk by Merchant ----------
st.subheader("🏪 Average Risk Score by Merchant")
merchant_risk = df.groupby('merchant')['risk_score'].mean().reset_index().sort_values('risk_score', ascending=False)
fig2 = px.bar(merchant_risk, x='merchant', y='risk_score', title='Average Risk per Merchant',
              color='risk_score', color_continuous_scale='Reds')
st.plotly_chart(fig2, width='stretch')   # ✅ updated

# Footer
st.markdown("---")
st.caption("⚡ Built with PySpark, Streamlit & Plotly | AI risk scoring simulates a Random Forest model")