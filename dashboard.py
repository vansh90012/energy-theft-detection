import streamlit as st
import pandas as pd
import plotly.express as px

from preprocess import load_consumption_data, build_feature_matrix
from model import load_model, explain_predictions

st.set_page_config(layout="wide")
st.title("⚡ Energy Theft Detection Dashboard")

df = load_consumption_data("datax/energy_data.csv")

# Sidebar
meter_id = st.sidebar.selectbox("Select Meter", sorted(df['meter_id'].unique()))
num_records = st.sidebar.slider("Records", 1, 5, 2)

filtered_df = df[df['meter_id'] == meter_id]
X_filtered, y_filtered, features = build_feature_matrix(filtered_df)

model, _ = load_model("models/theft_detector.joblib")

# ---------------- GRAPH ----------------
st.subheader("📈 Consumption")

fig = px.line(filtered_df, x="timestamp", y="consumption")
st.plotly_chart(fig, use_container_width=True)

# ---------------- TOP SUSPICIOUS METERS ----------------
st.subheader("🚨 Top Suspicious Meters")

X_all, _, features_all = build_feature_matrix(df)
probs = model.predict_proba(X_all)[:, 1]

df_risk = df.copy()
df_risk["risk"] = probs

meter_risk = df_risk.groupby("meter_id")["risk"].mean().reset_index()
meter_risk = meter_risk.sort_values(by="risk", ascending=False)

top_meters = meter_risk.head(10)

st.dataframe(top_meters)

highest = top_meters.iloc[0]
st.error(f"🚨 Highest Risk Meter: {int(highest['meter_id'])} (Risk: {highest['risk']:.3f})")

fig2 = px.bar(top_meters, x="meter_id", y="risk", color="risk")
st.plotly_chart(fig2, use_container_width=True)

# ---------------- DETECTION ----------------
st.subheader("🔍 Meter Analysis")

if st.button("Run Detection"):

    results = explain_predictions(model, X_filtered, features, num_records)

    for i, item in enumerate(results):

        st.markdown("---")

        if item["prediction"] == "Theft":
            st.error("⚠️ Theft Detected")
        else:
            st.success("Normal")

        st.write(f"Risk Score: {item['score']:.3f}")

        df_feat = pd.DataFrame({
            "Feature": list(item["top_features"].keys()),
            "Value": list(item["top_features"].values())
        })

        fig3 = px.bar(df_feat, x="Value", y="Feature", orientation="h")

        st.plotly_chart(fig3, key=f"chart_{i}")