"""Interactive churn model demo - Streamlit app."""

import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Insight ML", layout="wide")

st.title("Customer Churn Prediction")
st.caption("Predicting subscription churn with Python, scikit-learn, and an LLM-generated executive summary.")

with open("outputs/model_results.json") as f:
    results = json.load(f)

st.header("Model performance")
cols = st.columns(len(results))
for col, (name, r) in zip(cols, results.items()):
    col.metric(name, f"ROC-AUC {r['roc_auc']}")

st.header("Evaluation charts")
c1, c2 = st.columns(2)
c1.image("outputs/roc_curves.png", caption="ROC curves: model comparison")
c2.image("outputs/confusion_matrix.png", caption="Confusion matrix (best model)")
st.image("outputs/feature_importance.png", caption="What drives churn")

st.header("Executive summary")
with open("outputs/executive_summary.md") as f:
    st.markdown(f.read())

st.header("Sample of the data")
df = pd.read_csv("data/customers.csv")
st.dataframe(df.head(20), use_container_width=True)
