# Churn Model Executive Summary

The stronger model in this comparison was **Logistic Regression**, with a ROC-AUC of 0.7156. ROC-AUC measures how well the model separates customers who churn from those who stay: 0.5 is a coin flip and 1.0 is perfect, so 0.7156 indicates useful, better-than-random predictive signal.

Commercially, a model at this level lets the business rank customers by churn risk and focus retention spend on the highest-risk, highest-value accounts rather than treating everyone the same. Even modest accuracy improvements translate into meaningful retained revenue when applied across a large base.

The feature importance analysis shows which behaviours drive churn, giving the business levers to act on (for example contract type, support contacts, and engagement) rather than just a score.

Recommended next step: validate the model on real historical data, then pilot a targeted retention campaign against the top decile of predicted-risk customers and measure the change in retention.

---
Generated locally (LLM API unavailable: ResourceExhausted).
