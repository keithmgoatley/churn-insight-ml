"""Generate an executive summary of model results.

Uses the Gemini API when quota is available; otherwise falls back to a
templated summary built from the metrics so the pipeline always completes.
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()


def _template_summary(results: dict) -> str:
    best = max(results, key=lambda k: results[k]["roc_auc"])
    auc = results[best]["roc_auc"]
    return (
        f"The stronger model in this comparison was **{best}**, with a "
        f"ROC-AUC of {auc}. ROC-AUC measures how well the model separates "
        f"customers who churn from those who stay: 0.5 is a coin flip and "
        f"1.0 is perfect, so {auc} indicates useful, better-than-random "
        f"predictive signal.\n\n"
        f"Commercially, a model at this level lets the business rank "
        f"customers by churn risk and focus retention spend on the highest-"
        f"risk, highest-value accounts rather than treating everyone the "
        f"same. Even modest accuracy improvements translate into meaningful "
        f"retained revenue when applied across a large base.\n\n"
        f"The feature importance analysis shows which behaviours drive "
        f"churn, giving the business levers to act on (for example contract "
        f"type, support contacts, and engagement) rather than just a score.\n\n"
        f"Recommended next step: validate the model on real historical data, "
        f"then pilot a targeted retention campaign against the top decile of "
        f"predicted-risk customers and measure the change in retention."
    )


def summarise() -> str:
    with open("outputs/model_results.json") as f:
        results = json.load(f)

    summary = None
    try:
        import google.generativeai as genai

        prompt = (
            "You are a data science communicator. In 4 short paragraphs, "
            "explain these churn model results to a non-technical business "
            "audience: what ROC-AUC means, which model performed better and "
            "why that matters commercially, the business value of predicting "
            "churn, and one recommended next step. Under 250 words.\n\n"
            f"Model results (JSON):\n{json.dumps(results, indent=2)}"
        )
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash")
        summary = model.generate_content(prompt).text
        source = "Generated with Google Gemini."
    except Exception as e:
        summary = _template_summary(results)
        source = f"Generated locally (LLM API unavailable: {type(e).__name__})."

    with open("outputs/executive_summary.md", "w") as f:
        f.write("# Churn Model Executive Summary\n\n" + summary + "\n\n---\n" + source + "\n")
    return summary


if __name__ == "__main__":
    print(summarise())
    print("\nSaved to outputs/executive_summary.md")
