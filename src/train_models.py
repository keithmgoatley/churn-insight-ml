"""Train and evaluate churn prediction models."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets_90d",
    "logins_per_month",
    "on_discount",
    "monthly_contract",
]


def main() -> None:
    df = pd.read_csv("data/customers.csv")
    X, y = df[FEATURES], df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=42
        ),
    }

    results = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        results[name] = {
            "roc_auc": round(auc, 4),
            "report": classification_report(
                y_test, model.predict(X_test), output_dict=True
            ),
        }
        RocCurveDisplay.from_predictions(
            y_test, proba, name=f"{name} (AUC={auc:.3f})", ax=ax
        )
    ax.set_title("ROC curves: churn model comparison")
    fig.tight_layout()
    fig.savefig("outputs/roc_curves.png", dpi=150)

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = models[best_name]
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, ax=ax2, colorbar=False)
    ax2.set_title(f"Confusion matrix: {best_name}")
    fig2.tight_layout()
    fig2.savefig("outputs/confusion_matrix.png", dpi=150)

    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.barplot(x=imp.values, y=imp.index, ax=ax3, color="#3b6ea5")
    ax3.set_title("Random forest feature importances")
    ax3.set_xlabel("Importance")
    fig3.tight_layout()
    fig3.savefig("outputs/feature_importance.png", dpi=150)

    with open("outputs/model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Best model: {best_name} | ROC-AUC: {results[best_name]['roc_auc']}")
    print("Charts and metrics saved to outputs/")


if __name__ == "__main__":
    main()
