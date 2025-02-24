import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, log_loss
import pandas as pd
import mlflow
from typing import Any, Dict


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_log_loss": log_loss(y_test, y_pred_proba),
    }
    mlflow.log_metrics(metrics)

    # **Confusion Matrix Plot**
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # **ROC Curve Plot**
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()
    mlflow.log_artifact("roc_curve.png")

    # **Test Predictions CSV**
    pred_df = pd.DataFrame(
        {
            "true_label": y_test,
            "predicted_label": y_pred,
            "predicted_prob_0": y_pred_proba[:, 0],
            "predicted_prob_1": y_pred_proba[:, 1],
        }
    )
    pred_df.to_csv("test_predictions.csv", index=False)
    mlflow.log_artifact("test_predictions.csv")

    print(f"🔹 Evaluation metrics: {metrics}")
    return metrics
