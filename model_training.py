import matplotlib.pyplot as plt
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import BaggingClassifier
import pandas as pd
from typing import Union, Any


def train_model(
    X_train: Union[pd.DataFrame, Any],
    y_train: Any,
    X_val: Union[pd.DataFrame, Any],
    y_val: Any,
    params: dict,
) -> BaggingClassifier:
    # Example parameters (adjust as needed)
    params = params or {"n_estimators": 100, "max_depth": 10, "random_state": 42}

    # Train the model
    model = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=params["max_depth"]),
        n_estimators=params["n_estimators"],
        random_state=params["random_state"],
    )
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_params(params)

    # **Feature Importance Plot**
    # Compute average feature importances across all base estimators
    feature_importances = np.mean(
        [tree.feature_importances_ for tree in model.estimators_], axis=0
    )
    feature_names = (
        X_train.columns
        if isinstance(X_train, pd.DataFrame)
        else [f"feature_{i}" for i in range(len(feature_importances))]
    )

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importances)
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance Plot")
    plt.savefig("feature_importance.png")
    plt.close()
    mlflow.log_artifact("feature_importance.png")

    # **Validation Metrics vs. Number of Estimators**
    val_accuracies = []
    val_loglosses = []
    for k in range(1, params["n_estimators"] + 1):
        # Use the first k estimators to make predictions
        estimators = model.estimators_[:k]
        pred_proba = np.mean([est.predict_proba(X_val) for est in estimators], axis=0)
        pred = np.argmax(pred_proba, axis=1)
        acc = accuracy_score(y_val, pred)
        ll = log_loss(y_val, pred_proba)
        val_accuracies.append(acc)
        val_loglosses.append(ll)

    # Plot validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, params["n_estimators"] + 1),
        val_accuracies,
        label="Validation Accuracy",
    )
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy vs. Number of Estimators")
    plt.legend()
    plt.savefig("accuracy_vs_estimators.png")
    plt.close()
    mlflow.log_artifact("accuracy_vs_estimators.png")

    # Plot validation log loss
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, params["n_estimators"] + 1), val_loglosses, label="Validation Log Loss"
    )
    plt.xlabel("Number of Estimators")
    plt.ylabel("Log Loss")
    plt.title("Validation Log Loss vs. Number of Estimators")
    plt.legend()
    plt.savefig("logloss_vs_estimators.png")
    plt.close()
    mlflow.log_artifact("logloss_vs_estimators.png")

    print("🔹 Model training complete")
    return model
