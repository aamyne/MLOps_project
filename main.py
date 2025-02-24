"""Main module for running the ML pipeline with enhanced MLflow tracking."""

import argparse
import sys
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc
from data_processing import (
    prepare_data as process_data,
)  # Assumed data processing module
from model_persistence import save_model  # Assumed persistence module


def run_full_pipeline(train_file: str, test_file: str) -> None:
    """Execute the complete ML pipeline with enhanced MLflow tracking."""
    print("Running full pipeline...")

    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Adjust URI as needed
    experiment_name = "churn_prediction"  # Customize experiment name
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name="Enhanced_Pipeline"):
        try:
            # Log input files
            mlflow.log_param("train_file", train_file)
            mlflow.log_param("test_file", test_file)

            # Data preparation
            print("🔹 Preparing data...")
            X_train, X_test, y_train, y_test = process_data(train_file, test_file)
            print("🔹 Data preparation complete")

            # Create validation set from training data
            X_val = X_train.sample(frac=0.2, random_state=42)
            y_val = y_train[X_val.index]
            X_train_final = X_train.drop(X_val.index)
            y_train_final = y_train[X_train_final.index]

            # Define and log model parameters
            params = {
                "n_estimators": 50,
                "random_state": 42,
                "estimator": DecisionTreeClassifier(max_depth=6),
            }
            mlflow.log_params(
                {
                    "n_estimators": params["n_estimators"],
                    "random_state": params["random_state"],
                    "max_depth": params["estimator"].max_depth,
                }
            )
            mlflow.log_param("model_version", "1.0.0")

            # Train the model
            print("🔹 Training model...")
            model = BaggingClassifier(**params)
            model.fit(X_train_final, y_train_final)
            print("🔹 Model training complete")

            # **Feature Importance Plot**
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
                estimators = model.estimators_[:k]
                pred_proba = np.mean(
                    [est.predict_proba(X_val) for est in estimators], axis=0
                )
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
                range(1, params["n_estimators"] + 1),
                val_loglosses,
                label="Validation Log Loss",
            )
            plt.xlabel("Number of Estimators")
            plt.ylabel("Log Loss")
            plt.title("Validation Log Loss vs. Number of Estimators")
            plt.legend()
            plt.savefig("logloss_vs_estimators.png")
            plt.close()
            mlflow.log_artifact("logloss_vs_estimators.png")

            # Evaluate the model
            print("🔹 Evaluating model...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Log evaluation metrics
            metrics = {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_log_loss": log_loss(y_test, y_pred_proba),
            }
            mlflow.log_metrics(metrics)
            print(f"🔹 Evaluation metrics: {metrics}")

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

            # Save the model
            save_model(model, "model.joblib")
            print("🔹 Model saved successfully")

        except Exception as e:
            print(f"❌ Error in pipeline: {str(e)}")
            raise e


def main() -> None:
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Machine Learning Pipeline with Enhanced MLflow Tracking"
    )
    parser.add_argument(
        "action",
        type=str,
        nargs="?",
        default="all",
        help="Action to perform: 'all' to run the complete pipeline.",
    )
    args = parser.parse_args()

    train_file = "churn-bigml-80.csv"  # Replace with your training file
    test_file = "churn-bigml-20.csv"  # Replace with your test file

    try:
        if args.action == "all":
            run_full_pipeline(train_file, test_file)
        else:
            print("\n❌ Invalid action! Use 'all' to run the complete pipeline.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
