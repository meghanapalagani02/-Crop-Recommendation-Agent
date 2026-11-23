"""
Model Training and Evaluation Pipeline for Crop Recommendation
Trains and compares Random Forest, XGBoost, SVM, KNN, and MLP on the Kaggle dataset.
Generates evaluation metrics, confusion matrices, feature importances, and comparison charts.
Saves the best-performing model and supporting artifacts using Joblib.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.data_loader import (
    load_crop_data,
    preprocess_data,
    get_train_test_split,
    FEATURE_COLUMNS
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")


def create_model_suite() -> dict:
    """
    Initializes the five classification algorithms with standardized random seeds
    and scaling pipelines where appropriate.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                random_state=42
            ))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="minkowski"
            ))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                learning_rate_init=0.003,
                max_iter=600,
                random_state=42
            ))
        ])
    }
    return models


def evaluate_models(models: dict, X_train, X_test, y_train, y_test, class_names):
    """
    Trains each model on the training partition and evaluates it on the holdout test partition.
    Returns:
        metrics_df: DataFrame of summary metrics
        detailed_results: Dict of per-model reports, predictions, and confusion matrices
        best_model_name: Name of model with highest test accuracy
    """
    records = []
    detailed_results = {}
    best_acc = -1.0
    best_name = None

    print("\n" + "=" * 70)
    print("🌾 CROP RECOMMENDATION - MODEL TRAINING & BENCHMARKING")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n--> Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        print(f"    Accuracy : {acc * 100:.2f}% | Precision : {prec * 100:.2f}% | Recall : {rec * 100:.2f}% | F1-Score : {f1 * 100:.2f}%")

        records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

        detailed_results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report
        }

        if acc > best_acc:
            best_acc = acc
            best_name = name

    metrics_df = pd.DataFrame(records).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return metrics_df, detailed_results, best_name


def plot_and_save_charts(metrics_df: pd.DataFrame, detailed_results: dict, best_name: str, best_model, class_names, df_raw):
    """
    Generates and saves modern agricultural-themed charts:
      1. Model Comparison Chart
      2. Confusion Matrix Heatmap
      3. Feature Importance Chart
      4. Correlation Heatmap
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Palette
    forest_green = "#2d6a4f"
    leaf_green = "#52b788"
    pale_green = "#74c69d"
    amber = "#d4a373"
    colors = [forest_green, leaf_green, pale_green, amber]

    # --- 1. Model Comparison Multi-Metric Bar Chart ---
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(metrics_df))
    width = 0.2

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    for i, m in enumerate(metrics):
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, metrics_df[m] * 100, width, label=m, color=colors[i], edgecolor="#1b4332", linewidth=0.8)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold", rotation=0)

    ax.set_ylabel("Score Percentage (%)", fontsize=12, fontweight="bold", color="#1b4332")
    ax.set_title("Machine Learning Models Benchmark Comparison (Test Split 20%)", fontsize=14, fontweight="bold", pad=15, color="#1b4332")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Model"], fontsize=11, fontweight="bold")
    ax.set_ylim(85, 103)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#2d6a4f", fontsize=10)
    plt.tight_layout()
    comp_path = os.path.join(CHARTS_DIR, "model_comparison.png")
    plt.savefig(comp_path)
    plt.close()
    print(f"Saved: {comp_path}")

    # --- 2. Confusion Matrix Heatmap for Best Model ---
    cm = np.array(detailed_results[best_name]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGn",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="#d8f3dc",
        ax=ax
    )
    ax.set_title(f"Confusion Matrix - Best Model: {best_name} (Acc: {detailed_results[best_name]['accuracy']*100:.2f}%)",
                 fontsize=15, fontweight="bold", pad=15, color="#1b4332")
    ax.set_xlabel("Predicted Crop Label", fontsize=12, fontweight="bold", color="#1b4332")
    ax.set_ylabel("Actual Ground Truth Label", fontsize=12, fontweight="bold", color="#1b4332")
    plt.xticks(rotation=45, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    cm_path = os.path.join(CHARTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved: {cm_path}")

    # --- 3. Feature Importance Chart ---
    # Extract feature importances if Random Forest or XGBoost, or compute from RF
    feat_importances = None
    if hasattr(best_model, "feature_importances_"):
        feat_importances = best_model.feature_importances_
        source_model_name = best_name
    elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps.get("model", None), "feature_importances_"):
        feat_importances = best_model.named_steps["model"].feature_importances_
        source_model_name = best_name
    else:
        # Fallback to Random Forest feature importance
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(df_raw[FEATURE_COLUMNS], df_raw["label"])
        feat_importances = rf_model.feature_importances_
        source_model_name = "Random Forest"

    feat_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": feat_importances
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"] * 100, color="#40916c", edgecolor="#1b4332", height=0.6)
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.1f}%",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=9.5, fontweight="bold", color="#1b4332")
    ax.set_xlabel("Relative Importance (%)", fontsize=11, fontweight="bold", color="#1b4332")
    ax.set_title(f"Agricultural Feature Importance ({source_model_name})", fontsize=13, fontweight="bold", pad=12, color="#1b4332")
    ax.set_xlim(0, max(feat_df["Importance"] * 100) * 1.15)
    plt.tight_layout()
    fi_path = os.path.join(CHARTS_DIR, "feature_importance.png")
    plt.savefig(fi_path)
    plt.close()
    print(f"Saved: {fi_path}")

    # --- 4. Feature Correlation Matrix ---
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    corr = df_raw[FEATURE_COLUMNS].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="#ffffff",
        ax=ax
    )
    ax.set_title("Environmental & Soil Nutrient Correlation Matrix", fontsize=13, fontweight="bold", pad=12, color="#1b4332")
    plt.tight_layout()
    corr_path = os.path.join(CHARTS_DIR, "correlation_matrix.png")
    plt.savefig(corr_path)
    plt.close()
    print(f"Saved: {corr_path}")


def train_and_save():
    """
    End-to-end pipeline: loads data, trains 5 models, saves evaluation metrics,
    creates charts, and serializes the best performing model.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # 1. Load Data
    df = load_crop_data()
    X, y, encoder = preprocess_data(df)
    class_names = [str(c) for c in encoder.classes_]

    # 2. Consistent Stratified Split
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.20, random_state=42)

    # 3. Model Suite
    models = create_model_suite()

    # 4. Train and Evaluate
    metrics_df, detailed_results, best_name = evaluate_models(
        models, X_train, X_test, y_train, y_test, class_names
    )

    best_model = models[best_name]

    print("\n" + "=" * 70)
    print("🏆 MODEL BENCHMARK LEADERBOARD")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print(f"\n🌟 Best Performing Model: {best_name} (Accuracy: {metrics_df.iloc[0]['Accuracy']*100:.2f}%)")

    # 5. Save Leaderboard CSV & Detailed JSON
    csv_path = os.path.join(MODELS_DIR, "model_comparison_results.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics table: {csv_path}")

    json_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(json_path, "w") as f:
        json.dump({
            "best_model_name": best_name,
            "leaderboard": metrics_df.to_dict(orient="records"),
            "detailed_results": detailed_results,
            "classes": class_names
        }, f, indent=4)
    print(f"Saved detailed metrics JSON: {json_path}")

    # 6. Generate & Save Charts
    plot_and_save_charts(metrics_df, detailed_results, best_name, best_model, class_names, df)

    # 7. Serialize Artifacts
    best_model_path = os.path.join(MODELS_DIR, "best_crop_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"Saved Best Model: {best_model_path}")

    encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    joblib.dump(encoder, encoder_path)
    print(f"Saved Label Encoder: {encoder_path}")

    all_models_path = os.path.join(MODELS_DIR, "all_trained_models.joblib")
    joblib.dump(models, all_models_path)
    print(f"Saved All Trained Models: {all_models_path}")

    print("\n✅ Phase 1 Machine Learning Pipeline Completed Successfully!")
    return best_model, encoder, metrics_df


if __name__ == "__main__":
    train_and_save()
