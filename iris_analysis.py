"""
BDA Lab 8 - Full Data Analytics Revision
Option 1: Predicting Iris Flower Species

Pipeline:
  1. Load data
  2. Exploratory Data Analysis (EDA)
  3. Preprocessing (train/test split, scaling)
  4. Train multiple classification models
  5. Evaluate & compare models (accuracy, CV, confusion matrix, classification report)
  6. Feature importance
  7. Save all plots to ./outputs for use in the video presentation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("=" * 70)
print("STEP 1: DATA LOADED")
print("=" * 70)
print(df.head(), "\n")
print("Shape:", df.shape)

# ---------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nInfo:")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe())

print("\nClass balance:")
print(df["species"].value_counts())

feature_cols = iris.feature_names

# Pairplot
pairplot = sns.pairplot(df, vars=feature_cols, hue="species", corner=True)
pairplot.fig.suptitle("Iris Features Pairplot by Species", y=1.02)
pairplot.savefig(os.path.join(OUT_DIR, "01_pairplot.png"), bbox_inches="tight", dpi=150)
plt.close(pairplot.fig)
print(f"Saved: {os.path.join(OUT_DIR, '01_pairplot.png')}")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[feature_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
save_fig(fig, "02_correlation_heatmap.png")

# Boxplots per feature by species
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), feature_cols):
    sns.boxplot(data=df, x="species", y=col, ax=ax, hue="species", legend=False)
    ax.set_title(col)
fig.suptitle("Feature Distributions by Species")
save_fig(fig, "03_boxplots_by_species.png")

# Class balance bar chart
fig, ax = plt.subplots(figsize=(5, 4))
df["species"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_title("Class Balance")
ax.set_ylabel("Count")
save_fig(fig, "04_class_balance.png")

# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: PREPROCESSING")
print("=" * 70)

X = df[feature_cols]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. TRAIN MULTIPLE CLASSIFICATION MODELS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: MODEL TRAINING")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    fitted_models[name] = model

    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    results.append(
        {
            "Model": name,
            "Test Accuracy": test_acc,
            "CV Mean Accuracy": cv_scores.mean(),
            "CV Std": cv_scores.std(),
        }
    )

    print(f"\n--- {name} ---")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

# Bar chart comparing model accuracy
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=results_df, x="Test Accuracy", y="Model", hue="Model", legend=False, palette="viridis")
ax.set_xlim(0.8, 1.01)
ax.set_title("Test Accuracy by Model")
save_fig(fig, "05_model_comparison.png")

# ---------------------------------------------------------------------------
# 5. BEST MODEL - DETAILED EVALUATION
# ---------------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
print(f"\nBest model: {best_model_name}")

y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best_model_name}")
save_fig(fig, "06_confusion_matrix_best_model.png")

# ---------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------------------------
rf = fitted_models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(7, 4))
importances.plot(kind="barh", ax=ax, color="#55A868")
ax.invert_yaxis()
ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance")
save_fig(fig, "07_feature_importance.png")

print("\nFeature importance (Random Forest):")
print(importances)

print("\n" + "=" * 70)
print("PIPELINE COMPLETE. All figures and model_comparison.csv saved to:")
print(OUT_DIR)
print("=" * 70)
