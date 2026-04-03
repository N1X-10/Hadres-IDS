import pandas as pd
import numpy as np
import json
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 55)
print("HADRES — Phase 4: ML Model Training")
print("=" * 55)

print("\n[1] Loading cleaned dataset...")
df = pd.read_csv("data/processed/cleaned_dataset.csv", low_memory=False)
print(f"    Loaded {len(df):,} rows")

print("\n[2] Loading top 20 features...")
with open("data/processed/top_features.json", "r") as f:
    top_features = json.load(f)

available = [f for f in top_features if f in df.columns]
X = df[available]
y = df['label_binary']

print("\n[3] Splitting dataset 80/20...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"    Train: {len(X_train):,} rows")
print(f"    Test:  {len(X_test):,} rows")

os.makedirs("models", exist_ok=True)
os.makedirs("results/charts", exist_ok=True)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight='balanced', n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, class_weight='balanced'),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(
        n_estimators=100, random_state=42,
        eval_metric='logloss', verbosity=0,
        scale_pos_weight=5)
}

results = {}

for name, model in models.items():
    print(f"\n[4] Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "precision": round(report['weighted avg']['precision'], 4),
        "recall": round(report['weighted avg']['recall'], 4),
        "f1": round(report['weighted avg']['f1-score'], 4),
        "auc": round(auc, 4),
        "confusion_matrix": cm.tolist()
    }

    print(f"    Precision : {results[name]['precision']}")
    print(f"    Recall    : {results[name]['recall']}")
    print(f"    F1 Score  : {results[name]['f1']}")
    print(f"    ROC-AUC   : {results[name]['auc']}")

    model_name = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"    Saved models/{model_name}.pkl")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'{name} — Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"results/charts/confusion_matrix_{model_name}.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} — ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"results/charts/roc_curve_{model_name}.png", dpi=150)
    plt.close()
    print(f"    Saved charts")

print("\n[5] Saving results...")
with open("results/model_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 55)
print("Phase 4 Complete — Model Comparison")
print("=" * 55)
print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 55)
for name, r in results.items():
    print(f"{name:<20} {r['precision']:>10} {r['recall']:>10} {r['f1']:>10} {r['auc']:>10}")
print("=" * 55)
print("\nAll models saved to models/")
print("All charts saved to results/charts/")
print("Phase 4 complete!")
