import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score, classification_report

print("=" * 55)
print("HADRES — Phase 5: Adversarial Evasion Attack")
print("=" * 55)

# ─────────────────────────────────────────────
# STEP 1 — Load dataset and features
# ─────────────────────────────────────────────
print("\n[1] Loading dataset and top features...")
df = pd.read_csv("data/processed/cleaned_dataset.csv", low_memory=False)
with open("data/processed/top_features.json", "r") as f:
    top_features = json.load(f)

available = [f for f in top_features if f in df.columns]
X = df[available]
y = df['label_binary']

X_attack = X[y == 1].copy()
y_attack = y[y == 1].copy()
print(f"    Attack samples to perturb: {len(X_attack):,}")

# ─────────────────────────────────────────────
# STEP 2 — Generate adversarial samples
# ─────────────────────────────────────────────
print("\n[2] Generating adversarial samples...")
print("    Strategy: feature perturbation (add small noise to fool ML)")

np.random.seed(42)
X_adversarial = X_attack.copy()

for col in available:
    std = X_attack[col].std()
    noise = np.random.normal(0, std * 0.1, size=len(X_attack))
    X_adversarial[col] = X_attack[col] + noise

X_adversarial = X_adversarial.clip(lower=0)
print(f"    Generated {len(X_adversarial):,} adversarial samples")

# ─────────────────────────────────────────────
# STEP 3 — Test each model on adversarial data
# ─────────────────────────────────────────────
print("\n[3] Testing models on adversarial traffic...")

models = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "Naive Bayes":   joblib.load("models/naive_bayes.pkl"),
    "XGBoost":       joblib.load("models/xgboost.pkl")
}

clean_f1 = {
    "Random Forest": 0.9972,
    "Decision Tree": 0.9967,
    "Naive Bayes":   0.3431,
    "XGBoost":       0.9964
}

adv_results = {}

for name, model in models.items():
    y_pred = model.predict(X_adversarial)
    f1_adv = round(f1_score(y_attack, y_pred), 4)
    f1_clean = clean_f1[name]
    drop = round(f1_clean - f1_adv, 4)

    beta = 1.0
    if f1_clean > 0 and f1_adv > 0:
        ars = round((1 + beta**2) * (f1_clean * f1_adv) / (beta**2 * f1_clean + f1_adv), 4)
    else:
        ars = 0.0

    adv_results[name] = {
        "f1_clean": f1_clean,
        "f1_adversarial": f1_adv,
        "f1_drop": drop,
        "ars": ars
    }

    print(f"\n    {name}:")
    print(f"      F1 (clean traffic)      : {f1_clean}")
    print(f"      F1 (adversarial traffic): {f1_adv}")
    print(f"      F1 drop                 : {drop}")
    print(f"      ARS score               : {ars}")

# ─────────────────────────────────────────────
# STEP 4 — Save adversarial results
# ─────────────────────────────────────────────
print("\n[4] Saving adversarial results...")
with open("results/adversarial_results.json", "w") as f:
    json.dump(adv_results, f, indent=2)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Adversarial Attack Results — Summary")
print("=" * 55)
print(f"{'Model':<20} {'F1 Clean':>10} {'F1 Adv':>10} {'Drop':>10} {'ARS':>10}")
print("-" * 55)
for name, r in adv_results.items():
    print(f"{name:<20} {r['f1_clean']:>10} {r['f1_adversarial']:>10} {r['f1_drop']:>10} {r['ars']:>10}")
print("=" * 55)
print("\nPhase 5 Step 1 complete — adversarial results saved!")
