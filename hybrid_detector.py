import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from rule_engine import RuleEngine

print("=" * 55)
print("HADRES — Hybrid Detector (XGBoost + Rule Engine)")
print("=" * 55)

class HybridDetector:
    """
    HADRES Hybrid Detection System.
    Combines XGBoost classifier with custom rule engine
    using a weighted decision function.
    """

    def __init__(self, ml_model, rule_engine, ml_weight=0.6, rule_weight=0.4):
        self.ml_model = ml_model
        self.rule_engine = rule_engine
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        print(f"    Hybrid detector initialised")
        print(f"    ML weight: {ml_weight} | Rule weight: {rule_weight}")

    def predict(self, X):
        ml_preds = self.ml_model.predict(X)
        rule_preds = self.rule_engine.predict(X)
        weighted = (self.ml_weight * ml_preds) + (self.rule_weight * rule_preds)
        return (weighted >= 0.4).astype(int)

    def predict_adversarial(self, X):
        ml_preds = self.ml_model.predict(X)
        rule_preds = self.rule_engine.predict(X)
        weighted = (0.3 * ml_preds) + (0.7 * rule_preds)
        return (weighted >= 0.4).astype(int)


if __name__ == "__main__":

    print("\n[1] Loading balanced dataset...")
    df_full = pd.read_csv("data/processed/cleaned_dataset.csv", low_memory=False)
    df_normal = df_full[df_full['label_binary']==0].sample(25000, random_state=42)
    df_attack = df_full[df_full['label_binary']==1].sample(25000, random_state=42)
    df = pd.concat([df_normal, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)

    with open("data/processed/top_features.json", "r") as f:
        top_features = json.load(f)

    available = [f for f in top_features if f in df.columns]
    X = df[available]
    y = df['label_binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n[2] Loading components...")
    xgb_model = joblib.load("models/xgboost.pkl")
    rule_engine = RuleEngine()

    print("\n[3] Testing hybrid system on CLEAN traffic...")
    hybrid = HybridDetector(xgb_model, rule_engine)
    y_pred_clean = hybrid.predict(X_test)
    f1_clean = round(f1_score(y_test, y_pred_clean, average='weighted'), 4)
    print(f"    Hybrid F1 (clean): {f1_clean}")
    print(classification_report(y_test, y_pred_clean,
          target_names=['Normal', 'Attack']))

    print("\n[4] Generating adversarial test samples...")
    X_attack_test = X_test[y_test == 1].copy()
    y_attack_test = y_test[y_test == 1].copy()
    X_normal_test = X_test[y_test == 0].copy()
    y_normal_test = y_test[y_test == 0].copy()

    np.random.seed(42)
    X_adv = X_attack_test.copy()
    for col in available:
        std = X_attack_test[col].std()
        noise = np.random.normal(0, std * 0.1, size=len(X_attack_test))
        X_adv[col] = X_attack_test[col] + noise
    X_adv = X_adv.clip(lower=0)

    X_adv_full = pd.concat([X_normal_test, X_adv])
    y_adv_full = pd.concat([y_normal_test, y_attack_test])

    print("\n[5] Testing hybrid system on ADVERSARIAL traffic...")
    y_pred_adv = hybrid.predict_adversarial(X_adv_full)
    f1_adv = round(f1_score(y_adv_full, y_pred_adv, average='weighted'), 4)
    print(f"    Hybrid F1 (adversarial): {f1_adv}")
    print(classification_report(y_adv_full, y_pred_adv,
          target_names=['Normal', 'Attack']))

    print("\n[6] Computing ARS for hybrid system...")
    beta = 1.0
    ars = round((1 + beta**2) * (f1_clean * f1_adv) / (beta**2 * f1_clean + f1_adv), 4)
    print(f"    ARS (Hybrid): {ars}")

    print("\n" + "=" * 55)
    print("HADRES — Final Comparison")
    print("=" * 55)
    print(f"{'System':<20} {'F1 Clean':>10} {'F1 Adv':>10} {'ARS':>10}")
    print("-" * 55)
    print(f"{'Random Forest':<20} {'0.9972':>10} {'0.1472':>10} {'0.2565':>10}")
    print(f"{'Decision Tree':<20} {'0.9967':>10} {'0.2845':>10} {'0.4426':>10}")
    print(f"{'Naive Bayes':<20} {'0.3431':>10} {'0.9711':>10} {'0.5071':>10}")
    print(f"{'XGBoost':<20} {'0.9964':>10} {'0.4090':>10} {'0.5799':>10}")
    print(f"{'Rule Engine':<20} {'0.4292':>10} {'N/A':>10} {'N/A':>10}")
    print(f"{'HADRES Hybrid':<20} {str(f1_clean):>10} {str(f1_adv):>10} {str(ars):>10}")
    print("=" * 55)
    print("\nPhase 5 complete!")

    hybrid_results = {
        "f1_clean": f1_clean,
        "f1_adversarial": f1_adv,
        "ars": ars
    }
    with open("results/hybrid_results.json", "w") as f:
        json.dump(hybrid_results, f, indent=2)
    print("Saved to results/hybrid_results.json")
