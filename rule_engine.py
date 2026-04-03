import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("HADRES — Rule Engine")
print("=" * 55)

class RuleEngine:
    """
    Custom signature-based rule engine for network intrusion detection.
    Written from scratch in Python — not Snort.
    Rules based on known attack patterns in CICIDS2017.
    """

    def __init__(self):
        self.rules = [
            self.rule_port_scan,
            self.rule_dos_flood,
            self.rule_high_packet_rate,
            self.rule_abnormal_flow_duration,
            self.rule_large_forward_packets,
        ]
        print(f"    Rule engine initialised with {len(self.rules)} rules")

    def rule_port_scan(self, row):
        return (row.get('Flow Duration', 0) < 100 and
                row.get('Total Fwd Packets', 0) < 5 and
                row.get('Destination Port', 0) > 0)

    def rule_dos_flood(self, row):
        return row.get('Flow Packets/s', 0) > 10000

    def rule_high_packet_rate(self, row):
        return row.get('Flow Bytes/s', 0) > 500000

    def rule_abnormal_flow_duration(self, row):
        return (row.get('Flow Duration', 1) < 10 and
                row.get('Total Fwd Packets', 0) > 100)

    def rule_large_forward_packets(self, row):
        return row.get('Total Length of Fwd Packets', 0) > 100000

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            flagged = any(rule(row) for rule in self.rules)
            predictions.append(1 if flagged else 0)
        return np.array(predictions)

    def predict_proba(self, X):
        preds = self.predict(X)
        return np.column_stack([1 - preds, preds])


if __name__ == "__main__":
    from sklearn.metrics import f1_score, classification_report
    from sklearn.model_selection import train_test_split

    print("\n[1] Loading balanced dataset sample...")
    df_full = pd.read_csv("data/processed/cleaned_dataset.csv", low_memory=False)
    df_normal = df_full[df_full['label_binary']==0].sample(25000, random_state=42)
    df_attack = df_full[df_full['label_binary']==1].sample(25000, random_state=42)
    df = pd.concat([df_normal, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"    Loaded 50,000 balanced rows (25k normal + 25k attack)")

    with open("data/processed/top_features.json", "r") as f:
        top_features = json.load(f)

    available = [f for f in top_features if f in df.columns]
    X = df[available]
    y = df['label_binary']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n[2] Running rule engine on test data...")
    engine = RuleEngine()
    y_pred = engine.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\n    Rule Engine F1 Score: {round(f1, 4)}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])}")
    print("Rule engine test complete!")
