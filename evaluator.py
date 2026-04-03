import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

print("=" * 55)
print("HADRES — Phase 6: Evaluator")
print("All 4 Experiments")
print("=" * 55)

# ─────────────────────────────────────────────
# Load data and models
# ─────────────────────────────────────────────
print("\n[1] Loading data and models...")
df_full = pd.read_csv("data/processed/cleaned_dataset.csv", low_memory=False)
df_normal = df_full[df_full['label_binary']==0].sample(25000, random_state=42)
df_attack = df_full[df_full['label_binary']==1].sample(25000, random_state=42)
df = pd.concat([df_normal, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)

with open("data/processed/top_features.json") as f:
    top_features = json.load(f)

available = [c for c in top_features if c in df.columns]
X = df[available]
y = df['label_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

with open("results/model_results.json") as f:
    model_results = json.load(f)

with open("results/adversarial_results.json") as f:
    adv_results = json.load(f)

with open("results/hybrid_results.json") as f:
    hybrid_results = json.load(f)

print("    Data and results loaded")

# ─────────────────────────────────────────────
# EXPERIMENT 1 — Baseline ML comparison chart
# ─────────────────────────────────────────────
print("\n[2] Experiment 1 — Baseline ML comparison chart...")

models_names = list(model_results.keys())
f1_scores = [model_results[m]['f1'] for m in models_names]
auc_scores = [model_results[m]['auc'] for m in models_names]

x = np.arange(len(models_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='#378ADD')
bars2 = ax.bar(x + width/2, auc_scores, width, label='ROC-AUC', color='#1D9E75')

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Experiment 1 — Baseline ML Model Comparison (Clean Traffic)')
ax.set_xticks(x)
ax.set_xticklabels(models_names)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.annotate(f'{bar.get_height():.4f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.4f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("results/charts/experiment1_baseline_comparison.png", dpi=150)
plt.close()
print("    Saved experiment1_baseline_comparison.png")

# ─────────────────────────────────────────────
# EXPERIMENT 2 — Adversarial F1 drop chart
# ─────────────────────────────────────────────
print("\n[3] Experiment 2 — Adversarial F1 drop chart...")

adv_models = list(adv_results.keys())
f1_clean = [adv_results[m]['f1_clean'] for m in adv_models]
f1_adv = [adv_results[m]['f1_adversarial'] for m in adv_models]

x = np.arange(len(adv_models))
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, f1_clean, width, label='F1 Clean', color='#1D9E75')
bars2 = ax.bar(x + width/2, f1_adv, width, label='F1 Adversarial', color='#E24B4A')

ax.set_xlabel('Model')
ax.set_ylabel('F1 Score')
ax.set_title('Experiment 2 — F1 Score Drop Under Adversarial Evasion Attack')
ax.set_xticks(x)
ax.set_xticklabels(adv_models)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("results/charts/experiment2_adversarial_drop.png", dpi=150)
plt.close()
print("    Saved experiment2_adversarial_drop.png")

# ─────────────────────────────────────────────
# EXPERIMENT 3 — ARS comparison chart
# ─────────────────────────────────────────────
print("\n[4] Experiment 3 — ARS comparison chart...")

all_systems = list(adv_results.keys()) + ['HADRES Hybrid']
all_ars = [adv_results[m]['ars'] for m in adv_results.keys()] + [hybrid_results['ars']]
colors = ['#378ADD', '#378ADD', '#378ADD', '#378ADD', '#1D9E75']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(all_systems, all_ars, color=colors)
ax.set_xlabel('System')
ax.set_ylabel('ARS Score')
ax.set_title('Experiment 3 — Adversarial Robustness Score (ARS) Comparison')
ax.set_ylim(0, 1.0)
ax.axhline(y=hybrid_results['ars'], color='#1D9E75',
           linestyle='--', alpha=0.5, label=f"Hybrid ARS = {hybrid_results['ars']}")
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    ax.annotate(f'{bar.get_height():.4f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("results/charts/experiment3_ars_comparison.png", dpi=150)
plt.close()
print("    Saved experiment3_ars_comparison.png")

# ─────────────────────────────────────────────
# EXPERIMENT 4 — Latency benchmark
# ─────────────────────────────────────────────
print("\n[5] Experiment 4 — Latency benchmark...")

import time

models_loaded = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "Naive Bayes":   joblib.load("models/naive_bayes.pkl"),
    "XGBoost":       joblib.load("models/xgboost.pkl"),
}

latency_results = {}
sample_sizes = [100, 500, 1000, 5000, 10000]

for name, model in models_loaded.items():
    latencies = []
    for n in sample_sizes:
        X_sample = X_test.iloc[:n]
        start = time.time()
        model.predict(X_sample)
        end = time.time()
        latencies.append(round((end - start) * 1000, 2))
    latency_results[name] = latencies
    print(f"    {name} latency measured")

fig, ax = plt.subplots(figsize=(10, 6))
for name, latencies in latency_results.items():
    ax.plot(sample_sizes, latencies, marker='o', label=name, linewidth=2)

ax.set_xlabel('Number of packets')
ax.set_ylabel('Detection latency (ms)')
ax.set_title('Experiment 4 — Detection Latency vs Throughput')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/charts/experiment4_latency.png", dpi=150)
plt.close()
print("    Saved experiment4_latency.png")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Phase 6 — Evaluator Complete")
print("=" * 55)
print("Charts generated:")
print("  - experiment1_baseline_comparison.png")
print("  - experiment2_adversarial_drop.png")
print("  - experiment3_ars_comparison.png")
print("  - experiment4_latency.png")
print("=" * 55)
