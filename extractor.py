import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("Loading CICIDS2017 dataset...")

data_path = "data/processed/MachineLearningCVE/"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

print(f"Found {len(csv_files)} CSV files")

dfs = []
for f in csv_files:
    print(f"  Loading {f}...")
    df = pd.read_csv(os.path.join(data_path, f), encoding='utf-8', low_memory=False)
    dfs.append(df)

print("Merging all files...")
data = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(data):,}")
print(f"Total columns: {len(data.columns)}")

print("\nCleaning column names...")
data.columns = data.columns.str.strip()

print("Checking label column...")
print(data[' Label'].value_counts() if ' Label' in data.columns else data['Label'].value_counts())

label_col = ' Label' if ' Label' in data.columns else 'Label'

print("\nCleaning data...")
data = data.replace([np.inf, -np.inf], np.nan)
before = len(data)
data = data.dropna()
after = len(data)
print(f"Removed {before - after:,} rows with null/inf values")

data = data.drop_duplicates()
print(f"Rows after deduplication: {len(data):,}")

print("\nEncoding labels...")
data['label_binary'] = (data[label_col].str.strip() != 'BENIGN').astype(int)
print(f"Normal (0): {(data['label_binary']==0).sum():,}")
print(f"Attack (1): {(data['label_binary']==1).sum():,}")

print("\nSaving cleaned dataset...")
data.to_csv("data/processed/cleaned_dataset.csv", index=False)
print("Saved to data/processed/cleaned_dataset.csv")
print("\nPhase 3 - Data extraction complete!")

print("\nSelecting top 20 features...")
df_sample = pd.read_csv("data/processed/cleaned_dataset.csv", nrows=100000)
feature_cols = [c for c in df_sample.columns if c not in ['Label', ' Label', 'label_binary']]
X = df_sample[feature_cols]
y = df_sample['label_binary']

from sklearn.ensemble import RandomForestClassifier
import json
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=feature_cols)
top20 = importances.nlargest(20).index.tolist()
print("Top 20 features:")
for i, f in enumerate(top20, 1):
    print(f"  {i}. {f}")

with open("data/processed/top_features.json", "w") as f:
    json.dump(top20, f)
print("Saved to data/processed/top_features.json")
print("Phase 3 complete!")
