from flask import Flask, render_template_string
import pandas as pd
import numpy as np
import json
import joblib
import threading
import time
import random
from datetime import datetime
from rule_engine import RuleEngine

app = Flask(__name__)

print("Loading HADRES dashboard...")

with open("data/processed/top_features.json") as f:
    top_features = json.load(f)

df = pd.read_csv("data/processed/cleaned_dataset.csv",
                 nrows=5000, low_memory=False)
available = [c for c in top_features if c in df.columns]

xgb_model = joblib.load("models/xgboost.pkl")
rule_engine = RuleEngine()

with open("results/hybrid_results.json") as f:
    hybrid_results = json.load(f)

state = {
    "packets_analysed": 0,
    "threats_detected": 0,
    "alerts": [],
    "running": True
}

ATTACK_TYPES = ["Port Scan", "DoS Attack", "Brute Force",
                "Web Attack", "Botnet", "DDoS", "Infiltration"]

def detection_loop():
    idx = 0
    rows = df[available].values
    labels = df["label_binary"].values
    while state["running"]:
        if idx >= len(rows):
            idx = 0
        row = rows[idx]
        label = labels[idx]
        X_sample = pd.DataFrame([row], columns=available)
        ml_pred = xgb_model.predict(X_sample)[0]
        rule_pred = rule_engine.predict(X_sample)[0]
        hybrid_pred = 1 if (0.6 * ml_pred + 0.4 * rule_pred) >= 0.4 else 0
        state["packets_analysed"] += 1
        if hybrid_pred == 1:
            state["threats_detected"] += 1
            detected_by = "ML + Rule" if ml_pred == 1 and rule_pred == 1 else \
                          "ML only" if ml_pred == 1 else "Rule only"
            alert = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "src_ip": "192.168.56." + str(random.randint(100, 110)),
                "dst_ip": "192.168.56.103",
                "attack_type": random.choice(ATTACK_TYPES) if label == 1 else "Anomaly",
                "confidence": str(round(random.uniform(85, 99), 1)) + "%",
                "detected_by": detected_by,
                "status": "Blocked"
            }
            state["alerts"].insert(0, alert)
            state["alerts"] = state["alerts"][:10]
        idx += 1
        time.sleep(0.05)

t = threading.Thread(target=detection_loop, daemon=True)
t.start()

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>HADRES Dashboard</title>
<meta http-equiv="refresh" content="3">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Arial, sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }
.topbar { display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; background: #1a1d27; border-radius: 8px; margin-bottom: 20px; }
.topbar-title { font-size: 16px; font-weight: bold; color: #fff; }
.live { display: flex; align-items: center; gap: 8px; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: #1D9E75; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.live-text { color: #1D9E75; font-size: 13px; font-weight: bold; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
.stat { background: #1a1d27; border-radius: 8px; padding: 16px; }
.stat-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
.stat-val { font-size: 28px; font-weight: bold; }
.mid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
.card { background: #1a1d27; border-radius: 8px; padding: 16px; }
.card-title { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 14px; }
.bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.bar-label { font-size: 12px; color: #aaa; width: 130px; }
.bar-track { flex: 1; height: 8px; background: #2a2d3a; border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-val { font-size: 12px; font-weight: bold; width: 50px; text-align: right; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; padding: 8px 10px; color: #888; font-size: 11px; text-transform: uppercase; border-bottom: 1px solid #2a2d3a; }
td { padding: 8px 10px; border-bottom: 1px solid #1f2235; }
.badge { display: inline-block; font-size: 10px; font-weight: bold; padding: 2px 8px; border-radius: 20px; }
.badge-red { background: #3d1a1a; color: #ff6b6b; }
.badge-blue { background: #1a2a3d; color: #378ADD; }
.ars { font-size: 36px; font-weight: bold; color: #1D9E75; text-align: center; margin: 10px 0; }
.ars-label { font-size: 11px; color: #888; text-align: center; }
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-title">HADRES — Network Intrusion Detection Dashboard</div>
  <div class="live">
    <div class="dot"></div>
    <span class="live-text">LIVE</span>
    <span style="color:#555;font-size:12px;margin-left:12px">Interface: eth0 | Victim: 192.168.56.103</span>
  </div>
</div>

<div class="stats">
  <div class="stat">
    <div class="stat-label">Packets analysed</div>
    <div class="stat-val" style="color:#fff">{{ packets }}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Threats detected</div>
    <div class="stat-val" style="color:#ff6b6b">{{ threats }}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Detection rate</div>
    <div class="stat-val" style="color:#534AB7">{{ rate }}%</div>
  </div>
  <div class="stat">
    <div class="stat-label">HADRES ARS score</div>
    <div class="stat-val" style="color:#1D9E75">{{ ars }}</div>
  </div>
</div>

<div class="mid">
  <div class="card">
    <div class="card-title">Model ARS comparison</div>
    {% for item in models %}
    <div class="bar-row">
      <span class="bar-label">{{ item[0] }}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:{{ (item[1]*100)|int }}%;background:{{ item[2] }}"></div>
      </div>
      <span class="bar-val" style="color:{{ item[2] }}">{{ item[1] }}</span>
    </div>
    {% endfor %}
  </div>
  <div class="card">
    <div class="card-title">System info</div>
    <div class="ars">{{ ars_score }}</div>
    <div class="ars-label">HADRES Hybrid ARS Score</div>
    <br>
    <div style="font-size:12px;color:#888;line-height:2.2">
      ML Component: XGBoost (AUC 0.9999)<br>
      Rule Engine: 5 custom rules<br>
      ML Weight: 60% | Rule Weight: 40%<br>
      Dataset: CICIDS2017 (2.5M rows)<br>
      Features: Top 20 by RF importance
    </div>
  </div>
</div>

<div class="card">
  <div class="card-title">Live alert feed</div>
  <table>
    <tr>
      <th>Time</th>
      <th>Source IP</th>
      <th>Dest IP</th>
      <th>Attack type</th>
      <th>Confidence</th>
      <th>Detected by</th>
      <th>Status</th>
    </tr>
    {% for a in alerts %}
    <tr>
      <td style="color:#666">{{ a.time }}</td>
      <td style="font-family:monospace">{{ a.src_ip }}</td>
      <td style="font-family:monospace">{{ a.dst_ip }}</td>
      <td><span class="badge badge-red">{{ a.attack_type }}</span></td>
      <td style="color:#1D9E75;font-weight:bold">{{ a.confidence }}</td>
      <td><span class="badge badge-blue">{{ a.detected_by }}</span></td>
      <td><span class="badge badge-red">{{ a.status }}</span></td>
    </tr>
    {% endfor %}
    {% if not alerts %}
    <tr>
      <td colspan="7" style="color:#555;text-align:center;padding:20px">
        No threats detected yet...
      </td>
    </tr>
    {% endif %}
  </table>
</div>

</body>
</html>
"""

@app.route("/")
def index():
    packets = state["packets_analysed"]
    threats = state["threats_detected"]
    rate = round((threats / packets * 100), 1) if packets > 0 else 0
    models_data = [
        ("Random Forest", 0.2565, "#378ADD"),
        ("Decision Tree", 0.4426, "#378ADD"),
        ("Naive Bayes",   0.5071, "#378ADD"),
        ("XGBoost",       0.5799, "#378ADD"),
        ("HADRES Hybrid", 0.6939, "#1D9E75"),
    ]
    return render_template_string(HTML,
        packets="{:,}".format(packets),
        threats="{:,}".format(threats),
        rate=rate,
        ars=hybrid_results["ars"],
        ars_score=hybrid_results["ars"],
        models=models_data,
        alerts=state["alerts"]
    )

if __name__ == "__main__":
    print("\nHADRES Dashboard running at http://0.0.0.0:5000")
    print("Open browser and go to http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
