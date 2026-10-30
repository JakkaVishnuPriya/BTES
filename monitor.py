# -*- coding: utf-8 -*-
"""
BTES Tree Extraction - Live Training Monitor
Lightweight Flask GUI, runs in a background thread on CFG.MONITOR_PORT,
polls itself every 2s. Call start_monitor() once, update_metrics() per epoch.

@author: saurav_kumar
"""

import threading
from flask import Flask, jsonify, Response

from config import CFG


_history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "iou": [],
    "lr": []
}

_lock = threading.Lock()

_app = Flask(__name__)


_HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>BTES Tree Extraction - Training Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body { font-family: Arial, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:20px; }
h1 { color:#38bdf8; margin-bottom:4px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
.card { background:#1e293b; border-radius:10px; padding:15px; }
canvas { background:#0f172a; border-radius:8px; }
.stat { font-size:14px; color:#94a3b8; margin-top:0; }
</style>
</head>
<body>
<h1>BTES Tree Extraction &mdash; Live Training Monitor</h1>
<p class="stat">Auto-refreshes every 2 seconds &middot; Model: MODEL_NAME_PLACEHOLDER</p>
<div class="grid">
  <div class="card"><canvas id="lossChart"></canvas></div>
  <div class="card"><canvas id="iouChart"></canvas></div>
  <div class="card"><canvas id="prChart"></canvas></div>
  <div class="card"><canvas id="lrChart"></canvas></div>
</div>
<script>
const ctxLoss = document.getElementById('lossChart').getContext('2d');
const ctxIou = document.getElementById('iouChart').getContext('2d');
const ctxPr = document.getElementById('prChart').getContext('2d');
const ctxLr = document.getElementById('lrChart').getContext('2d');

function mkChart(ctx, labels, datasets, title) {
  return new Chart(ctx, {
    type: 'line',
    data: { labels: labels, datasets: datasets },
    options: {
      responsive: true,
      plugins: {
        title: { display: true, text: title, color: '#e2e8f0' },
        legend: { labels: { color: '#e2e8f0' } }
      },
      scales: {
        x: { ticks: { color: '#94a3b8' } },
        y: { ticks: { color: '#94a3b8' } }
      }
    }
  });
}

let lossChart, iouChart, prChart, lrChart;

async function refresh() {
  const res = await fetch('/api/history');
  const h = await res.json();

  if (!lossChart) {
    lossChart = mkChart(ctxLoss, h.epoch, [
      { label: 'Train Loss', data: h.train_loss, borderColor: '#38bdf8', tension: 0.3 },
      { label: 'Val Loss', data: h.val_loss, borderColor: '#f97316', tension: 0.3 }
    ], 'Loss');

    iouChart = mkChart(ctxIou, h.epoch, [
      { label: 'IoU', data: h.iou, borderColor: '#22c55e', tension: 0.3 },
      { label: 'F1', data: h.f1, borderColor: '#eab308', tension: 0.3 }
    ], 'IoU / F1');

    prChart = mkChart(ctxPr, h.epoch, [
      { label: 'Precision', data: h.precision, borderColor: '#a78bfa', tension: 0.3 },
      { label: 'Recall', data: h.recall, borderColor: '#f472b6', tension: 0.3 }
    ], 'Precision / Recall');

    lrChart = mkChart(ctxLr, h.epoch, [
      { label: 'Learning Rate', data: h.lr, borderColor: '#facc15', tension: 0.3 }
    ], 'Learning Rate');
  } else {
    [lossChart, iouChart, prChart, lrChart].forEach(c => c.data.labels = h.epoch);
    lossChart.data.datasets[0].data = h.train_loss;
    lossChart.data.datasets[1].data = h.val_loss;
    iouChart.data.datasets[0].data = h.iou;
    iouChart.data.datasets[1].data = h.f1;
    prChart.data.datasets[0].data = h.precision;
    prChart.data.datasets[1].data = h.recall;
    lrChart.data.datasets[0].data = h.lr;
    [lossChart, iouChart, prChart, lrChart].forEach(c => c.update());
  }
}

setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
""".replace("MODEL_NAME_PLACEHOLDER", CFG.MODEL_NAME)


@_app.route("/")
def index():

    return Response(_HTML_PAGE, mimetype="text/html")


@_app.route("/api/history")
def api_history():

    with _lock:

        return jsonify(_history)


def update_metrics(epoch, train_loss, val_loss, acc, precision, recall, f1, iou, lr):

    with _lock:

        _history["epoch"].append(epoch)
        _history["train_loss"].append(round(float(train_loss), 5))
        _history["val_loss"].append(round(float(val_loss), 5))
        _history["accuracy"].append(round(float(acc), 5))
        _history["precision"].append(round(float(precision), 5))
        _history["recall"].append(round(float(recall), 5))
        _history["f1"].append(round(float(f1), 5))
        _history["iou"].append(round(float(iou), 5))
        _history["lr"].append(float(lr))


def start_monitor():

    def _run():

        _app.run(host="0.0.0.0", port=CFG.MONITOR_PORT, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)

    t.start()

    print(f"\nLive Training Monitor running at: http://localhost:{CFG.MONITOR_PORT}")

    return t
