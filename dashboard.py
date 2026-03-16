# dashboard.py  —  single-file Flask dashboard for result.db
# Usage:  python dashboard.py   →   open http://localhost:5050

import os, sqlite3
import numpy as np
from collections import defaultdict
from flask import Flask, jsonify, request, Response

# ── Config ────────────────────────────────────────────────────────────────────
# Resolve RESULT_DB relative to this script file so the dashboard works
# even when launched from a different current working directory.
BASE_DIR = os.path.dirname(__file__)
RESULT_DB = os.path.join(BASE_DIR, 'results', 'result.db')
PORT      = 5050

app = Flask(__name__)

# ── DB helpers ────────────────────────────────────────────────────────────────

def _conn():
    c = sqlite3.connect(RESULT_DB)
    c.row_factory = sqlite3.Row
    return c

# ── API routes ────────────────────────────────────────────────────────────────

@app.route('/api/options')
def api_options():
    if not os.path.exists(RESULT_DB):
        return jsonify({'error': f'Database not found: {RESULT_DB}'}), 404
    with _conn() as c:
        models    = [r[0] for r in c.execute("SELECT DISTINCT model      FROM metrics ORDER BY model")]
        encodings = [r[0] for r in c.execute("SELECT DISTINCT encoding   FROM metrics ORDER BY encoding")]
        ns        = [r[0] for r in c.execute("SELECT DISTINCT n_features FROM metrics ORDER BY n_features")]
    return jsonify({'models': models, 'encodings': encodings, 'n_features': ns})

@app.route('/api/data')
def api_data():
    if not os.path.exists(RESULT_DB):
        return jsonify({'error': f'Database not found: {RESULT_DB}'}), 404
    models    = request.args.getlist('models')
    encodings = request.args.getlist('encodings')
    ns        = [int(x) for x in request.args.getlist('n_features')]
    if not models or not encodings or not ns:
        return jsonify([])
    pm = ','.join('?' * len(models))
    pe = ','.join('?' * len(encodings))
    pn = ','.join('?' * len(ns))
    with _conn() as c:
        rows = c.execute(
            f"SELECT model, encoding, n_features, mae, smape, r2, rmse, train_time_s, infer_time_s FROM metrics WHERE model IN ({pm}) AND encoding IN ({pe}) AND n_features IN ({pn})",
            models + encodings + ns
        ).fetchall()
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r['model'], r['encoding'], r['n_features'])
        for m in ['mae', 'smape', 'r2', 'rmse', 'train_time_s', 'infer_time_s']:
            if r[m] is not None:
                buckets[k][m].append(float(r[m]))
    out = []
    for (model, encoding, n_features), md in buckets.items():
        e = {'model': model, 'encoding': encoding, 'n_features': int(n_features)}
        for m, vals in md.items():
            e[f'{m}_mean'] = round(float(np.mean(vals)), 4)
            e[f'{m}_std']  = round(float(np.std(vals)),  4)
        out.append(e)
    return jsonify(out)

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Dashboard — Surgical Duration Prediction</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {
  --navy:    #1a1a2e;
  --blue:    #16213e;
  --accent:  #4361ee;
  --accent2: #7209b7;
  --bg:      #f0f2f5;
  --card:    #ffffff;
  --text:    #2d3748;
  --muted:   #718096;
  --border:  #e2e8f0;
  --sidebar: 300px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); display: flex; flex-direction: column; min-height: 100vh; }

/* ── Header ── */
header {
  background: var(--navy);
  color: white;
  padding: 14px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 2px 12px rgba(0,0,0,0.35);
}
header h1 { font-size: 1.15rem; font-weight: 600; letter-spacing: 0.4px; }
.db-badge { font-size: 0.75rem; padding: 5px 12px; border-radius: 20px; background: rgba(255,255,255,0.1); transition: all 0.3s; }
.db-badge.ok  { background: rgba(72,187,120,0.25); color: #9ae6b4; }
.db-badge.err { background: rgba(252,129,129,0.25); color: #feb2b2; }

/* ── Layout ── */
.layout { display: flex; flex: 1; overflow: hidden; height: calc(100vh - 52px); }

/* ── Sidebar ── */
aside {
  width: var(--sidebar);
  min-width: var(--sidebar);
  background: var(--blue);
  color: #e2e8f0;
  padding: 16px 14px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
aside::-webkit-scrollbar { width: 4px; }
aside::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }

.filter-card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 10px;
  padding: 13px;
}
.filter-card h3 {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.1px;
  color: #90cdf4;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.toggle-btns { display: flex; gap: 4px; }
.toggle-btns button {
  font-size: 0.62rem;
  padding: 2px 7px;
  border: 1px solid rgba(144,205,244,0.35);
  background: transparent;
  color: #90cdf4;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.15s;
}
.toggle-btns button:hover { background: rgba(144,205,244,0.15); }

.checkbox-list { display: flex; flex-direction: column; gap: 4px; max-height: 200px; overflow-y: auto; padding-right: 2px; }
.checkbox-list::-webkit-scrollbar { width: 3px; }
.checkbox-list::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }

.cb-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 7px;
  border-radius: 7px;
  cursor: pointer;
  transition: background 0.15s;
  user-select: none;
}
.cb-item:hover { background: rgba(255,255,255,0.08); }
.cb-item input { width: 14px; height: 14px; accent-color: var(--accent); cursor: pointer; flex-shrink: 0; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 4px rgba(0,0,0,0.3); }
.cb-item span.lbl { font-size: 0.8rem; }

/* ── Group-by ── */
.radio-group { display: flex; flex-direction: column; gap: 6px; }
.rb-item {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 7px 9px;
  border-radius: 8px;
  cursor: pointer;
  border: 1px solid rgba(255,255,255,0.09);
  font-size: 0.82rem;
  transition: all 0.15s;
}
.rb-item:hover { background: rgba(255,255,255,0.07); }
.rb-item.active { background: rgba(67,97,238,0.22); border-color: rgba(67,97,238,0.5); color: #a5b4fc; }
.rb-item input { accent-color: var(--accent); cursor: pointer; }

/* ── Update button ── */
.btn-update {
  width: 100%;
  padding: 13px;
  background: linear-gradient(135deg, #4361ee 0%, #7209b7 100%);
  color: white;
  border: none;
  border-radius: 10px;
  font-size: 0.9rem;
  font-weight: 700;
  cursor: pointer;
  letter-spacing: 0.3px;
  transition: opacity 0.2s, transform 0.1s, box-shadow 0.2s;
  box-shadow: 0 4px 14px rgba(67,97,238,0.4);
  margin-top: 4px;
}
.btn-update:hover  { opacity: 0.92; box-shadow: 0 6px 18px rgba(67,97,238,0.55); }
.btn-update:active { transform: scale(0.98); }

/* ── Main ── */
main { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; }
main::-webkit-scrollbar { width: 6px; }
main::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 4px; }

.charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

.chart-card {
  background: var(--card);
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 1px 8px rgba(0,0,0,0.07);
  border: 1px solid var(--border);
  min-height: 420px;
  display: flex;
  flex-direction: column;
}
.chart-div { flex: 1; width: 100%; min-height: 380px; }

/* ── Table section ── */
.table-section {
  background: var(--card);
  border-radius: 14px;
  padding: 20px;
  box-shadow: 0 1px 8px rgba(0,0,0,0.07);
  border: 1px solid var(--border);
}
.table-section h2 { font-size: 0.95rem; font-weight: 700; color: var(--navy); margin-bottom: 16px; }

table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
thead th {
  background: var(--navy);
  color: #e2e8f0;
  padding: 10px 13px;
  text-align: left;
  font-weight: 600;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  white-space: nowrap;
}
thead th:first-child { border-radius: 8px 0 0 0; }
thead th:last-child  { border-radius: 0 8px 0 0; }
tbody tr { border-bottom: 1px solid var(--border); transition: background 0.12s; }
tbody tr:hover { background: #f7f8ff; }
tbody td { padding: 9px 13px; white-space: nowrap; }
.best { font-weight: 700; color: #2b6cb0; background: #ebf4ff !important; }
.rank-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px; height: 22px;
  border-radius: 50%;
  font-size: 0.72rem;
  font-weight: 700;
  background: #e2e8f0;
  color: #4a5568;
}
.rank-badge.gold   { background: #f6e05e; color: #744210; }
.rank-badge.silver { background: #e2e8f0; color: #4a5568; }
.rank-badge.bronze { background: #fbd38d; color: #7b341e; }

.model-tag {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 12px;
  font-size: 0.74rem;
  font-weight: 600;
  color: white;
}
.enc-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 0.74rem;
  font-weight: 500;
  background: #edf2ff;
  color: #3730a3;
}

/* ── Spinner / empty states ── */
.loading {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  min-height: 340px;
  color: var(--muted);
  font-size: 0.88rem;
  gap: 12px;
}
.spinner {
  width: 28px; height: 28px;
  border: 3px solid #e2e8f0;
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.75s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.empty-msg {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  min-height: 340px;
  color: var(--muted);
  font-size: 0.88rem;
  text-align: center;
  padding: 20px;
}
</style>
</head>

<body>
<header>
  <h1>🏥 Surgical Duration Prediction &mdash; ML Dashboard</h1>
  <span id="db-status" class="db-badge">Connecting&hellip;</span>
</header>

<div class="layout">
  <!-- ── Sidebar ── -->
  <aside>
    <div class="filter-card">
      <h3>Models
        <span class="toggle-btns">
          <button onclick="toggleAll('models-list', true)">All</button>
          <button onclick="toggleAll('models-list', false)">None</button>
        </span>
      </h3>
      <div id="models-list" class="checkbox-list"><div class="loading"><div class="spinner"></div></div></div>
    </div>

    <div class="filter-card">
      <h3>Encodings
        <span class="toggle-btns">
          <button onclick="toggleAll('encodings-list', true)">All</button>
          <button onclick="toggleAll('encodings-list', false)">None</button>
        </span>
      </h3>
      <div id="encodings-list" class="checkbox-list"><div class="loading"><div class="spinner"></div></div></div>
    </div>

    <div class="filter-card">
      <h3>Feature Count (n)
        <span class="toggle-btns">
          <button onclick="toggleAll('n-list', true)">All</button>
          <button onclick="toggleAll('n-list', false)">None</button>
        </span>
      </h3>
      <div id="n-list" class="checkbox-list"><div class="loading"><div class="spinner"></div></div></div>
    </div>

    <div class="filter-card">
      <h3>X-axis Grouping</h3>
      <div class="radio-group">
        <label class="rb-item active" id="rb-enc">
          <input type="radio" name="groupby" value="encoding" checked onchange="onGroupByChange()">
          Group by Encoding
        </label>
        <label class="rb-item" id="rb-n">
          <input type="radio" name="groupby" value="n" onchange="onGroupByChange()">
          Group by Feature Count
        </label>
      </div>
    </div>

    <button class="btn-update" onclick="updateCharts()">&#x1F504; Update Charts</button>
  </aside>

  <!-- ── Main ── -->
  <main>
    <div class="charts-grid">
      <div class="chart-card"><div id="chart-mae"          class="chart-div"><div class="loading"><div class="spinner"></div>Loading&hellip;</div></div></div>
      <div class="chart-card"><div id="chart-smape"        class="chart-div"><div class="loading"><div class="spinner"></div>Loading&hellip;</div></div></div>
      <div class="chart-card"><div id="chart-r2"           class="chart-div"><div class="loading"><div class="spinner"></div>Loading&hellip;</div></div></div>
      <div class="chart-card"><div id="chart-train_time_s" class="chart-div"><div class="loading"><div class="spinner"></div>Loading&hellip;</div></div></div>
    </div>

    <div class="table-section">
      <h2>&#x1F4CA; Top Configurations &mdash; sorted by MAE (current selection)</h2>
      <div id="summary-table"><div class="empty-msg">Select filters and click Update Charts.</div></div>
    </div>
  </main>
</div>

<script>
// ── Colour palettes ───────────────────────────────────────────────────────────
const MODEL_COLORS = {
  linear:       '#636EFA',
  ridge:        '#EF553B',
  lasso:        '#00CC96',
  randomforest: '#AB63FA',
  xgboost:      '#FFA15A',
  mlp:          '#19D3F3'
};
const ENC_COLORS = {
  only_structured: '#636EFA',
  label:           '#EF553B',
  count:           '#00CC96',
  tfidf:           '#AB63FA',
  clinicalbert:    '#FFA15A',
  sentencebert:    '#19D3F3'
};
const N_COLORS = {
  '0':   '#636EFA',
  '10':  '#EF553B',
  '50':  '#00CC96',
  '100': '#AB63FA',
  '200': '#FFA15A'
};

// ── Metrics config ────────────────────────────────────────────────────────────
const METRICS = [
  { key: 'mae',          label: 'MAE (minutes)',  yTitle: 'MAE (min)',    lower: true  },
  { key: 'smape',        label: 'SMAPE (%)',       yTitle: 'SMAPE (%)',    lower: true  },
  { key: 'r2',           label: 'R\u00B2',          yTitle: 'R\u00B2',      lower: false },
  { key: 'train_time_s', label: 'Train Time (s)',  yTitle: 'Seconds',      lower: true  }
];

let _nOptions = [];

// ── Helpers ───────────────────────────────────────────────────────────────────
function colorFor(map, key) { return map[key] || '#888888'; }

function toggleAll(listId, state) {
  document.querySelectorAll('#' + listId + ' input[type=checkbox]').forEach(cb => { cb.checked = state; });
}

function getChecked(listId) {
  return [...document.querySelectorAll('#' + listId + ' input[type=checkbox]:checked')].map(cb => cb.value);
}

function onGroupByChange() {
  const val = document.querySelector('input[name=groupby]:checked').value;
  document.getElementById('rb-enc').classList.toggle('active', val === 'encoding');
  document.getElementById('rb-n').classList.toggle('active', val === 'n');
}

function xLabel(row) {
  const nStr = row.n_features === 0 ? 'struct' : 'n=' + row.n_features;
  return row.encoding + '\u000A(' + nStr + ')';
}

function showSpinner(metricKey) {
  document.getElementById('chart-' + metricKey).innerHTML = '<div class="loading"><div class="spinner"></div>Loading\u2026</div>';
}

function showEmpty(metricKey, msg) {
  document.getElementById('chart-' + metricKey).innerHTML = '<div class="empty-msg">' + msg + '</div>';
}

// ── Load options on startup ───────────────────────────────────────────────────
async function loadOptions() {
  try {
    const res  = await fetch('/api/options');
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }
    const opts = await res.json();
    if (opts.error) throw new Error(opts.error);

    document.getElementById('db-status').textContent = '\u2705 ' + opts.models.length + ' models \u00B7 ' + opts.encodings.length + ' encodings';
    document.getElementById('db-status').className = 'db-badge ok';

    renderCheckboxes('models-list',    opts.models,      MODEL_COLORS);
    renderCheckboxes('encodings-list', opts.encodings,   ENC_COLORS);
    renderCheckboxes('n-list',         opts.n_features,  N_COLORS);
    _nOptions = opts.n_features;

    updateCharts();
  } catch (e) {
    console.error('loadOptions error:', e);
    document.getElementById('db-status').textContent = '\u274C DB error';
    document.getElementById('db-status').className = 'db-badge err';
    METRICS.forEach(m => showEmpty(m.key, 'Could not connect to result.db. See console for details.'));
  }
}

function renderCheckboxes(listId, items, colorMap) {
  const el = document.getElementById(listId);
  el.innerHTML = items.map(item => {
    const col   = colorFor(colorMap, String(item));
    const label = item === 0 ? 'struct (no text)' : String(item);
    return '<label class="cb-item"><input type="checkbox" value="' + item + '" checked><span class="dot" style="background:' + col + '"></span><span class="lbl">' + label + '</span></label>';
  }).join('');
}

// ── Fetch data and render ─────────────────────────────────────────────────────
async function updateCharts() {
  const models    = getChecked('models-list');
  const encodings = getChecked('encodings-list');
  const nLabels   = getChecked('n-list');
  const ns        = _nOptions.filter(n => nLabels.includes(String(n)));

  if (!models.length || !encodings.length || !ns.length) {
    METRICS.forEach(m => showEmpty(m.key, 'Please select at least one option in each filter group.'));
    document.getElementById('summary-table').innerHTML = '<div class="empty-msg">No filters selected.</div>';
    return;
  }

  METRICS.forEach(m => showSpinner(m.key));

  const params = new URLSearchParams();
  models.forEach(m    => params.append('models',     m));
  encodings.forEach(e => params.append('encodings',  e));
  ns.forEach(n        => params.append('n_features', n));

  try {
    const res  = await fetch('/api/data?' + params.toString());
    const data = await res.json();
    if (!data.length) {
      METRICS.forEach(m => showEmpty(m.key, 'No results found for the selected combination.'));
      document.getElementById('summary-table').innerHTML = '<div class="empty-msg">No data.</div>';
      return;
    }
    const groupBy = document.querySelector('input[name=groupby]:checked').value;
    renderCharts(data, groupBy);
    renderTable(data);
  } catch (e) {
    console.error('updateCharts error:', e);
    METRICS.forEach(m => showEmpty(m.key, 'Error: ' + e.message));
  }
}

// ── Chart rendering ───────────────────────────────────────────────────────────
function renderCharts(data, groupBy) {
  // Sort data to control x-axis order
  if (groupBy === 'encoding') {
    data.sort((a, b) => a.encoding.localeCompare(b.encoding) || a.n_features - b.n_features);
  } else {
    data.sort((a, b) => a.n_features - b.n_features || a.encoding.localeCompare(b.encoding));
  }

  // Build unique x labels preserving sort order
  const seen = new Set();
  const xLabels = [];
  data.forEach(d => { const l = xLabel(d); if (!seen.has(l)) { seen.add(l); xLabels.push(l); } });

  // Build fast lookup: xLabel → model → row
  const lkp = {};
  data.forEach(d => {
    const xl = xLabel(d);
    if (!lkp[xl]) lkp[xl] = {};
    lkp[xl][d.model] = d;
  });

  const models = [...new Set(data.map(d => d.model))].sort();

  METRICS.forEach(metric => {
    // Find globally best value to highlight
    let bestVal = metric.lower ? Infinity : -Infinity;
    let bestXL  = null;
    data.forEach(d => {
      const v = d[metric.key + '_mean'];
      if (v == null) return;
      if ((metric.lower && v < bestVal) || (!metric.lower && v > bestVal)) { bestVal = v; bestXL = xLabel(d); }
    });
    const bestXIdx = xLabels.indexOf(bestXL);

    const traces = models.map(model => {
      const y   = xLabels.map(xl => lkp[xl]?.[model]?.[metric.key + '_mean'] ?? null);
      const err = xLabels.map(xl => lkp[xl]?.[model]?.[metric.key + '_std']  ?? null);
      return {
        type: 'bar',
        name: model,
        x: xLabels,
        y: y,
        error_y: {
          type: 'data', array: err, visible: true,
          color: 'rgba(0,0,0,0.3)', thickness: 1.4, width: 3
        },
        marker: {
          color: colorFor(MODEL_COLORS, model),
          opacity: 0.87,
          line: { color: 'rgba(255,255,255,0.55)', width: 0.8 }
        },
        hovertemplate: '<b>' + model + '</b><br>%{x}<br>' + metric.label + ': <b>%{y:.3f}</b> \u00B1 %{error_y.array:.3f}<extra></extra>'
      };
    });

    const shapes = bestXIdx >= 0 ? [{
      type: 'rect', xref: 'x', yref: 'paper',
      x0: bestXIdx - 0.48, x1: bestXIdx + 0.48,
      y0: 0, y1: 1,
      fillcolor: 'rgba(67,97,238,0.07)',
      line: { color: 'rgba(67,97,238,0.4)', width: 1.5, dash: 'dot' }
    }] : [];

    const annotations = bestXIdx >= 0 ? [{
      xref: 'x', yref: 'paper',
      x: bestXIdx, y: 1.04,
      text: '\u2B50 best',
      showarrow: false,
      font: { size: 10, color: '#4361ee' }
    }] : [];

    const layout = {
      title: { text: '<b>' + metric.label + '</b>', font: { size: 13, color: '#1a1a2e' }, x: 0.5, xanchor: 'center' },
      barmode: 'group',
      paper_bgcolor: 'white',
      plot_bgcolor: '#fafbff',
      font: { family: 'Segoe UI, system-ui, sans-serif', size: 11, color: '#2d3748' },
      xaxis: {
        tickangle: -32,
        tickfont: { size: 8.5 },
        showgrid: false,
        linecolor: '#e2e8f0',
        automargin: true
      },
      yaxis: {
        gridcolor: '#e8ecf4',
        linecolor: '#e2e8f0',
        zeroline: false,
        title: { text: metric.yTitle, font: { size: 11 } }
      },
      legend: {
        orientation: 'h',
        y: -0.30,
        x: 0.5,
        xanchor: 'center',
        font: { size: 10.5 },
        bgcolor: 'rgba(255,255,255,0.85)',
        bordercolor: '#e2e8f0',
        borderwidth: 1
      },
      margin: { t: 55, b: 140, l: 65, r: 20 },
      hoverlabel: { bgcolor: '#1a1a2e', font: { color: 'white', size: 12 }, bordercolor: '#4361ee' },
      shapes: shapes,
      annotations: annotations
    };

    Plotly.react(
      'chart-' + metric.key,
      traces,
      layout,
      { responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['sendDataToCloud', 'editInChartStudio', 'lasso2d', 'select2d'] }
    );
  });
}

// ── Table rendering ───────────────────────────────────────────────────────────
function renderTable(data) {
  if (!data.length) {
    document.getElementById('summary-table').innerHTML = '<div class="empty-msg">No data for current selection.</div>';
    return;
  }

  const sorted   = [...data].sort((a, b) => a.mae_mean - b.mae_mean);
  const minMAE   = Math.min(...sorted.map(d => d.mae_mean   ?? Infinity));
  const minSMAPE = Math.min(...sorted.map(d => d.smape_mean ?? Infinity));
  const maxR2    = Math.max(...sorted.map(d => d.r2_mean    ?? -Infinity));
  const minTrain = Math.min(...sorted.map(d => d.train_time_s_mean ?? Infinity));

  function rankBadge(i) {
    if (i === 0) return '<span class="rank-badge gold">1</span>';
    if (i === 1) return '<span class="rank-badge silver">2</span>';
    if (i === 2) return '<span class="rank-badge bronze">3</span>';
    return '<span class="rank-badge">' + (i + 1) + '</span>';
  }

  const rows = sorted.slice(0, 20).map((d, i) => {
    const nStr     = d.n_features === 0 ? 'struct' : 'n=' + d.n_features;
    const mCol     = colorFor(MODEL_COLORS, d.model);
    const eCol     = colorFor(ENC_COLORS,   d.encoding);
    const maeTd    = d.mae_mean   === minMAE   ? 'class="best"' : '';
    const smapeTd  = d.smape_mean === minSMAPE ? 'class="best"' : '';
    const r2Td     = d.r2_mean    === maxR2    ? 'class="best"' : '';
    const trainTd  = (d.train_time_s_mean ?? Infinity) === minTrain ? 'class="best"' : '';
    const maeStr   = d.mae_mean   != null ? d.mae_mean.toFixed(3)   + ' <small style="color:#999">\u00B1' + (d.mae_std ?? 0).toFixed(3)   + '</small>' : '\u2014';
    const smapeStr = d.smape_mean != null ? d.smape_mean.toFixed(3) + ' <small style="color:#999">\u00B1' + (d.smape_std ?? 0).toFixed(3) + '</small>' : '\u2014';
    const r2Str    = d.r2_mean    != null ? d.r2_mean.toFixed(4)    + ' <small style="color:#999">\u00B1' + (d.r2_std ?? 0).toFixed(4)    + '</small>' : '\u2014';
    const rmseStr  = d.rmse_mean  != null ? d.rmse_mean.toFixed(3)  : '\u2014';
    const trainStr = d.train_time_s_mean != null ? d.train_time_s_mean.toFixed(2) + 's' : '\u2014';
    const inferStr = d.infer_time_s_mean != null ? d.infer_time_s_mean.toFixed(4) + 's' : '\u2014';
    return '<tr>' +
      '<td>' + rankBadge(i) + '</td>' +
      '<td><span class="model-tag" style="background:' + mCol + '">' + d.model + '</span></td>' +
      '<td><span class="enc-tag" style="border-left:3px solid ' + eCol + '">' + d.encoding + '</span></td>' +
      '<td style="color:#4a5568;font-weight:600">' + nStr + '</td>' +
      '<td ' + maeTd   + '>' + maeStr   + '</td>' +
      '<td ' + smapeTd + '>' + smapeStr + '</td>' +
      '<td ' + r2Td    + '>' + r2Str    + '</td>' +
      '<td>' + rmseStr + '</td>' +
      '<td ' + trainTd + '>' + trainStr + '</td>' +
      '<td>' + inferStr + '</td>' +
      '</tr>';
  }).join('');

  document.getElementById('summary-table').innerHTML =
    '<table>' +
    '<thead><tr><th>#</th><th>Model</th><th>Encoding</th><th>n</th><th>MAE mean</th><th>SMAPE mean</th><th>R\u00B2 mean</th><th>RMSE mean</th><th>Train time</th><th>Infer time</th></tr></thead>' +
    '<tbody>' + rows + '</tbody>' +
    '</table>';
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', loadOptions);
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"  Dashboard → http://localhost:{PORT}")
    print(f"  DB path   → {os.path.abspath(RESULT_DB)}")
    app.run(debug=True, port=PORT, use_reloader=False)