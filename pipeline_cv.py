# =============================================================================
# pipeline_cv.py  —  Stages 01 · 02 · 03 · 04
#
# Two alternative cross-validation strategies on the same cleaned dataset:
#   Hospital CV  : leave-one-location-out  (each surgical_location = 1 fold)
#   Temporal CV  : expanding time-series split  (sklearn TimeSeriesSplit,
#                  sorted by case_date derived from scheduled_start_dttm)
#
# Prerequisites (must be complete before running this pipeline):
#   pipeline.py Stage 01  →  Clean table + fold_indices in DB_PATH
#   pipeline.py Stage 02  →  BERT caches in BERT_DIR
#
# What this pipeline adds (pipeline.py outputs are never modified):
#   DB_PATH receives two new fold-index tables: fold_hospital, fold_temporal
#   ./data/fold_encoded_hospital.db  —  encoded matrices for hospital CV
#   ./data/fold_encoded_temporal.db  —  encoded matrices for temporal CV
#   ./results_hospital/              —  metrics, predictions, logs, PDFs
#   ./results_temporal/              —  metrics, predictions, logs, PDFs
#
# NOTE on case_date: pipeline.py Stage 01 drops all datetime columns after
#   deriving time features.  Stage 01 here re-derives case_date by joining
#   scheduled_start_dttm from the raw CSV to the Clean table via case_id.
#   This is safe — the original pipeline confirmed every Clean row has a
#   valid scheduled_start_dttm before those columns were dropped.
#
# Stages 01–03 check completion and skip if already done.
# Stage 04 always asks which CV strategy and model(s) to run.
# All tuneable settings live in the CONFIG block — nowhere else.
# =============================================================================

# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG  —  edit only this block                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

# ── Paths shared with pipeline.py (read-only from this pipeline) ─────────────
RAW_CSV  = './data/casetime.csv'
DB_PATH  = './data/surgical_data.db'
BERT_DIR = './data/bert_cache'

# ── Paths exclusive to pipeline_cv.py ────────────────────────────────────────
ENCODED_DB_HOSPITAL = './data/fold_encoded_hospital.db'
ENCODED_DB_TEMPORAL = './data/fold_encoded_temporal.db'
RESULT_DB_HOSPITAL  = './results_hospital/result.db'
RESULT_DB_TEMPORAL  = './results_temporal/result.db'
LOG_DIR_HOSPITAL    = './results_hospital'
LOG_DIR_TEMPORAL    = './results_temporal'

# ── Column names (must match pipeline.py) ────────────────────────────────────
TARGET       = 'actual_casetime_minutes'
TEXT_COL     = 'scheduled_procedure'
HOSPITAL_COL = 'surgical_location'   # grouping column for leave-one-location-out CV

CATEGORICAL_FOLD_COLS = ['case_service', 'surgical_location']
IMPUTE_COLS   = ['age_at_discharge', 'avg_BMI', 'anesthetic_type']
IMPUTE_TYPES  = ['regression', 'regression', 'classification']
EXCLUDE_COLS  = ['procedure_minutes', 'procedure_time', 'induction_time', 'emergence_time', 'scheduled_duration']

# ── CV strategy settings ──────────────────────────────────────────────────────
N_SPLITS_TEMPORAL   = 5    # number of temporal folds for TimeSeriesSplit (expanding window)
FOLD_TABLE_HOSPITAL = 'fold_hospital'
FOLD_TABLE_TEMPORAL = 'fold_temporal'
MIN_VAL_FOLD_SIZE   = 30   # folds with fewer val cases than this are skipped in Stage 04

# ── Shared seed ───────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Text feature dimensionality ───────────────────────────────────────────────
FEATURES_PER_COL = [100] #[10, 50, 100, 200]

# ── Optuna general ────────────────────────────────────────────────────────────
N_TRIALS                = 50
TUNE_SET_FRACTION       = 0.25
OPTUNA_N_STARTUP_TRIALS = 10

# ── Optuna search bounds: Ridge / Lasso ──────────────────────────────────────
OPTUNA_ALPHA_LOW  = 1e-3
OPTUNA_ALPHA_HIGH = 100.0

# ── Optuna search bounds: Random Forest ──────────────────────────────────────
OPTUNA_RF_N_EST_LOW              = 100
OPTUNA_RF_N_EST_HIGH             = 500
OPTUNA_RF_MAX_DEPTH_LOW          = 3
OPTUNA_RF_MAX_DEPTH_HIGH         = 15
OPTUNA_RF_MIN_SAMPLES_SPLIT_LOW  = 2
OPTUNA_RF_MIN_SAMPLES_SPLIT_HIGH = 10
OPTUNA_RF_MIN_SAMPLES_LEAF_LOW   = 1
OPTUNA_RF_MIN_SAMPLES_LEAF_HIGH  = 5
OPTUNA_RF_MAX_FEATURES           = ['sqrt', 'log2', 0.3, 0.5, 0.7]

# ── Optuna search bounds: XGBoost ────────────────────────────────────────────
OPTUNA_XGB_N_EST_LOW        = 100
OPTUNA_XGB_N_EST_HIGH       = 500
OPTUNA_XGB_LR_LOW           = 0.01
OPTUNA_XGB_LR_HIGH          = 0.3
OPTUNA_XGB_MAX_DEPTH_LOW    = 3
OPTUNA_XGB_MAX_DEPTH_HIGH   = 8
OPTUNA_XGB_SUBSAMPLE_LOW    = 0.6
OPTUNA_XGB_SUBSAMPLE_HIGH   = 1.0
OPTUNA_XGB_COLSAMPLE_LOW    = 0.6
OPTUNA_XGB_COLSAMPLE_HIGH   = 1.0
OPTUNA_XGB_REG_ALPHA_LOW    = 1e-4
OPTUNA_XGB_REG_ALPHA_HIGH   = 10.0
OPTUNA_XGB_REG_LAMBDA_LOW   = 1e-4
OPTUNA_XGB_REG_LAMBDA_HIGH  = 10.0

# ── Optuna search bounds: MLP ────────────────────────────────────────────────
OPTUNA_MLP_N_LAYERS_LOW      = 1
OPTUNA_MLP_N_LAYERS_HIGH     = 3
OPTUNA_MLP_UNITS1_LOW        = 32
OPTUNA_MLP_UNITS1_HIGH       = 256
OPTUNA_MLP_UNITS2_LOW        = 16
OPTUNA_MLP_UNITS2_HIGH       = 128
OPTUNA_MLP_UNITS3_LOW        = 8
OPTUNA_MLP_UNITS3_HIGH       = 64
OPTUNA_MLP_DROPOUT_LOW       = 0.0
OPTUNA_MLP_DROPOUT_HIGH      = 0.5
OPTUNA_MLP_LR_LOW            = 1e-4
OPTUNA_MLP_LR_HIGH           = 1e-2
OPTUNA_MLP_WEIGHT_DECAY_LOW  = 1e-6
OPTUNA_MLP_WEIGHT_DECAY_HIGH = 1e-2
OPTUNA_MLP_ACTIVATIONS       = ['relu', 'elu', 'tanh']

# ── MLP fixed training settings ───────────────────────────────────────────────
MLP_EPOCHS_FINAL       = 200
MLP_EPOCHS_OPTUNA      = 30
MLP_BATCH_SIZE         = 512
MLP_PATIENCE_ES        = 15
MLP_PATIENCE_LR        = 5
MLP_LR_DECAY_FACTOR    = 0.5
MLP_MIN_LR             = 1e-6
MLP_CLIPNORM           = 1.0
MLP_OPTUNA_SUBSET_SIZE = 5000   # max rows per Optuna trial; None = all tune rows

# ── XGBoost training settings ─────────────────────────────────────────────────
XGB_TREE_METHOD           = 'hist'
XGB_DEVICE                = 'cuda'   # runtime-checked; falls back to 'cpu' automatically
XGB_EARLY_STOPPING_ROUNDS = 20

# ── BERT model identifiers ────────────────────────────────────────────────────
CLINICALBERT_MODEL_ID = 'emilyalsentzer/Bio_ClinicalBERT'
SENTENCEBERT_MODEL_ID = 'all-MiniLM-L6-v2'
BERT_BATCH_CLINICAL   = 32
BERT_BATCH_SENTENCE   = 64
BERT_MAX_LENGTH       = 64

# ── Encodings and models ──────────────────────────────────────────────────────
CLASSICAL_ENCODINGS_WITH_N = ['label', 'tfidf', 'count']
BERT_ENCODINGS             = ['clinicalbert', 'sentencebert']
ALL_MODELS                 = ['linear', 'ridge', 'lasso', 'randomforest', 'xgboost', 'mlp']

# ── DB table names ────────────────────────────────────────────────────────────
CLEAN_TABLE = 'Clean'

# ── BERT cache task map (auto-derived — do not edit) ─────────────────────────
S02_TASKS = {
    1: ('clinicalbert', f'clinicalbert_{TEXT_COL}.npy'),
    2: ('sentencebert', f'sentencebert_{TEXT_COL}.npy'),
}

# =============================================================================
# IMPORTS
# =============================================================================
import os, sys, sqlite3, time, warnings
import numpy as np
import pandas as pd

os.makedirs(LOG_DIR_HOSPITAL, exist_ok=True)
os.makedirs(LOG_DIR_TEMPORAL, exist_ok=True)
os.makedirs('./data',         exist_ok=True)
os.makedirs(BERT_DIR,         exist_ok=True)

warnings.filterwarnings('ignore')

# =============================================================================
# SHARED UTILITIES  (identical to pipeline.py)
# =============================================================================

class _Tee:
    def __init__(self, path):
        self._t = sys.stdout
        self._f = open(path, 'w', encoding='utf-8', buffering=1)
    def write(self, m):  self._t.write(m); self._f.write(m)
    def flush(self):     self._t.flush();  self._f.flush()
    def close(self):     sys.stdout = self._t; self._f.close()

def sep(title='', width=70, char='='):
    if title:
        print(f"\n{char*width}\n  {title}\n{char*width}")
    else:
        print(char * width)

def _print_numeric(df, cols, label=''):
    print(f"\n  Numeric summary — {label}:")
    print(f"  {'Column':<35} {'N':>8} {'Mean':>9} {'SD':>9} {'Min':>9} {'Median':>9} {'Max':>9} {'NaN':>6}")
    print(f"  {'-'*97}")
    for col in cols:
        if col not in df.columns: continue
        s = df[col].dropna()
        if len(s) == 0: continue
        print(f"  {col:<35} {len(s):>8,} {s.mean():>9.2f} {s.std():>9.2f} {s.min():>9.2f} {s.median():>9.2f} {s.max():>9.2f} {df[col].isna().sum():>6,}")

# =============================================================================
# COMPLETION CHECKS
# =============================================================================

def _s01_base_is_done():
    """Checks that pipeline.py Stage 01 already ran (Clean table exists)."""
    if not os.path.exists(DB_PATH): return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if CLEAN_TABLE not in tables: return False
            return conn.execute(f"SELECT COUNT(*) FROM {CLEAN_TABLE}").fetchone()[0] > 0
    except: return False

def _s01_cv_is_done():
    """Checks that both CV fold tables exist and are populated."""
    if not _s01_base_is_done(): return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if FOLD_TABLE_HOSPITAL not in tables or FOLD_TABLE_TEMPORAL not in tables: return False
            return (conn.execute(f"SELECT COUNT(*) FROM {FOLD_TABLE_HOSPITAL}").fetchone()[0] > 0 and
                    conn.execute(f"SELECT COUNT(*) FROM {FOLD_TABLE_TEMPORAL}").fetchone()[0] > 0)
    except: return False

def _s02_task_is_done(task_id):
    _, fname = S02_TASKS[task_id]
    return os.path.exists(os.path.join(BERT_DIR, fname))

def _get_n_folds(cv_type):
    """Returns the actual fold count stored in the fold table (dynamic for hospital CV)."""
    fold_table = FOLD_TABLE_HOSPITAL if cv_type == 'hospital' else FOLD_TABLE_TEMPORAL
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return conn.execute(f"SELECT COUNT(DISTINCT fold) FROM {fold_table}").fetchone()[0]
    except: return 0

def _s03_expected_count(n_folds):
    n_bert = sum(1 for tid in S02_TASKS if _s02_task_is_done(tid))
    n_text = len(CLASSICAL_ENCODINGS_WITH_N) + n_bert
    return n_folds * 2 * (1 + n_text * len(FEATURES_PER_COL))

def _s03_cv_is_done(cv_type):
    """
    Returns True only when the encoded DB contains the expected number of matrix
    rows AND every fold that exists in the fold-index table has both a 'train'
    and a 'val' entry in encoded_targets.  A row-count check alone is not
    sufficient because duplicates from a partial/crashed prior run can satisfy
    the count while leaving individual folds absent.
    """
    encoded_db  = ENCODED_DB_HOSPITAL if cv_type == 'hospital' else ENCODED_DB_TEMPORAL
    fold_table  = FOLD_TABLE_HOSPITAL  if cv_type == 'hospital' else FOLD_TABLE_TEMPORAL
    if not os.path.exists(encoded_db): return False
    n_folds = _get_n_folds(cv_type)
    if n_folds == 0: return False
    try:
        # ── Fetch the authoritative set of fold ids from the fold-index table ─
        with sqlite3.connect(DB_PATH) as conn:
            expected_folds = {int(r[0]) for r in conn.execute(f"SELECT DISTINCT fold FROM {fold_table}").fetchall()}
        if not expected_folds: return False

        with sqlite3.connect(encoded_db) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if 'encoded_matrices' not in tables or 'encoded_targets' not in tables: return False

            # ── Matrix count ─────────────────────────────────────────────────
            n_mat = conn.execute("SELECT COUNT(*) FROM encoded_matrices").fetchone()[0]
            if n_mat < _s03_expected_count(n_folds): return False

            # ── Verify every expected fold has a train AND val target row ────
            train_folds = {int(r[0]) for r in conn.execute("SELECT DISTINCT fold FROM encoded_targets WHERE split='train'").fetchall()}
            val_folds   = {int(r[0]) for r in conn.execute("SELECT DISTINCT fold FROM encoded_targets WHERE split='val'").fetchall()}
            if not expected_folds <= train_folds: return False
            if not expected_folds <= val_folds:   return False

        return True
    except: return False

# =============================================================================
# STAGE 01 — CV FOLD GENERATION
# =============================================================================

def run_stage01():
    if _s01_cv_is_done():
        print(f"  ⏭  Stage 01 already complete (fold_hospital + fold_temporal in {DB_PATH}). Skipping.")
        with sqlite3.connect(DB_PATH) as conn:
            for tbl in [FOLD_TABLE_HOSPITAL, FOLD_TABLE_TEMPORAL]:
                n_folds = conn.execute(f"SELECT COUNT(DISTINCT fold) FROM {tbl}").fetchone()[0]
                print(f"    {tbl}: {n_folds} folds")
        return

    if not _s01_base_is_done():
        print(f"  ❌ pipeline.py Stage 01 not complete — run it first to generate the Clean table.")
        return

    tee = _Tee(f'{LOG_DIR_HOSPITAL}/01_cv_fold_generation.log')
    sys.stdout = tee
    try:
        sep("STAGE 01 — CV FOLD GENERATION  (hospital + temporal)")

        # ── Load Clean table ─────────────────────────────────────────────────
        sep("LOAD CLEAN TABLE")
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(f"SELECT * FROM {CLEAN_TABLE}", conn)
        df = df[df[TARGET].notna()].copy().reset_index(drop=True)
        print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns from '{CLEAN_TABLE}'")
        print(f"  {HOSPITAL_COL} distribution:")
        for loc, cnt in df[HOSPITAL_COL].value_counts().items():
            print(f"    {loc:<25} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

        # ── Re-derive case_date from raw CSV via case_id ──────────────────────
        # pipeline.py Stage 01 drops all 'dttm'/'date' columns after feature
        # engineering. We recover the chronological order here without altering
        # the Clean table.
        sep("DERIVE case_date FROM RAW CSV  (scheduled_start_dttm via case_id join)")
        df_raw = pd.read_csv(RAW_CSV)
        df_raw.columns = df_raw.columns.str.strip()
        if 'scheduled_start_dttm' not in df_raw.columns:
            print(f"  ❌ 'scheduled_start_dttm' not found in {RAW_CSV}. Available datetime-like columns:")
            print(f"    {[c for c in df_raw.columns if any(x in c.lower() for x in ['dttm', 'date', 'time'])]}")
            return
        df_raw['_case_id']   = df_raw['case_id'].astype(str).str.strip()
        df_raw['_case_date'] = pd.to_datetime(df_raw['scheduled_start_dttm'], errors='coerce').dt.normalize()
        date_map = df_raw.set_index('_case_id')['_case_date'].to_dict()
        df['case_date'] = df['case_id'].astype(str).str.strip().map(date_map)
        n_missing_date = df['case_date'].isna().sum()
        print(f"  Mapped case_date for {df['case_date'].notna().sum():,}/{len(df):,} rows")
        if n_missing_date > 0:
            print(f"  ⚠ {n_missing_date} rows have no case_date — they will be placed at the START of the temporal sort (oldest assumed)")
            df['case_date'] = df['case_date'].fillna(pd.Timestamp('1900-01-01'))
        print(f"  Date range: {df['case_date'].min().date()} → {df['case_date'].max().date()}")

        # ── HOSPITAL CV: leave-one-location-out ───────────────────────────────
        sep("HOSPITAL CV — LEAVE-ONE-LOCATION-OUT")
        unique_locs = sorted(df[HOSPITAL_COL].dropna().unique())
        print(f"  Locations found ({len(unique_locs)} folds): {unique_locs}")
        hosp_rows = []
        for fold, loc in enumerate(unique_locs):
            va_mask = df[HOSPITAL_COL] == loc
            tr_mask = ~va_mask
            tr_idx  = df.index[tr_mask].tolist()
            va_idx  = df.index[va_mask].tolist()
            for idx in tr_idx:
                hosp_rows.append({'fold': fold, 'split': 'train', 'row_index': int(idx), 'case_id': df['case_id'].iloc[idx], 'fold_label': loc})
            for idx in va_idx:
                hosp_rows.append({'fold': fold, 'split': 'val',   'row_index': int(idx), 'case_id': df['case_id'].iloc[idx], 'fold_label': loc})
            print(f"  Fold {fold}  [{loc:<25}]  train={len(tr_idx):>6,}  val={len(va_idx):>6,}")
        hosp_df = pd.DataFrame(hosp_rows)
        with sqlite3.connect(DB_PATH) as conn:
            hosp_df.to_sql(FOLD_TABLE_HOSPITAL, conn, if_exists='replace', index=False)
        print(f"  ✅ Saved '{FOLD_TABLE_HOSPITAL}' → {DB_PATH}  ({len(unique_locs)} folds)")

        # ── TEMPORAL CV: expanding time-series split ──────────────────────────
        sep("TEMPORAL CV — EXPANDING TIME-SERIES SPLIT")
        from sklearn.model_selection import TimeSeriesSplit
        df_sorted     = df.sort_values('case_date', kind='mergesort').reset_index(drop=True)
        # df_sorted.index = positions in sorted order; original row indices are in the column below
        orig_indices  = df_sorted.index.tolist()   # after reset_index, positional == original
        # We need to map sorted positions back to Clean table row indices.
        # Because df was already reset_index(drop=True) above, df.index == row_index.
        # After sort+reset, df_sorted's positional index is the sorted position,
        # but the original Clean-table row index is stored in a separate mapping.
        sorted_to_orig = df.sort_values('case_date', kind='mergesort').index.tolist()

        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TEMPORAL)
        temp_rows = []
        print(f"  TimeSeriesSplit: n_splits={N_SPLITS_TEMPORAL}  (expanding window, each val set is a future chunk)")
        print(f"  Sorted date range: {df_sorted['case_date'].iloc[0].date()} → {df_sorted['case_date'].iloc[-1].date()}")
        print(f"\n  {'Fold':<6} {'Train':>12} {'Val':>12}  {'Val date range'}")
        print(f"  {'-'*65}")
        for fold, (tr_pos, va_pos) in enumerate(tscv.split(np.arange(len(df_sorted)))):
            tr_idx = [sorted_to_orig[p] for p in tr_pos]
            va_idx = [sorted_to_orig[p] for p in va_pos]
            va_dates = df_sorted.iloc[va_pos]['case_date']
            date_range = f"{va_dates.min().date()} → {va_dates.max().date()}"
            for idx in tr_idx:
                temp_rows.append({'fold': fold, 'split': 'train', 'row_index': int(idx), 'case_id': df['case_id'].iloc[idx]})
            for idx in va_idx:
                temp_rows.append({'fold': fold, 'split': 'val',   'row_index': int(idx), 'case_id': df['case_id'].iloc[idx]})
            print(f"  {fold:<6} {len(tr_idx):>12,} {len(va_idx):>12,}  {date_range}")
        temp_df = pd.DataFrame(temp_rows)
        with sqlite3.connect(DB_PATH) as conn:
            temp_df.to_sql(FOLD_TABLE_TEMPORAL, conn, if_exists='replace', index=False)
        print(f"\n  ✅ Saved '{FOLD_TABLE_TEMPORAL}' → {DB_PATH}  ({N_SPLITS_TEMPORAL} folds)")

        print(f"\n  ✅ Stage 01 complete.  Log → {LOG_DIR_HOSPITAL}/01_cv_fold_generation.log")
    finally:
        tee.close()

# =============================================================================
# STAGE 02 — BERT CACHE  (wrapper — skips if pipeline.py already ran it)
# =============================================================================

def run_stage02():
    sep("STAGE 02 — BERT CACHE CHECK")
    all_done = all(_s02_task_is_done(tid) for tid in S02_TASKS)
    if all_done:
        print(f"  ⏭  All BERT cache files already exist — nothing to do.")
        for tid, (method, fname) in S02_TASKS.items():
            arr = np.load(os.path.join(BERT_DIR, fname))
            print(f"    Task {tid}  {fname:<52}  shape={arr.shape}")
        return
    print(f"  ⚠ Some BERT cache files are missing. Run pipeline.py Stage 02 first.")
    for tid, (method, fname) in S02_TASKS.items():
        status = "✅ exists" if _s02_task_is_done(tid) else "❌ missing"
        print(f"    [{tid}]  {method:<16}  {fname}  {status}")

# =============================================================================
# STAGE 03 — FOLD ENCODING  (shared helpers + parameterized runner)
# =============================================================================

def _s03_init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encoded_matrices (
            fold INTEGER, split TEXT, encoding TEXT, n_features INTEGER,
            rows INTEGER, cols INTEGER, dtype TEXT, data BLOB
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encoded_targets (
            fold INTEGER, split TEXT, rows INTEGER, dtype TEXT, data BLOB
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_enc ON encoded_matrices (fold, split, encoding, n_features)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tgt ON encoded_targets (fold, split)")
    conn.commit()

def _s03_save_matrix(conn, fold, split, encoding, n_features, arr):
    conn.execute(
        "INSERT INTO encoded_matrices (fold,split,encoding,n_features,rows,cols,dtype,data) VALUES (?,?,?,?,?,?,?,?)",
        (int(fold), split, encoding, int(n_features), arr.shape[0], arr.shape[1], str(arr.dtype), arr.tobytes())
    )

def _s03_save_target(conn, fold, split, arr):
    conn.execute(
        "INSERT INTO encoded_targets (fold,split,rows,dtype,data) VALUES (?,?,?,?,?)",
        (int(fold), split, len(arr), str(arr.dtype), arr.tobytes())
    )

def _s03_impute_fold(train_df, val_df):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor, XGBClassifier
    train_df, val_df = train_df.copy(), val_df.copy()
    print(f"\n  Imputation (fit on training rows only):")
    for col, ptype in zip(IMPUTE_COLS, IMPUTE_TYPES):
        n_tr = train_df[col].isna().sum(); n_va = val_df[col].isna().sum()
        if n_tr == 0 and n_va == 0:
            print(f"    [{col}]  No NaN — skipping."); continue
        print(f"    [{col}]  type={ptype}  train_NaN={n_tr:,}  val_NaN={n_va:,}")
        num_feats = [c for c in train_df.columns if c not in [col, TARGET, TEXT_COL] and pd.api.types.is_numeric_dtype(train_df[c])]
        pre      = Pipeline([('imp', SimpleImputer(strategy='median'))])
        tr_known = train_df[train_df[col].notna()]
        tr_miss  = train_df[train_df[col].isna()]
        va_miss  = val_df[val_df[col].isna()]
        if len(tr_known) == 0:
            print(f"      → ⚠ No known train values — skipping."); continue
        X_tr_k = pre.fit_transform(tr_known[num_feats])
        if ptype == 'classification':
            le    = LabelEncoder()
            y_trk = le.fit_transform(tr_known[col].astype(str))
            mdl   = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss', verbosity=0)
            mdl.fit(X_tr_k, y_trk)
            def predict_col(X, _le=le, _mdl=mdl): return _le.inverse_transform(np.round(_mdl.predict(X)).astype(int))
        else:
            mdl = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
            mdl.fit(X_tr_k, tr_known[col].values)
            predict_col = mdl.predict
        if len(tr_miss) > 0:
            train_df.loc[train_df[col].isna(), col] = predict_col(pre.transform(tr_miss[num_feats]))
            print(f"      → Filled {len(tr_miss):,} train NaN")
        if len(va_miss) > 0:
            val_df.loc[val_df[col].isna(), col] = predict_col(pre.transform(va_miss[num_feats]))
            print(f"      → Filled {len(va_miss):,} val NaN")
    return train_df, val_df

def _s03_onehot_fold(train_df, val_df, col, fold_label):
    train_df, val_df = train_df.copy(), val_df.copy()
    cats = sorted(train_df[col].dropna().unique())
    print(f"    [{col}]  fold '{fold_label}' train categories: {cats}")
    for cat in cats:
        cname = f'{col}__{cat}'
        train_df[cname] = (train_df[col] == cat).astype(int)
        val_df[cname]   = (val_df[col]   == cat).astype(int)
    train_df.drop(columns=[col], inplace=True)
    val_df.drop(columns=[col], inplace=True)
    return train_df, val_df

def _s03_onehot_all(train_df, val_df, fold_label):
    print(f"\n  Fold-wise one-hot (train categories only — no leakage):")
    for col in ['anesthetic_type'] + CATEGORICAL_FOLD_COLS:
        train_df, val_df = _s03_onehot_fold(train_df, val_df, col, fold_label)
    return train_df, val_df

def _s03_encode_label(tr_texts, va_texts, n):
    enc_tr, enc_va = [], []
    for tr_t, va_t in zip(tr_texts, va_texts):
        top = pd.Series(tr_t).value_counts().nlargest(n - 1).index
        def _enc(texts, _top=top):
            mapped = pd.Series(texts).where(pd.Series(texts).isin(_top), 'Other')
            return pd.get_dummies(mapped, prefix=f'{TEXT_COL}_lbl').reindex(
                columns=[f'{TEXT_COL}_lbl_{c}' for c in list(_top) + ['Other']], fill_value=0
            ).values
        enc_tr.append(_enc(tr_t)); enc_va.append(_enc(va_t))
    return np.hstack(enc_tr).astype(np.float32), np.hstack(enc_va).astype(np.float32)

def _s03_encode_tfidf(tr_texts, va_texts, n):
    from sklearn.feature_extraction.text import TfidfVectorizer
    enc_tr, enc_va = [], []
    for tr_t, va_t in zip(tr_texts, va_texts):
        vec = TfidfVectorizer(max_features=n, stop_words='english', ngram_range=(1, 2))
        enc_tr.append(vec.fit_transform(tr_t).toarray())
        enc_va.append(vec.transform(va_t).toarray())
    return np.hstack(enc_tr).astype(np.float32), np.hstack(enc_va).astype(np.float32)

def _s03_encode_count(tr_texts, va_texts, n):
    from sklearn.feature_extraction.text import CountVectorizer
    enc_tr, enc_va = [], []
    for tr_t, va_t in zip(tr_texts, va_texts):
        vec = CountVectorizer(max_features=n, stop_words='english', ngram_range=(1, 2))
        enc_tr.append(vec.fit_transform(tr_t).toarray())
        enc_va.append(vec.transform(va_t).toarray())
    return np.hstack(enc_tr).astype(np.float32), np.hstack(enc_va).astype(np.float32)

def _s03_apply_encoding(train_df, val_df, encoding, n):
    struct_cols = [c for c in train_df.columns if c not in [TEXT_COL, TARGET, 'case_id']]
    S_tr = train_df[struct_cols].values.astype(np.float32)
    S_va = val_df[struct_cols].values.astype(np.float32)
    if encoding == 'only_structured':
        return S_tr, S_va
    tr_texts = [train_df[TEXT_COL].astype(str).tolist()]
    va_texts = [val_df[TEXT_COL].astype(str).tolist()]
    if encoding == 'label':
        T_tr, T_va = _s03_encode_label(tr_texts, va_texts, n)
    elif encoding == 'tfidf':
        T_tr, T_va = _s03_encode_tfidf(tr_texts, va_texts, n)
    elif encoding == 'count':
        T_tr, T_va = _s03_encode_count(tr_texts, va_texts, n)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    return np.hstack([S_tr, T_tr]), np.hstack([S_va, T_va])

def _s03_encode_bert_pca(X_struct_tr, X_struct_va, emb_tr, emb_va, n):
    from sklearn.decomposition import PCA
    n_comp    = min(n, emb_tr.shape[1], emb_tr.shape[0])
    pca       = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_bert_tr = pca.fit_transform(emb_tr)
    X_bert_va = pca.transform(emb_va)
    var_exp   = pca.explained_variance_ratio_.sum() * 100
    return np.hstack([X_struct_tr, X_bert_tr]).astype(np.float32), np.hstack([X_struct_va, X_bert_va]).astype(np.float32), n_comp, var_exp

def _run_encoding(cv_type):
    """Core encoding loop for one CV strategy. cv_type in ('hospital', 'temporal')."""
    fold_table  = FOLD_TABLE_HOSPITAL  if cv_type == 'hospital' else FOLD_TABLE_TEMPORAL
    encoded_db  = ENCODED_DB_HOSPITAL  if cv_type == 'hospital' else ENCODED_DB_TEMPORAL
    log_dir     = LOG_DIR_HOSPITAL     if cv_type == 'hospital' else LOG_DIR_TEMPORAL
    label       = cv_type.upper()

    if _s03_cv_is_done(cv_type):
        n_folds = _get_n_folds(cv_type)
        print(f"  ⏭  Stage 03 [{label}] already complete ({encoded_db}). Skipping.")
        with sqlite3.connect(encoded_db) as conn:
            for row in conn.execute("SELECT fold,split,encoding,n_features,rows,cols FROM encoded_matrices ORDER BY fold,split,encoding,n_features"):
                print(f"    fold={row[0]} split={row[1]} encoding={row[2]:<20} n={row[3]:>3}  shape=({row[4]},{row[5]})")
        return

    tee = _Tee(f'{log_dir}/03_fold_encoding_{cv_type}.log')
    sys.stdout = tee
    try:
        sep(f"STAGE 03 — FOLD ENCODING  [{label}]")

        # ── BERT cache ────────────────────────────────────────────────────────
        bert_cache = {}
        for tid, (method, fname) in S02_TASKS.items():
            path = os.path.join(BERT_DIR, fname)
            if os.path.exists(path):
                bert_cache[method] = np.load(path)
                print(f"  ✅ BERT cache  {fname}  shape={bert_cache[method].shape}")
            else:
                print(f"  ⚠ BERT cache missing: {fname} — BERT encodings skipped for this run")

        # ── Load Clean table ──────────────────────────────────────────────────
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(f"SELECT * FROM {CLEAN_TABLE}", conn)
        df = df[df[TARGET].notna()].copy().reset_index(drop=True)
        dropped_leak = [c for c in EXCLUDE_COLS if c in df.columns]
        df.drop(columns=dropped_leak, inplace=True)
        print(f"  Loaded   : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Excluded : {dropped_leak}  (intraoperative leakage)")

        # ── Load fold indices ─────────────────────────────────────────────────
        with sqlite3.connect(DB_PATH) as conn:
            fold_df = pd.read_sql(f"SELECT * FROM {fold_table}", conn)

        fold_indices  = {}
        fold_labels   = {}   # fold → label (location name for hospital, fold int for temporal)
        unique_folds  = sorted(fold_df['fold'].unique())
        for fold in active_folds:
            tr_idx = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='train')]['row_index'].values
            va_idx = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')]['row_index'].values
            fold_indices[fold]  = (tr_idx, va_idx)
            if 'fold_label' in fold_df.columns:
                fold_labels[fold] = fold_df[fold_df['fold']==fold]['fold_label'].iloc[0]
            else:
                fold_labels[fold] = str(fold)
        n_folds = len(unique_folds)
        print(f"  {n_folds} folds loaded from '{fold_table}'")
        for fold in unique_folds:
            tr, va = fold_indices[fold]
            print(f"    Fold {fold}  [{fold_labels[fold]}]  train={len(tr):,}  val={len(va):,}")

        if os.path.exists(encoded_db):
            os.remove(encoded_db)
            print(f"\n  Removed existing {encoded_db} — starting fresh.")
        with sqlite3.connect(encoded_db) as conn:
            _s03_init_db(conn)

        # ── Encoding loop ─────────────────────────────────────────────────────
        sep(f"ENCODING LOOP [{label}] — fold → impute → one-hot → encode → save")
        for fold in unique_folds:
            tr_idx, va_idx = fold_indices[fold]
            fold_lbl = fold_labels[fold]
            sep(f"FOLD {fold}  [{fold_lbl}]  train={len(tr_idx):,}  val={len(va_idx):,}", char='-')
            train_base = df.iloc[tr_idx].copy().reset_index(drop=True)
            val_base   = df.iloc[va_idx].copy().reset_index(drop=True)

            train_base, val_base = _s03_impute_fold(train_base, val_base)
            train_base, val_base = _s03_onehot_all(train_base, val_base, fold_lbl)

            y_train = train_base[TARGET].values.astype(np.float64)
            y_val   = val_base[TARGET].values.astype(np.float64)
            print(f"\n  y_train: mean={y_train.mean():.1f}  std={y_train.std():.1f}  min={y_train.min():.1f}  max={y_train.max():.1f}")
            print(f"  y_val  : mean={y_val.mean():.1f}  std={y_val.std():.1f}  min={y_val.min():.1f}  max={y_val.max():.1f}")
            with sqlite3.connect(encoded_db) as conn:
                _s03_save_target(conn, fold, 'train', y_train)
                _s03_save_target(conn, fold, 'val',   y_val)
                conn.commit()

            X_tr_struct, X_va_struct = _s03_apply_encoding(train_base, val_base, 'only_structured', 0)
            print(f"\n  only_structured  train={X_tr_struct.shape}  val={X_va_struct.shape}")
            with sqlite3.connect(encoded_db) as conn:
                _s03_save_matrix(conn, fold, 'train', 'only_structured', 0, X_tr_struct)
                _s03_save_matrix(conn, fold, 'val',   'only_structured', 0, X_va_struct)
                conn.commit()

            print(f"\n  Classical text encodings:")
            for n in FEATURES_PER_COL:
                for encoding in CLASSICAL_ENCODINGS_WITH_N:
                    X_tr, X_va = _s03_apply_encoding(train_base, val_base, encoding, n)
                    print(f"    n={n:>3}  {encoding:<8}  train={X_tr.shape}  val={X_va.shape}")
                    with sqlite3.connect(encoded_db) as conn:
                        _s03_save_matrix(conn, fold, 'train', encoding, n, X_tr)
                        _s03_save_matrix(conn, fold, 'val',   encoding, n, X_va)
                        conn.commit()

            if bert_cache:
                print(f"\n  BERT PCA encodings (PCA fit on train fold only — no leakage):")
            for method, emb in bert_cache.items():
                emb_tr = emb[tr_idx].astype(np.float32)
                emb_va = emb[va_idx].astype(np.float32)
                for n in FEATURES_PER_COL:
                    X_tr, X_va, n_comp, var_exp = _s03_encode_bert_pca(X_tr_struct, X_va_struct, emb_tr, emb_va, n)
                    print(f"    n={n:>3}  {method:<16}  PCA({n_comp}) → {var_exp:.1f}% var  train={X_tr.shape}  val={X_va.shape}")
                    with sqlite3.connect(encoded_db) as conn:
                        _s03_save_matrix(conn, fold, 'train', method, n, X_tr)
                        _s03_save_matrix(conn, fold, 'val',   method, n, X_va)
                        conn.commit()

        sep("SUMMARY")
        with sqlite3.connect(encoded_db) as conn:
            n_mat = conn.execute("SELECT COUNT(*) FROM encoded_matrices").fetchone()[0]
            n_tgt = conn.execute("SELECT COUNT(*) FROM encoded_targets").fetchone()[0]
            print(f"  encoded_matrices : {n_mat}  |  encoded_targets : {n_tgt}")
        print(f"  DB size : {os.path.getsize(encoded_db)/1024/1024:.1f} MB")
        print(f"  ✅ Stage 03 [{label}] complete.  Log → {log_dir}/03_fold_encoding_{cv_type}.log")
    finally:
        tee.close()

def run_stage03():
    sep("STAGE 03 — FOLD ENCODING  (hospital + temporal)")
    statuses = {cv: ('✅ done' if _s03_cv_is_done(cv) else '❌ pending') for cv in ('hospital', 'temporal')}
    print(f"  [1]  Hospital CV  (leave-one-location-out)  {statuses['hospital']}")
    print(f"  [2]  Temporal CV  (time-series split)       {statuses['temporal']}")
    print(f"  [0]  Both  [default]")
    raw = input("  Select CV type to encode (e.g. 1 or Enter for both): ").strip()
    if raw == '' or raw == '0':
        selected = ['hospital', 'temporal']
    else:
        MAP = {'1': 'hospital', '2': 'temporal'}
        selected = []
        for part in raw.split(','):
            part = part.strip()
            if part in MAP:
                selected.append(MAP[part])
            else:
                print(f"  ⚠ Ignored: '{part}'")
    if not selected:
        print("  No CV type selected — exiting."); return
    for cv in selected:
        _run_encoding(cv)

# =============================================================================
# STAGE 04 — MODELING  (parameterized by CV strategy)
# =============================================================================

def _run_modeling(cv_type, model_list):
    """Full modeling pipeline for one CV strategy."""

    import optuna
    from scipy.stats import t as scipy_t
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.base import BaseEstimator
    import xgboost as xgb
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import AdamW
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.mixed_precision import set_global_policy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fold_table  = FOLD_TABLE_HOSPITAL  if cv_type == 'hospital' else FOLD_TABLE_TEMPORAL
    encoded_db  = ENCODED_DB_HOSPITAL  if cv_type == 'hospital' else ENCODED_DB_TEMPORAL
    result_db   = RESULT_DB_HOSPITAL   if cv_type == 'hospital' else RESULT_DB_TEMPORAL
    log_dir     = LOG_DIR_HOSPITAL     if cv_type == 'hospital' else LOG_DIR_TEMPORAL
    label       = cv_type.upper()

    # ── TF GPU setup ──────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            set_global_policy('mixed_float16')
            print(f"  ✅ TF GPU: {[g.name for g in gpus]}  mixed_float16 enabled")
        except RuntimeError as e:
            print(f"  ⚠ TF GPU config error: {e}")
    else:
        print("  ⚠ No TF GPU detected — MLP will run on CPU")

    xgb_device = 'cpu'
    try:
        _probe = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1, verbosity=0)
        _probe.fit(np.zeros((10, 2)), np.zeros(10))
        xgb_device = 'cuda'
        print(f"  ✅ XGBoost CUDA available")
    except Exception:
        print(f"  ⚠ XGBoost CUDA unavailable — falling back to CPU")

    tee = _Tee(f'{log_dir}/04_modeling_{cv_type}.log')
    sys.stdout = tee
    try:
        sep(f"STAGE 04 — MODELING  [{label}]")
        print(f"  CV strategy   : {label}")
        print(f"  Fold table    : {fold_table}")
        print(f"  Encoded DB    : {encoded_db}")
        print(f"  Result DB     : {result_db}")
        print(f"  Log / PDFs    : {log_dir}/")
        print(f"  Models        : {model_list}")
        print(f"  XGBoost       : device={xgb_device}  early_stopping_rounds={XGB_EARLY_STOPPING_ROUNDS}")
        print(f"  MLP           : AdamW + BatchNorm  depth={OPTUNA_MLP_N_LAYERS_LOW}-{OPTUNA_MLP_N_LAYERS_HIGH} layers  epochs={MLP_EPOCHS_FINAL}/{MLP_EPOCHS_OPTUNA}")

        # ── Load fold indices ─────────────────────────────────────────────────
        with sqlite3.connect(DB_PATH) as conn:
            fold_df = pd.read_sql(f"SELECT * FROM {fold_table}", conn)
        unique_folds = sorted(fold_df['fold'].unique())
        fold_indices = {}
        fold_labels  = {}
        for fold in unique_folds:
            fold_indices[fold] = (
                fold_df[(fold_df['fold']==fold) & (fold_df['split']=='train')]['row_index'].values,
                fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')]['row_index'].values,
            )
            if 'fold_label' in fold_df.columns:
                fold_labels[fold] = fold_df[fold_df['fold']==fold]['fold_label'].iloc[0]
            else:
                fold_labels[fold] = str(fold)
        n_folds = len(unique_folds)
        print(f"\n  {n_folds} folds loaded")

        # ── Identify and skip degenerate folds (too few val cases) ───────────
        skipped_folds = {}
        for fold in unique_folds:
            n_val = len(fold_indices[fold][1])
            if n_val < MIN_VAL_FOLD_SIZE:
                skipped_folds[fold] = n_val
        if skipped_folds:
            print(f"\n  ⚠ Folds skipped (val size < MIN_VAL_FOLD_SIZE={MIN_VAL_FOLD_SIZE}):")
            for fold, n_val in skipped_folds.items():
                print(f"    Fold {fold}  [{fold_labels[fold]}]  val={n_val}  — too few cases for reliable evaluation")
        active_folds = [f for f in unique_folds if f not in skipped_folds]
        print(f"\n  Active folds for modeling: {len(active_folds)}/{len(unique_folds)}")
        for fold in active_folds:
            tr, va = fold_indices[fold]
            print(f"    Fold {fold}  [{fold_labels[fold]}]  train={len(tr):,}  val={len(va):,}")

        # ── Pre-flight: verify encoded DB is fully populated ──────────────────
        # This guard catches any mismatch between what _s03_cv_is_done approved
        # and what is actually readable at query time (e.g. leftover stale DB
        # from a crashed prior run, type coercion edge cases, WAL not flushed).
        sep(f"PRE-FLIGHT CHECK [{label}] — encoded_targets integrity")
        expected_folds = {int(f) for f in unique_folds}
        preflight_ok = True
        try:
            with sqlite3.connect(encoded_db) as conn:
                train_folds = {int(r[0]) for r in conn.execute("SELECT DISTINCT fold FROM encoded_targets WHERE split='train'").fetchall()}
                val_folds   = {int(r[0]) for r in conn.execute("SELECT DISTINCT fold FROM encoded_targets WHERE split='val'").fetchall()}
            missing_train = expected_folds - train_folds
            missing_val   = expected_folds - val_folds
            if missing_train or missing_val:
                preflight_ok = False
                print(f"  ❌ encoded_targets is incomplete in {encoded_db}")
                if missing_train: print(f"     Missing train targets for folds: {sorted(missing_train)}")
                if missing_val:   print(f"     Missing val targets for folds: {sorted(missing_val)}")
                print(f"     → Re-run Stage 03 [{label}] to rebuild the encoded DB, then retry Stage 04.")
            else:
                print(f"  ✅ All {n_folds} folds have train + val targets in {encoded_db}")
        except Exception as e:
            preflight_ok = False
            print(f"  ❌ Pre-flight check failed: {e}")
            print(f"     → Re-run Stage 03 [{label}] to rebuild the encoded DB, then retry Stage 04.")
        if not preflight_ok:
            return

        # ── Inner helpers ─────────────────────────────────────────────────────
        from sklearn.preprocessing import RobustScaler

        def load_fold_matrices(fold):
            result = {}
            with sqlite3.connect(encoded_db) as conn:
                rows = conn.execute(
                    "SELECT split, encoding, n_features, rows, cols, dtype, data FROM encoded_matrices WHERE fold=?",
                    (int(fold),)
                ).fetchall()
            for split, encoding, n_features, n_rows, n_cols, dtype, data in rows:
                result[(split, encoding, int(n_features))] = np.frombuffer(data, dtype=dtype).reshape(n_rows, n_cols).copy()
            return result

        def load_target(fold, split):
            with sqlite3.connect(encoded_db) as conn:
                row = conn.execute(
                    "SELECT rows,dtype,data FROM encoded_targets WHERE fold=? AND split=?",
                    (int(fold), split)
                ).fetchone()
            if row is None: raise ValueError(f"Target not found: fold={fold} split={split}  (encoded_db={encoded_db})")
            return np.frombuffer(row[2], dtype=row[1]).copy()

        def save_db(df_save, table):
            with sqlite3.connect(result_db, timeout=30) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                df_save.to_sql(table, conn, if_exists='append', index=False)

        result_db_is_new = not os.path.exists(result_db)

        def delete_existing(fold, encoding, n_features, model_name):
            if result_db_is_new: return
            with sqlite3.connect(result_db, timeout=30) as conn:
                for table in ['metrics', 'predictions', 'feature_importance', 'hyperparameter']:
                    try:
                        conn.execute(
                            f"DELETE FROM {table} WHERE fold=? AND encoding=? AND n_features=? AND model=?",
                            (int(fold), encoding, int(n_features), model_name)
                        )
                    except Exception: pass
                conn.commit()

        def compute_metrics(y_true, y_pred):
            y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
            errs  = y_true - y_pred
            mse   = mean_squared_error(y_true, y_pred)
            mae   = mean_absolute_error(y_true, y_pred)
            r2    = r2_score(y_true, y_pred)
            denom = np.abs(y_true) + np.abs(y_pred)
            vld   = denom != 0
            smape = 100 * np.mean(2 * np.abs(y_true[vld]-y_pred[vld]) / denom[vld]) if vld.any() else np.nan
            mu, sd = np.mean(errs), np.std(errs)
            ci = scipy_t.interval(0.95, len(errs)-1, loc=mu, scale=sd/np.sqrt(len(errs)))
            return {'mse': mse, 'rmse': np.sqrt(mse), 'mae': mae, 'r2': r2, 'smape': smape, 'mean_error': mu, 'std_error': sd, 'ci95_low': ci[0], 'ci95_high': ci[1]}

        def build_model(name, params=None):
            p = params or {}
            if name == 'linear':   return LinearRegression()
            if name == 'ridge':    return Ridge(alpha=p.get('alpha', 1.0))
            if name == 'lasso':    return Lasso(alpha=p.get('alpha', 1.0))
            if name == 'randomforest':
                return RandomForestRegressor(
                    n_estimators=p.get('n_estimators', 200), max_depth=p.get('max_depth', 6),
                    max_features=p.get('max_features', 'sqrt'), min_samples_split=p.get('min_samples_split', 2),
                    min_samples_leaf=p.get('min_samples_leaf', 1), random_state=RANDOM_STATE, n_jobs=-1
                )
            if name == 'xgboost':
                return xgb.XGBRegressor(
                    n_estimators=p.get('n_estimators', 200), learning_rate=p.get('learning_rate', 0.05),
                    max_depth=p.get('max_depth', 4), subsample=p.get('subsample', 0.8),
                    colsample_bytree=p.get('colsample_bytree', 0.8), reg_alpha=p.get('reg_alpha', 0.01),
                    reg_lambda=p.get('reg_lambda', 1.0), tree_method=XGB_TREE_METHOD, device=xgb_device,
                    early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS, random_state=RANDOM_STATE, n_jobs=-1
                )
            if name == 'mlp':
                act      = p.get('activation', 'relu')
                n_layers = p.get('n_layers', 2)
                opt = AdamW(learning_rate=p.get('lr', 1e-3), weight_decay=p.get('weight_decay', 1e-4), clipnorm=MLP_CLIPNORM)
                layer_sizes = [OPTUNA_MLP_UNITS1_HIGH, OPTUNA_MLP_UNITS2_HIGH, OPTUNA_MLP_UNITS3_HIGH]
                layer_list  = [Input(shape=(p['input_dim'],))]
                for i in range(n_layers):
                    layer_list.append(Dense(p.get(f'units{i+1}', layer_sizes[i] // 2), activation=act))
                    layer_list.append(BatchNormalization())
                    layer_list.append(Dropout(p.get(f'dropout{i+1}', 0.1)))
                layer_list.append(Dense(1, dtype='float32'))
                mdl = Sequential(layer_list)
                mdl.compile(loss='mse', optimizer=opt)
                return mdl
            raise ValueError(f"Unknown model: {name}")

        def fit_predict(name, mdl, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva=None, final=False):
            t0 = time.perf_counter()
            if name == 'mlp':
                Xtr_f = Xtr_sc.astype(np.float32); ytr_f = ytr.astype(np.float32)
                if final and yva is not None:
                    Xva_f = Xva_sc.astype(np.float32); yva_f = yva.astype(np.float32)
                    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr_f, ytr_f)).shuffle(min(10_000, len(ytr_f)), seed=RANDOM_STATE).batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    val_ds   = (tf.data.Dataset.from_tensor_slices((Xva_f, yva_f)).batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    cbs = [EarlyStopping(monitor='val_loss', patience=MLP_PATIENCE_ES, restore_best_weights=True, verbose=0), ReduceLROnPlateau(monitor='val_loss', factor=MLP_LR_DECAY_FACTOR, patience=MLP_PATIENCE_LR, min_lr=MLP_MIN_LR, verbose=0)]
                    mdl.fit(train_ds, validation_data=val_ds, epochs=MLP_EPOCHS_FINAL, callbacks=cbs, verbose=0)
                    train_t = time.perf_counter() - t0
                    ti = time.perf_counter(); preds = mdl.predict(val_ds, verbose=0).flatten(); infer_t = time.perf_counter() - ti
                else:
                    mdl.fit(Xtr_f, ytr_f, epochs=MLP_EPOCHS_OPTUNA, batch_size=MLP_BATCH_SIZE, verbose=0)
                    train_t = time.perf_counter() - t0
                    ti = time.perf_counter(); preds = mdl.predict(Xva_sc.astype(np.float32), verbose=0).flatten(); infer_t = time.perf_counter() - ti
            elif name == 'xgboost':
                if yva is not None: mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                else:               mdl.fit(Xtr, ytr)
                train_t = time.perf_counter() - t0
                ti = time.perf_counter(); preds = mdl.predict(Xva); infer_t = time.perf_counter() - ti
            elif name == 'randomforest':
                mdl.fit(Xtr, ytr)
                train_t = time.perf_counter() - t0
                ti = time.perf_counter(); preds = mdl.predict(Xva); infer_t = time.perf_counter() - ti
            else:
                mdl.fit(Xtr_sc, ytr)
                train_t = time.perf_counter() - t0
                ti = time.perf_counter(); preds = mdl.predict(Xva_sc); infer_t = time.perf_counter() - ti
            return preds, train_t, infer_t

        class _OptunaPruningCallback(tf.keras.callbacks.Callback):
            def __init__(self, trial):
                super().__init__(); self._trial = trial
            def on_epoch_end(self, epoch, logs=None):
                val_loss = (logs or {}).get('val_loss')
                if val_loss is None: return
                self._trial.report(float(val_loss), epoch)
                if self._trial.should_prune(): raise optuna.exceptions.TrialPruned()

        def make_objective(name, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva):
            def obj(trial):
                if name in ('ridge', 'lasso'):
                    p = {'alpha': trial.suggest_float('alpha', OPTUNA_ALPHA_LOW, OPTUNA_ALPHA_HIGH, log=True)}
                elif name == 'randomforest':
                    p = {'n_estimators': trial.suggest_int('n_estimators', OPTUNA_RF_N_EST_LOW, OPTUNA_RF_N_EST_HIGH), 'max_depth': trial.suggest_int('max_depth', OPTUNA_RF_MAX_DEPTH_LOW, OPTUNA_RF_MAX_DEPTH_HIGH), 'max_features': trial.suggest_categorical('max_features', OPTUNA_RF_MAX_FEATURES), 'min_samples_split': trial.suggest_int('min_samples_split', OPTUNA_RF_MIN_SAMPLES_SPLIT_LOW, OPTUNA_RF_MIN_SAMPLES_SPLIT_HIGH), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', OPTUNA_RF_MIN_SAMPLES_LEAF_LOW, OPTUNA_RF_MIN_SAMPLES_LEAF_HIGH), 'random_state': RANDOM_STATE, 'n_jobs': -1}
                elif name == 'xgboost':
                    p = {'n_estimators': trial.suggest_int('n_estimators', OPTUNA_XGB_N_EST_LOW, OPTUNA_XGB_N_EST_HIGH), 'learning_rate': trial.suggest_float('learning_rate', OPTUNA_XGB_LR_LOW, OPTUNA_XGB_LR_HIGH, log=True), 'max_depth': trial.suggest_int('max_depth', OPTUNA_XGB_MAX_DEPTH_LOW, OPTUNA_XGB_MAX_DEPTH_HIGH), 'subsample': trial.suggest_float('subsample', OPTUNA_XGB_SUBSAMPLE_LOW, OPTUNA_XGB_SUBSAMPLE_HIGH), 'colsample_bytree': trial.suggest_float('colsample_bytree', OPTUNA_XGB_COLSAMPLE_LOW, OPTUNA_XGB_COLSAMPLE_HIGH), 'reg_alpha': trial.suggest_float('reg_alpha', OPTUNA_XGB_REG_ALPHA_LOW, OPTUNA_XGB_REG_ALPHA_HIGH, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', OPTUNA_XGB_REG_LAMBDA_LOW, OPTUNA_XGB_REG_LAMBDA_HIGH, log=True), 'tree_method': XGB_TREE_METHOD, 'device': xgb_device, 'early_stopping_rounds': XGB_EARLY_STOPPING_ROUNDS, 'random_state': RANDOM_STATE, 'n_jobs': -1}
                elif name == 'mlp':
                    tf.keras.backend.clear_session()
                    n_layers = trial.suggest_int('n_layers', OPTUNA_MLP_N_LAYERS_LOW, OPTUNA_MLP_N_LAYERS_HIGH)
                    units_bounds = [(OPTUNA_MLP_UNITS1_LOW, OPTUNA_MLP_UNITS1_HIGH), (OPTUNA_MLP_UNITS2_LOW, OPTUNA_MLP_UNITS2_HIGH), (OPTUNA_MLP_UNITS3_LOW, OPTUNA_MLP_UNITS3_HIGH)]
                    p = {'input_dim': Xtr_sc.shape[1], 'n_layers': n_layers, 'activation': trial.suggest_categorical('activation', OPTUNA_MLP_ACTIVATIONS), 'lr': trial.suggest_float('lr', OPTUNA_MLP_LR_LOW, OPTUNA_MLP_LR_HIGH, log=True), 'weight_decay': trial.suggest_float('weight_decay', OPTUNA_MLP_WEIGHT_DECAY_LOW, OPTUNA_MLP_WEIGHT_DECAY_HIGH, log=True)}
                    for i in range(OPTUNA_MLP_N_LAYERS_HIGH):
                        lo_u, hi_u = units_bounds[i]
                        p[f'units{i+1}']   = trial.suggest_int(f'units{i+1}', lo_u, hi_u)
                        p[f'dropout{i+1}'] = trial.suggest_float(f'dropout{i+1}', OPTUNA_MLP_DROPOUT_LOW, OPTUNA_MLP_DROPOUT_HIGH)
                else:
                    p = {}

                if name == 'mlp':
                    mdl   = build_model('mlp', p)
                    Xtr_f = Xtr_sc.astype(np.float32); ytr_f = ytr.astype(np.float32)
                    Xva_f = Xva_sc.astype(np.float32); yva_f = yva.astype(np.float32)
                    if MLP_OPTUNA_SUBSET_SIZE and len(Xtr_f) > MLP_OPTUNA_SUBSET_SIZE:
                        rng = np.random.default_rng(RANDOM_STATE)
                        idx = rng.choice(len(Xtr_f), MLP_OPTUNA_SUBSET_SIZE, replace=False)
                        Xtr_f = Xtr_f[idx]; ytr_f = ytr_f[idx]
                    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr_f, ytr_f)).shuffle(min(10_000, len(ytr_f)), seed=RANDOM_STATE).batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    val_ds   = (tf.data.Dataset.from_tensor_slices((Xva_f, yva_f)).batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    history  = mdl.fit(train_ds, validation_data=val_ds, epochs=MLP_EPOCHS_OPTUNA, callbacks=[_OptunaPruningCallback(trial)], verbose=0)
                    return float(min(history.history.get('val_loss', [np.inf])))
                else:
                    mdl = build_model(name, p)
                    preds, _, _ = fit_predict(name, mdl, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva=yva)
                    return mean_squared_error(yva, preds)
            return obj

        def save_model_artifacts(model_name, model_rows):
            df_m = pd.DataFrame(model_rows)
            grp  = df_m.groupby(['encoding', 'n_features'])[['mae', 'smape', 'r2', 'rmse', 'train_time_s', 'infer_time_s']].agg(['mean', 'std'])
            grp.columns = ['_'.join(c) for c in grp.columns]
            grp = grp.reset_index()

            log_path = os.path.join(log_dir, f'{model_name}.log')
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*80}\n  MODEL: {model_name.upper()}  |  CV: {label}\n{'='*80}\n\n")
                f.write("PER-FOLD RESULTS\n")
                f.write(f"  {'Encoding':<18} {'n':>6} {'Fold':>5} {'FoldLabel':<28} {'MAE':>8} {'SMAPE':>8} {'R²':>8} {'RMSE':>8} {'Train(s)':>10} {'Infer(s)':>10}\n")
                f.write(f"  {'-'*115}\n")
                for _, row in df_m.sort_values(['encoding', 'n_features', 'fold']).iterrows():
                    n_str  = str(int(row['n_features'])) if row['n_features'] > 0 else 'struct'
                    flabel = row.get('fold_label', '')
                    f.write(f"  {row['encoding']:<18} {n_str:>6} {int(row['fold']):>5} {str(flabel):<28} {row['mae']:>8.3f} {row['smape']:>8.3f} {row['r2']:>8.4f} {row['rmse']:>8.3f} {row['train_time_s']:>10.2f} {row['infer_time_s']:>10.4f}\n")
                f.write(f"\n\nAGGREGATED SUMMARY  (mean ± std across {n_folds} folds)\n")
                f.write(f"  {'Encoding':<18} {'n':>6} {'MAE mean':>10} {'±':>4} {'SMAPE mean':>12} {'±':>4} {'R² mean':>9} {'±':>4} {'Train(s)':>10} {'Infer(s)':>10}\n")
                f.write(f"  {'-'*103}\n")
                for _, row in grp.sort_values('mae_mean').iterrows():
                    n_str = str(int(row['n_features'])) if row['n_features'] > 0 else 'struct'
                    f.write(f"  {row['encoding']:<18} {n_str:>6} {row['mae_mean']:>10.3f} {row['mae_std']:>4.3f} {row['smape_mean']:>12.3f} {row['smape_std']:>4.3f} {row['r2_mean']:>9.4f} {row['r2_std']:>4.4f} {row['train_time_s_mean']:>10.2f} {row['infer_time_s_mean']:>10.4f}\n")
            print(f"  📝 {model_name}.log  → {log_path}")

            plots_cfg = [
                ('mae_mean',   'mae_std',   'MAE (minutes)', 'MAE',   False),
                ('smape_mean', 'smape_std', 'SMAPE (%)',      'SMAPE', False),
                ('r2_mean',    'r2_std',    'R²',             'R²',    True),
                ('train_time_s_mean', 'train_time_s_std', 'Train Time (s)', 'Train Time', False),
            ]
            enc_order = ['only_structured'] + CLASSICAL_ENCODINGS_WITH_N + BERT_ENCODINGS
            enc_order = [e for e in enc_order if e in grp['encoding'].unique()]
            n_order   = [0] + FEATURES_PER_COL
            n_order   = [n for n in n_order if n in grp['n_features'].unique()]
            enc_palette = {enc: plt.cm.tab10.colors[i % 10] for i, enc in enumerate(enc_order)}
            n_palette   = {n:   plt.cm.Set2.colors[i % 8]   for i, n   in enumerate(n_order)}
            n_labels    = {n: ('struct' if n == 0 else str(n)) for n in n_order}
            lookup      = {(r['encoding'], int(r['n_features'])): r for _, r in grp.iterrows()}

            def _get(enc, n, col_mean, col_std):
                row = lookup.get((enc, int(n)))
                return (np.nan, np.nan) if row is None else (row[col_mean], row[col_std])

            pdf_path = os.path.join(log_dir, f'{model_name}.pdf')
            with PdfPages(pdf_path) as pdf:
                for page, (group_by, groups, series, s_palette, g_labels, s_labels, page_title) in enumerate([
                    ('encoding', enc_order, n_order, n_palette, enc_order, n_labels, 'Page 1: Grouped by Encoding Method  (colours = feature count n)'),
                    ('n',        n_order,   enc_order, enc_palette, [f'n={n_labels[n]}' for n in n_order], {e: e for e in enc_order}, 'Page 2: Grouped by Feature Count (n)  (colours = encoding method)'),
                ]):
                    n_groups  = len(groups)
                    n_series  = len(series)
                    bar_w     = min(0.8 / max(n_series, 1), 0.25)
                    offsets   = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_w
                    fig, axes = plt.subplots(2, 2, figsize=(max(14, n_groups * 1.8), 12))
                    fig.suptitle(f'Model: {model_name.upper()}  |  CV: {label}  —  {page_title}\n(error bars = ±1 std;  red border = best per metric)', fontsize=10, fontweight='bold', y=0.98)
                    if page == 0:
                        legend_handles = [plt.Rectangle((0,0), 1, 1, color=n_palette[nk], alpha=0.85, label=f'n={n_labels[nk]}') for nk in n_order]
                    else:
                        legend_handles = [plt.Rectangle((0,0), 1, 1, color=enc_palette[ek], alpha=0.85, label=ek) for ek in enc_order]
                    fig.legend(handles=legend_handles, loc='upper center', ncol=min(len(legend_handles), 6), fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.96))
                    for ax, (mean_col, std_col, ylabel, title, higher_better) in zip(axes.flat, plots_cfg):
                        best_val, best_bar = None, None
                        for si, s in enumerate(series):
                            if page == 0:
                                means = np.array([_get(g, s, mean_col, std_col)[0] for g in groups])
                                stds  = np.array([_get(g, s, mean_col, std_col)[1] for g in groups])
                                color = n_palette[s]
                            else:
                                means = np.array([_get(s, g, mean_col, std_col)[0] for g in groups])
                                stds  = np.array([_get(s, g, mean_col, std_col)[1] for g in groups])
                                color = enc_palette[s]
                            x_pos = np.arange(n_groups) + offsets[si]
                            valid = ~np.isnan(means)
                            if not valid.any(): continue
                            bars = ax.bar(x_pos[valid], means[valid], width=bar_w*0.92, yerr=stds[valid], capsize=2, color=color, alpha=0.85, edgecolor='black', linewidth=0.4, error_kw={'elinewidth': 0.9})
                            for bar, val in zip(bars, means[valid]):
                                ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height()*1.005, f'{val:.2f}', ha='center', va='bottom', fontsize=5.5, rotation=90)
                            for bar, val in zip(bars, means[valid]):
                                if best_val is None or (higher_better and val > best_val) or (not higher_better and val < best_val):
                                    best_val, best_bar = val, bar
                        if best_bar is not None:
                            best_bar.set_edgecolor('red'); best_bar.set_linewidth(2.0)
                        ax.set_xticks(np.arange(n_groups))
                        ax.set_xticklabels(g_labels if page == 0 else [f'n={n_labels[g]}' for g in groups], fontsize=9, rotation=25, ha='right')
                        ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11, fontweight='bold')
                        ax.grid(axis='y', linestyle='--', alpha=0.4)
                    plt.tight_layout(rect=[0, 0, 1, 0.91]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            print(f"  📊 {model_name}.pdf  → {pdf_path}  (2 pages)")

        # ── Main fold loop ────────────────────────────────────────────────────
        sep(f"MAIN LOOP [{label}]  — fold → load → scale → tune → fit → evaluate")
        all_metrics   = []
        model_buckets = {m: [] for m in model_list}

        for fold in unique_folds:
            train_idx, val_idx = fold_indices[fold]
            fold_lbl = fold_labels[fold]
            sep(f"FOLD {fold}  [{fold_lbl}]  train={len(train_idx):,}  val={len(val_idx):,}", char='-')

            y_train = load_target(fold, 'train')
            y_val   = load_target(fold, 'val')
            print(f"  y_train: mean={y_train.mean():.1f}  std={y_train.std():.1f}  min={y_train.min():.1f}  max={y_train.max():.1f}")
            print(f"  y_val  : mean={y_val.mean():.1f}  std={y_val.std():.1f}  min={y_val.min():.1f}  max={y_val.max():.1f}")

            print(f"\n  Loading all matrices for fold {fold} (one DB connection)...")
            raw_mats = load_fold_matrices(fold)

            fold_encoded = {}
            for (split, enc, n), arr in raw_mats.items():
                fold_encoded.setdefault((enc, n), {})[split] = arr

            for key in list(fold_encoded.keys()):
                d = fold_encoded[key]
                if 'train' not in d or 'val' not in d:
                    del fold_encoded[key]; continue
                scaler = RobustScaler()
                fold_encoded[key] = (d['train'], scaler.fit_transform(d['train']), d['val'], scaler.transform(d['val']))

            for enc in ['only_structured'] + CLASSICAL_ENCODINGS_WITH_N + BERT_ENCODINGS:
                keys = sorted([(e, n) for e, n in fold_encoded if e == enc], key=lambda x: x[1])
                for e, n in keys:
                    X_tr, _, X_va, _ = fold_encoded[(e, n)]
                    print(f"    {e:<20} n={n:>3}  train={X_tr.shape}  val={X_va.shape}")

            all_combos = sorted(fold_encoded.keys(), key=lambda x: (x[0], x[1]))

            for (encoding, n) in all_combos:
                X_train, X_train_sc, X_val, X_val_sc = fold_encoded[(encoding, n)]
                n_label = f"n={n}" if n > 0 else "no text"
                print(f"\n  {'─'*60}")
                print(f"  Encoding: {encoding.upper()}  {n_label}")

                for model_name in model_list:
                    delete_existing(fold, encoding, n, model_name)
                    print(f"    🔧 {model_name}")
                    try:
                        best_p = {}
                        if model_name != 'linear':
                            idx_tr2, idx_tune = train_test_split(np.arange(len(y_train)), test_size=TUNE_SET_FRACTION, random_state=RANDOM_STATE)
                            sc_inner  = RobustScaler()
                            X_tr2_sc  = sc_inner.fit_transform(X_train[idx_tr2])
                            X_tune_sc = sc_inner.transform(X_train[idx_tune])
                            sampler = optuna.samplers.TPESampler(n_startup_trials=OPTUNA_N_STARTUP_TRIALS, multivariate=True, seed=RANDOM_STATE)
                            pruner  = optuna.pruners.MedianPruner(n_startup_trials=OPTUNA_N_STARTUP_TRIALS, n_warmup_steps=5)
                            study   = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
                            study.optimize(make_objective(model_name, X_train[idx_tr2], X_tr2_sc, X_train[idx_tune], X_tune_sc, y_train[idx_tr2], y_train[idx_tune]), n_trials=N_TRIALS)
                            best_p = study.best_trial.params
                            if model_name == 'mlp':
                                best_p['input_dim'] = X_train_sc.shape[1]
                                tf.keras.backend.clear_session()
                            print(f"      Optuna best MSE={study.best_trial.value:.2f}  params={best_p}")

                        mdl = build_model(model_name, best_p)
                        preds, train_time_s, infer_time_s = fit_predict(model_name, mdl, X_train, X_train_sc, X_val, X_val_sc, y_train, yva=y_val, final=True)

                        metrics = compute_metrics(y_val, preds)
                        metrics.update({'fold': fold, 'fold_label': fold_lbl, 'encoding': encoding, 'n_features': n, 'model': model_name, 'cv_type': cv_type, 'train_time_s': round(train_time_s, 4), 'infer_time_s': round(infer_time_s, 4)})
                        all_metrics.append(metrics)
                        model_buckets[model_name].append(metrics)
                        print(f"      ✅ MAE={metrics['mae']:>7.2f}  SMAPE={metrics['smape']:>6.2f}%  R²={metrics['r2']:>6.4f}  RMSE={metrics['rmse']:>7.2f}  train={train_time_s:.2f}s  infer={infer_time_s:.4f}s")

                        save_db(pd.DataFrame([metrics]), 'metrics')

                        val_case_ids = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')]['case_id'].values
                        save_db(pd.DataFrame({'fold': fold, 'fold_label': fold_lbl, 'encoding': encoding, 'n_features': n, 'model': model_name, 'cv_type': cv_type, 'case_id': val_case_ids, 'actual': y_val, 'predicted': preds}), 'predictions')

                        imps = {}
                        try:
                            if model_name in ('randomforest', 'xgboost'):
                                imps = {f'f{i}': float(v) for i, v in enumerate(mdl.feature_importances_)}
                            elif model_name in ('linear', 'ridge', 'lasso'):
                                imps = {f'f{i}': float(v) for i, v in enumerate(mdl.coef_)}
                            elif model_name == 'mlp':
                                X_tf = tf.constant(X_val_sc.astype(np.float32))
                                with tf.GradientTape() as tape:
                                    tape.watch(X_tf)
                                    out = tf.cast(mdl(X_tf, training=False), tf.float32)
                                grads = tape.gradient(out, X_tf)
                                if grads is not None:
                                    imps = {f'f{i}': float(v) for i, v in enumerate(np.abs(grads.numpy()).mean(axis=0))}
                        except Exception as e:
                            print(f"      ⚠ Feature importance failed: {e}")

                        save_db(pd.DataFrame([{'fold': fold, 'fold_label': fold_lbl, 'encoding': encoding, 'n_features': n, 'model': model_name, 'cv_type': cv_type, 'importances': str(imps)}]), 'feature_importance')
                        params_clean = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in best_p.items()}
                        save_db(pd.DataFrame([{'fold': fold, 'fold_label': fold_lbl, 'encoding': encoding, 'n_features': n, 'model': model_name, 'cv_type': cv_type, 'params': str(params_clean)}]), 'hyperparameter')

                    except Exception as e:
                        print(f"      🚫 Error: {e}")

        # ── Per-model artifacts ───────────────────────────────────────────────
        sep("PER-MODEL ARTIFACTS  (.log + .pdf)")
        for model_name in model_list:
            rows = model_buckets[model_name]
            if rows:
                print(f"\n  Generating artifacts for: {model_name}")
                save_model_artifacts(model_name, rows)
            else:
                print(f"\n  ⚠ No results for {model_name} — skipping.")

        # ── Final summary ─────────────────────────────────────────────────────
        sep("FINAL AGGREGATED SUMMARY")
        if all_metrics:
            df_m    = pd.DataFrame(all_metrics)
            grouped = df_m.groupby(['encoding', 'n_features', 'model'])[['mae', 'smape', 'r2', 'rmse', 'train_time_s', 'infer_time_s']].agg(['mean', 'std'])
            grouped.columns = ['_'.join(c) for c in grouped.columns]
            grouped = grouped.reset_index().sort_values('mae_mean')
            print(f"\n  {'Encoding':<18} {'n':>5} {'Model':<14} {'MAE mean':>10} {'MAE std':>9} {'SMAPE mean':>12} {'R² mean':>9} {'RMSE mean':>11}")
            print(f"  {'-'*96}")
            for _, row in grouped.iterrows():
                n_str = str(int(row['n_features'])) if row['n_features'] > 0 else 'struct'
                print(f"  {row['encoding']:<18} {n_str:>5} {row['model']:<14} {row['mae_mean']:>10.3f} {row['mae_std']:>9.3f} {row['smape_mean']:>12.3f} {row['r2_mean']:>9.4f} {row['rmse_mean']:>11.3f}")
            sep("BEST CONFIGURATION PER METRIC")
            for metric, direction in [('mae_mean', 'min'), ('smape_mean', 'min'), ('r2_mean', 'max'), ('train_time_s_mean', 'min'), ('infer_time_s_mean', 'min')]:
                idx  = grouped[metric].idxmin() if direction == 'min' else grouped[metric].idxmax()
                best = grouped.loc[idx]
                n_str = str(int(best['n_features'])) if best['n_features'] > 0 else 'struct_only'
                print(f"  Best {metric:<18}: encoding={best['encoding']}  n={n_str}  model={best['model']}  value={best[metric]:.4f}")

        print(f"\n  Results → {result_db}")
        print(f"  Log     → {log_dir}/04_modeling_{cv_type}.log")
        print(f"  ✅ Stage 04 [{label}] complete.")
    finally:
        tee.close()

def run_stage04():
    sep("STAGE 04 — MODELING  (hospital + temporal)")

    # ── Select CV type ────────────────────────────────────────────────────────
    enc_status = {cv: ('✅ encoded' if _s03_cv_is_done(cv) else '❌ not encoded') for cv in ('hospital', 'temporal')}
    print(f"  [1]  Hospital CV  (leave-one-location-out)  {enc_status['hospital']}")
    print(f"  [2]  Temporal CV  (time-series split)       {enc_status['temporal']}")
    print(f"  [0]  Both  [default]")
    raw_cv = input("  Select CV type for modeling (e.g. 1 or Enter for both): ").strip()
    if raw_cv == '' or raw_cv == '0':
        cv_selected = ['hospital', 'temporal']
    else:
        CV_MAP = {'1': 'hospital', '2': 'temporal'}
        cv_selected = []
        for part in raw_cv.split(','):
            part = part.strip()
            if part in CV_MAP:
                cv_selected.append(CV_MAP[part])
            else:
                print(f"  ⚠ Ignored: '{part}'")
    if not cv_selected:
        print("  No CV type selected — exiting."); return

    # Verify encoding exists for each selected CV type
    cv_ready = []
    for cv in cv_selected:
        if not _s03_cv_is_done(cv):
            print(f"  ❌ Stage 03 [{cv.upper()}] not complete — run Stage 03 first. Skipping {cv}.")
        else:
            cv_ready.append(cv)
    if not cv_ready:
        return

    # ── Select models ─────────────────────────────────────────────────────────
    MODEL_EMOJIS = {'linear': '📏 Linear Regression', 'ridge': '🏔️  Ridge Regression', 'lasso': '🪢 Lasso Regression', 'randomforest': '🌲 Random Forest', 'xgboost': '⚡ XGBoost', 'mlp': '🧠 MLP Neural Network'}
    print(f"\n  [0]  🚀 Run ALL models  [default]")
    for i, m in enumerate(ALL_MODELS, 1):
        print(f"  [{i}]  {MODEL_EMOJIS[m]}")
    raw_m = input("  Select models (e.g. 1,3,5 or Enter for all): ").strip()
    if raw_m == '' or raw_m == '0':
        model_list = ALL_MODELS[:]
    else:
        model_list = []
        for part in raw_m.split(','):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= len(ALL_MODELS):
                model_list.append(ALL_MODELS[int(part)-1])
            else:
                print(f"  ⚠ Ignored: '{part}'")
        model_list = list(dict.fromkeys(model_list))
    if not model_list:
        print("  No valid models selected — exiting."); return
    print(f"\n  CV strategies : {cv_ready}")
    print(f"  Models        : {model_list}")

    for cv in cv_ready:
        sep(f"RUNNING [{cv.upper()}]")
        _run_modeling(cv, model_list)

# =============================================================================
# MAIN
# =============================================================================

def main():
    sep("pipeline_cv.py  —  Hospital CV + Temporal CV  —  Stages 01 · 02 · 03 · 04")
    print()
    print("  Prerequisites: pipeline.py Stages 01 and 02 must be complete.")
    print()
    print("  [1]  Stage 01 — CV Fold Generation   fold_hospital + fold_temporal → surgical_data.db")
    print("  [2]  Stage 02 — BERT Cache Check      verify caches exist (run pipeline.py Stage 02 if missing)")
    print("  [3]  Stage 03 — Fold Encoding         encode for selected CV strategy → separate .db files")
    print("  [4]  Stage 04 — Modeling              tune → fit → evaluate for selected CV strategy")
    print("  [0]  Run ALL stages in order  [default]")
    print()

    if not _s01_base_is_done():
        print(f"  ⚠ WARNING: pipeline.py Stage 01 not detected ({DB_PATH} / Clean table missing).")
        print(f"     Run pipeline.py Stage 01 first, then return here.")
        print()

    raw = input("  Select stage(s) (e.g. 2,4 or Enter for all): ").strip()
    STAGE_MAP = {1: run_stage01, 2: run_stage02, 3: run_stage03, 4: run_stage04}
    if raw == '' or raw == '0':
        selected = [1, 2, 3, 4]
    else:
        selected = []
        for part in raw.split(','):
            part = part.strip()
            if part.isdigit() and int(part) in STAGE_MAP:
                selected.append(int(part))
            else:
                print(f"  ⚠ Ignored unrecognized input: '{part}'")
        selected = list(dict.fromkeys(selected))
    if not selected:
        print("  No valid stages selected — exiting."); return
    print(f"\n  Running stages: {selected}\n")
    for s in selected:
        STAGE_MAP[s]()
    sep("ALL SELECTED STAGES COMPLETE")

if __name__ == '__main__':
    main()