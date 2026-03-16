# =============================================================================
# pipeline.py  —  Stages 01 · 02 · 03 · 04
#
# Stages 01–03 check completion on every run and skip if already done.
# Stage 04 always asks which model(s) to run — safe for multi-terminal use.
# All tuneable settings live in the CONFIG block below — nowhere else.
#
# Changes vs. original:
#   CONFIG  : MLP_EPOCHS_OPTUNA 10→30, MLP_PATIENCE_ES 5→15,
#             MLP_PATIENCE_LR 3→5, TUNE_SET_FRACTION 0.2→0.25,
#             MLP_N_LAYERS removed (now Optuna-tunable 1-3 layers),
#             OPTUNA_MLP_N_LAYERS_LOW/HIGH + UNITS3 bounds added,
#             OPTUNA_RF_MAX_FEATURES extended with float options,
#             XGB_EARLY_STOPPING_ROUNDS added,
#             MLP_OPTUNA_SUBSET_SIZE added.
#   Stage 03: BERT PCA now computed and persisted here (fold-wise,
#             train-only PCA) — eliminates recomputation in Stage 04.
#             DB index added for fast lookups.
#   Stage 04: AdamW replaces Adam (weight_decay now actually applied),
#             BatchNormalization added after each Dense layer,
#             MLP depth tunable via Optuna (1-3 layers),
#             XGBoost early stopping in both Optuna + final fit,
#             MinMaxScaler refitted on Optuna sub-split (no leakage),
#             yva passed to make_objective for XGB early stopping,
#             Gradient saliency replaces slow permutation importance for MLP,
#             BERT combos loaded from DB like classical (no PCA in Stage 04),
#             XGBoost CUDA availability checked at runtime with CPU fallback,
#             Single DB connection per fold for matrix loading,
#             MLP Optuna: static search space fixes TPE multivariate fallback,
#             MLP Optuna: single .fit() call with _OptunaPruningCallback
#             replaces epoch-by-epoch loop (eliminates per-epoch TF overhead),
#             MLP_OPTUNA_SUBSET_SIZE caps trial data for large-dataset speed.
# =============================================================================

# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG  —  edit only this block                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_CSV    = './data/casetime.csv'
DB_PATH    = './data/surgical_data.db'
BERT_DIR   = './data/bert_cache'
ENCODED_DB = './data/fold_encoded.db'
RESULT_DB  = './results/result.db'
LOG_DIR    = './results'

# ── Column names ──────────────────────────────────────────────────────────────
TARGET   = 'actual_casetime_minutes'
TEXT_COL = 'scheduled_procedure'

# Saved as raw strings in Clean; one-hot encoded fold-wise in Stage 03
CATEGORICAL_FOLD_COLS = ['case_service', 'surgical_location']

# NaN columns imputed fold-wise in Stage 03 (fit on train only — no leakage)
IMPUTE_COLS  = ['age_at_discharge', 'avg_BMI', 'anesthetic_type']
IMPUTE_TYPES = ['regression',       'regression', 'classification']

# Dropped in Stage 03 — intraoperative measurements known only after surgery
EXCLUDE_COLS = ['procedure_minutes', 'procedure_time', 'induction_time',
                'emergence_time',    'scheduled_duration']

# ── Cross-validation ──────────────────────────────────────────────────────────
N_SPLITS     = 5
RANDOM_STATE = 42

# ── Text feature dimensionality ───────────────────────────────────────────────
FEATURES_PER_COL = [10, 50, 100, 200]

# ── Optuna general ────────────────────────────────────────────────────────────
N_TRIALS                = 50
TUNE_SET_FRACTION       = 0.25   # was 0.20 — wider sub-split reduces noise in hyperparameter selection
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
OPTUNA_RF_MAX_FEATURES           = ['sqrt', 'log2', 0.3, 0.5, 0.7]  # extended: floats cover sparser high-dim BERT spaces

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
OPTUNA_MLP_N_LAYERS_LOW      = 1    # depth is now tunable (was hardcoded to 2)
OPTUNA_MLP_N_LAYERS_HIGH     = 3
OPTUNA_MLP_UNITS1_LOW        = 32
OPTUNA_MLP_UNITS1_HIGH       = 256
OPTUNA_MLP_UNITS2_LOW        = 16
OPTUNA_MLP_UNITS2_HIGH       = 128
OPTUNA_MLP_UNITS3_LOW        = 8    # only used when n_layers == 3
OPTUNA_MLP_UNITS3_HIGH       = 64
OPTUNA_MLP_DROPOUT_LOW       = 0.0
OPTUNA_MLP_DROPOUT_HIGH      = 0.5
OPTUNA_MLP_LR_LOW            = 1e-4
OPTUNA_MLP_LR_HIGH           = 1e-2
OPTUNA_MLP_WEIGHT_DECAY_LOW  = 1e-6
OPTUNA_MLP_WEIGHT_DECAY_HIGH = 1e-2
OPTUNA_MLP_ACTIVATIONS       = ['relu', 'elu', 'tanh']

# ── MLP fixed training settings ───────────────────────────────────────────────
MLP_EPOCHS_FINAL      = 200
MLP_EPOCHS_OPTUNA     = 30     # was 10 — more epochs needed to distinguish architectures with BatchNorm warmup
MLP_BATCH_SIZE        = 512
MLP_PATIENCE_ES       = 15    # was 5 — less aggressive early stopping across 200 epochs
MLP_PATIENCE_LR       = 5     # was 3
MLP_LR_DECAY_FACTOR   = 0.5
MLP_MIN_LR            = 1e-6
MLP_CLIPNORM          = 1.0
MLP_OPTUNA_SUBSET_SIZE = 5000  # max rows fed to MLP per Optuna trial (None = all tune rows)
                                # dramatically reduces per-trial cost on large datasets;
                                # the final fit always uses the full training fold

# ── XGBoost training settings ─────────────────────────────────────────────────
XGB_TREE_METHOD           = 'hist'
XGB_DEVICE                = 'cuda'   # runtime-checked in Stage 04; falls back to 'cpu' automatically
XGB_EARLY_STOPPING_ROUNDS = 20       # applied in both Optuna trials and final fit

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
FOLD_TABLE  = 'fold_indices'

# ── BERT cache task map (auto-derived — do not edit) ─────────────────────────
S02_TASKS = {
    1: ('clinicalbert', f'clinicalbert_{TEXT_COL}.npy'),
    2: ('sentencebert', f'sentencebert_{TEXT_COL}.npy'),
}

# =============================================================================
# IMPORTS
# =============================================================================
import os, sys, re, sqlite3, time, warnings
import numpy as np
import pandas as pd

os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs('./data', exist_ok=True)
os.makedirs(BERT_DIR, exist_ok=True)

warnings.filterwarnings('ignore')

# =============================================================================
# SHARED UTILITIES
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

def _print_missing(df, label=''):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print(f"  No missing values{' — ' + label if label else ''}."); return
    total = len(df)
    print(f"\n  Missing — {label}:")
    print(f"  {'Column':<45} {'N':>8} {'%':>8}")
    print(f"  {'-'*63}")
    for col, cnt in missing.items():
        print(f"  {col:<45} {cnt:>8,} {cnt/total*100:>7.2f}%")

def _print_freq(series, label='', top_n=20):
    vc    = series.value_counts(dropna=False)
    total = len(series)
    print(f"\n  Freq — {label or series.name}  (top {top_n}):")
    print(f"  {'Value':<45} {'N':>8} {'%':>8}")
    print(f"  {'-'*63}")
    for val, cnt in vc.head(top_n).items():
        print(f"  {str(val):<45} {cnt:>8,} {cnt/total*100:>7.2f}%")

def _print_numeric(df, cols, label=''):
    print(f"\n  Numeric summary — {label}:")
    print(f"  {'Column':<35} {'N':>8} {'Mean':>9} {'SD':>9} {'Min':>9} {'Median':>9} {'Max':>9} {'NaN':>6}")
    print(f"  {'-'*97}")
    for col in cols:
        if col not in df.columns: continue
        s = df[col].dropna()
        if len(s) == 0: continue
        print(f"  {col:<35} {len(s):>8,} {s.mean():>9.2f} {s.std():>9.2f} {s.min():>9.2f} {s.median():>9.2f} {s.max():>9.2f} {df[col].isna().sum():>6,}")

# ── Stage completion checks ───────────────────────────────────────────────────

def _s01_is_done():
    if not os.path.exists(DB_PATH): return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if CLEAN_TABLE not in tables or FOLD_TABLE not in tables: return False
            return (conn.execute(f"SELECT COUNT(*) FROM {CLEAN_TABLE}").fetchone()[0] > 0 and
                    conn.execute(f"SELECT COUNT(*) FROM {FOLD_TABLE}").fetchone()[0] > 0)
    except: return False

def _s02_task_is_done(task_id):
    _, fname = S02_TASKS[task_id]
    return os.path.exists(os.path.join(BERT_DIR, fname))

def _s03_expected_count():
    """Expected encoded_matrices row count given whichever BERT caches exist."""
    n_bert_available = sum(1 for tid in S02_TASKS if _s02_task_is_done(tid))
    n_text_encs      = len(CLASSICAL_ENCODINGS_WITH_N) + n_bert_available
    return N_SPLITS * 2 * (1 + n_text_encs * len(FEATURES_PER_COL))

def _s03_is_done():
    if not os.path.exists(ENCODED_DB): return False
    try:
        with sqlite3.connect(ENCODED_DB) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if 'encoded_matrices' not in tables: return False
            n = conn.execute("SELECT COUNT(*) FROM encoded_matrices").fetchone()[0]
            return n >= _s03_expected_count()
    except: return False

# =============================================================================
# STAGE 01 — PRE-PROCESSING  (unchanged)
# =============================================================================

def run_stage01():
    if _s01_is_done():
        print(f"  ⏭  Stage 01 already complete ({DB_PATH} → {CLEAN_TABLE} + {FOLD_TABLE}). Skipping.")
        return

    tee = _Tee(f'{LOG_DIR}/01_preprocessing.log')
    sys.stdout = tee
    TIME_COLS = ['procedure_minutes', 'actual_casetime_minutes', 'procedure_time',
                 'induction_time', 'emergence_time', 'scheduled_duration']
    try:
        sep("STAGE 01 — PRE-PROCESSING")

        sep("1 — LOAD & INITIAL CLEANING")
        df = pd.read_csv(RAW_CSV)
        df.columns = df.columns.str.strip()
        n_init = len(df)
        print(f"  Loaded : {n_init:,} rows × {df.shape[1]} columns")
        print(f"  Columns: {df.columns.tolist()}")
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace(r'^\s*$', np.nan, regex=True)
        pat = re.compile(r'^\s*(nan|none|null|na|n/a|missing|unknown|\?|-|)\s*$', re.IGNORECASE)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and pat.match(x) else x)
        print(f"  Shape after normalize + clean_missing: {df.shape[0]:,} × {df.shape[1]}")
        _print_missing(df, "after initial cleaning")

        sep("2 — DATETIME FEATURES")
        dt_all = [c for c in df.columns if any(x in c.lower() for x in ['dttm', 'date', 'time', 'minute'])]
        before = len(df); df.dropna(subset=dt_all, inplace=True)
        print(f"  Removed {before - len(df):,} rows with missing datetime values → {len(df):,} remaining")
        parse_cols = [c for c in df.columns if any(x in c.lower() for x in ['dttm', 'date'])]
        for c in parse_cols:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        df['procedure_time']       = (df['procedure_stop_dttm']  - df['procedure_start_dttm']).dt.total_seconds() / 60
        df['induction_time']       = (df['procedure_start_dttm'] - df['OR_entered_dttm']).dt.total_seconds() / 60
        df['emergence_time']       = (df['OR_left_dttm']         - df['procedure_stop_dttm']).dt.total_seconds() / 60
        df['scheduled_duration']   = (df['scheduled_end_dttm']   - df['scheduled_start_dttm']).dt.total_seconds() / 60
        df['scheduled_start_hour'] = df['scheduled_start_dttm'].dt.hour
        df['or_entry_hour']        = df['OR_entered_dttm'].dt.hour
        df['month_of_year']        = df['scheduled_start_dttm'].dt.month
        df['day_of_week']          = df['scheduled_start_dttm'].dt.dayofweek
        df.drop(columns=parse_cols, inplace=True)
        print(f"  Derived: procedure_time, induction_time, emergence_time, scheduled_duration, scheduled_start_hour, or_entry_hour, month_of_year, day_of_week")
        print(f"  Target : {TARGET}  (full OR occupancy — room-in to room-out)")

        sep("3 — IMPLAUSIBLE DURATION FILTER  [0, 2880 min]  (hardcoded rule)")
        for col in TIME_COLS:
            if col not in df.columns: continue
            before = len(df); df = df[df[col].between(0, 2880)]
            if before - len(df): print(f"  {col}: removed {before - len(df):,} rows outside [0, 2880] min")
        print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")

        sep("4 — TARGET DISTRIBUTION")
        _print_numeric(df, ['actual_casetime_minutes', 'scheduled_duration',
                             'procedure_time', 'induction_time', 'emergence_time'])
        s = df['actual_casetime_minutes'].dropna()
        print(f"\n  Percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]:
            print(f"    {p:5.1f}th : {s.quantile(p/100):>8.1f} min")

        sep("5 — CLAMP INVALID AGE / BMI  (hardcoded bounds)")
        inv_age = ~df['age_at_discharge'].between(18, 130)
        df.loc[inv_age, 'age_at_discharge'] = np.nan
        print(f"  age_at_discharge: clamped {inv_age.sum():,} outside [18,130] → NaN  (fold-wise imputation)")
        inv_bmi = ~df['avg_BMI'].between(5, 200)
        df.loc[inv_bmi, 'avg_BMI'] = np.nan
        print(f"  avg_BMI:          clamped {inv_bmi.sum():,} outside [5,200]   → NaN  (fold-wise imputation)")

        sep("6 — DROP IDENTIFIER COLUMNS")
        drop_ids = [c for c in ['patient_id', 'avg_wt_enct', 'avg_ht_enct', 'week_day'] if c in df.columns]
        df.drop(columns=drop_ids, inplace=True)
        print(f"  Dropped: {drop_ids}  →  shape={df.shape[0]:,} × {df.shape[1]}")
        print(f"  Retained: case_id  (join key for post-hoc subgroup analysis — not used as a feature)")

        sep("7 — MISSINGNESS")
        _print_missing(df, "before dropping non-imputable")
        must_have = ['ASA_score', 'operative_dx', 'sex', 'surg_encounter_type', 'case_service']
        before = len(df); df.dropna(subset=must_have, inplace=True)
        print(f"\n  Dropped {before - len(df):,} rows missing in {must_have}")
        _print_missing(df, "after dropping non-imputable")

        sep("8 — CATEGORICAL CLEANING")

        df['ASA_score'] = df['ASA_score'].apply(
            lambda x: int(m.group(1)) if (m := re.match(r'^([1-5])(?:e)?$', str(x).strip().lower())) else np.nan
        )
        df.dropna(subset=['ASA_score'], inplace=True)
        _print_freq(df['ASA_score'], "ASA_score after cleanup")

        df['OR_trip_sequence'] = (df['OR_trip_sequence'] == 1).astype(int)
        for col, val in [
            ('first_scheduled_case_of_day_status', 'first scheduled case of day'),
            ('last_scheduled_case_of_day_status',  'last scheduled case of day'),
            ('primary_procedure_status',           'primary procedure'),
        ]:
            df[col] = (df[col] == val).astype(int)

        before = len(df)
        df = df[df['sex'].isin(['male', 'female'])].copy()
        print(f"\n  [Sex]  dropped {before - len(df):,} rows not in ['male','female']  (hardcoded rule)")
        df['sex'] = (df['sex'] == 'male').astype(int)

        df['surgery_encounter_inpatient'] = np.where(
            df['surg_encounter_type'].str.lower().isin(['same day admission', 'one day stay']), 0,
            np.where(df['surg_encounter_type'].str.lower() == 'inpatient', 1, np.nan)
        )
        df.drop(columns=['surg_encounter_type'], inplace=True)
        before = len(df)
        df = df[df['surgery_encounter_inpatient'].notna()].copy()
        print(f"\n  [Surgical Encounter]  mapped: 0=outpatient  1=inpatient")
        print(f"  Dropped {before - len(df):,} rows with unknown encounter type  (hardcoded rule — dropped before fold generation to keep all indices consistent)")

        def _simplify_loc(loc):
            loc = str(loc).strip().lower()
            if loc.startswith('vh or'):   return 'VH_OR'
            if loc.startswith('uh or'):   return 'UH_OR'
            if loc.startswith('vsc or'):  return 'VSC_OR'
            if loc.startswith('zzvh ob'): return 'OB_VH'
            if 'anesthesia' in loc:       return 'Anesthesia'
            if any(x in loc for x in ['pacu', 'pmdu', 'phase', 'recovery']): return 'Recovery'
            if any(x in loc for x in ['tee', 'pain']): return 'Procedure_Room'
            if 'alternate' in loc:        return 'Alternate_OR'
            return 'Other'
        df['surgical_location'] = df['surgical_location'].apply(_simplify_loc)
        before = len(df)
        df = df[df['surgical_location'] != 'Other'].copy()
        print(f"\n  [Surgical Location]  dropped {before - len(df):,} rows mapped to 'Other'  (hardcoded rule)")
        print(f"  NOTE: one-hot deferred to Stage 03 (train-fold categories only)")
        _print_freq(df['surgical_location'], "surgical_location after mapping")

        svc_map = {
            'orthopedic surgery':    'Orthopedic',       'general surgery':      'General_Surgery',
            'obstetrics/gynecology': 'OB_GYN',           'otolaryngology':       'ENT',
            'urology':               'Urology',          'plastic surgery':      'Plastic_Surgery',
            'neurosurgery':          'Neurosurgery',     'cardiac surgery':      'Cardiac_Surgery',
            'vascular surgery':      'Vascular_Surgery', 'thoracic surgery':     'Thoracic_Surgery',
            'dental surgery':        'Dental_Surgery',   'ophthalmology':        'Ophthalmology',
            'lrcp surg':             'Surgical_Oncology','cardiology surg':      'Cardiac_Surgery',
            'medicine surg':         np.nan,             'unknown case service': np.nan,
            'anesthesia surg':       np.nan,
        }
        df['case_service'] = df['case_service'].str.lower().map(svc_map)
        before = len(df); df.dropna(subset=['case_service'], inplace=True)
        print(f"\n  [Case Service]  dropped {before - len(df):,} unmapped rows  (hardcoded rule)")
        print(f"  NOTE: one-hot deferred to Stage 03 (train-fold categories only)")
        _print_freq(df['case_service'], "case_service after mapping")

        df['anesthetic_type'] = df['anesthetic_type'].str.replace(r'^general/|/general', '', regex=True).str.strip()
        anesthesia_map = {
            "general": "General",                "general/epidural": "Combined",
            "general/regional": "Combined",      "general/spinal": "Combined",
            "general/spinal opioid": "Combined",  "general/axillary": "Combined",
            "general rectal": "General",          "general endo": "General",
            "general/home regional": "Combined",  "spinal block": "Neuraxial",
            "epidural block": "Neuraxial",        "combined spinal/epidural": "Neuraxial",
            "lumbar epidural block": "Neuraxial", "brachial plexus block": "Regional",
            "supraclavicular block": "Regional",  "interscalene block": "Regional",
            "infraclavicular block": "Regional",  "intercostal brachial": "Regional",
            "sciatic catheter block": "Regional", "paravertebral nerve block": "Regional",
            "transverse abdominus plane block": "Regional", "lumbar plexus block": "Regional",
            "cervical plexus block": "Regional",  "ilioinguinal block": "Regional",
            "axillary block": "Regional",         "femoral block": "Regional",
            "popliteal block": "Regional",        "saphenous knee block": "Regional",
            "saphenous elbow block": "Regional",  "suprascapular block": "Regional",
            "home regional": "Regional",          "regional": "Regional",
            "regional/home regional": "Regional", "local": "Local",
            "local with standby": "Local",        "local/sedation": "Local",
            "local - monitored anesthesia care": "Local", "facial block": "Local",
            "o'brien block": "Local",             "peribulbar and retrobulbar block": "Local",
            "ankle block": "Local",               "caudal block": "Local",
            "bier block": "Local",                "local neurolept": "Sedation",
            "iv sedation": "Sedation",            "neurolept": "Sedation",
            "iv regional": "Sedation",            "no anesthesia given": np.nan,
            "system": np.nan,                     "other": np.nan,
        }
        df['anesthetic_type'] = df['anesthetic_type'].map(anesthesia_map)
        print(f"\n  [Anesthetic Type]  NaN kept for fold-wise imputation")
        print(f"  NOTE: one-hot deferred to Stage 03 (train-fold categories only)")
        _print_freq(df['anesthetic_type'], "anesthetic_type after mapping")

        sep("9 — TEXT COLUMN")
        drop_text = [c for c in ['procedure', 'operative_dx', 'most_responsible_dx'] if c in df.columns]
        df.drop(columns=drop_text, inplace=True)
        print(f"  Retained : {TEXT_COL}  (encoded fold-wise in Stage 03)")
        print(f"  Dropped  : {drop_text}")
        print(f"\n  Sample values (first 5 rows):")
        for i in range(min(5, len(df))):
            print(f"    Row {i}: {str(df[TEXT_COL].iloc[i])[:120]}")

        sep("10 — FINAL DATASET SUMMARY")
        df.reset_index(drop=True, inplace=True)
        text_c   = [TEXT_COL]
        cat_c    = [c for c in df.columns if c in CATEGORICAL_FOLD_COLS + ['anesthetic_type']]
        struct_c = [c for c in df.columns if c not in text_c + cat_c + [TARGET]]
        print(f"\n  Rows          : {len(df):,}  (removed {n_init - len(df):,} = {(n_init-len(df))/n_init*100:.1f}%)")
        print(f"  Columns       : {df.shape[1]}")
        print(f"  Target        : {TARGET}")
        print(f"  Text column   : {text_c}  (encoded in Stage 03)")
        print(f"  Fold one-hot  : {cat_c}  (one-hot in Stage 03, train categories only)")
        print(f"  Structured    : {len(struct_c)}  {struct_c}")
        _print_missing(df, "final Clean table")

        sep("SAVE + FOLD INDEX GENERATION")
        with sqlite3.connect(DB_PATH) as conn:
            df.to_sql(CLEAN_TABLE, conn, if_exists='replace', index=False)
        print(f"  Saved '{CLEAN_TABLE}' → {DB_PATH}  shape={df.shape}")

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_rows = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(df)):
            for idx in tr_idx:
                fold_rows.append({'fold': int(fold), 'split': 'train', 'row_index': int(idx), 'case_id': df['case_id'].iloc[idx]})
            for idx in va_idx:
                fold_rows.append({'fold': int(fold), 'split': 'val', 'row_index': int(idx), 'case_id': df['case_id'].iloc[idx]})
        fold_df = pd.DataFrame(fold_rows)
        with sqlite3.connect(DB_PATH) as conn:
            fold_df.to_sql(FOLD_TABLE, conn, if_exists='replace', index=False)
        print(f"  KFold: n_splits={N_SPLITS}  shuffle=True  random_state={RANDOM_STATE}")
        print(f"\n  {'Fold':<6} {'Train':>12} {'Val':>12}")
        print(f"  {'-'*32}")
        for fold in range(N_SPLITS):
            nt = len(fold_df[(fold_df['fold']==fold) & (fold_df['split']=='train')])
            nv = len(fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')])
            print(f"  {fold:<6} {nt:>12,} {nv:>12,}")
        print(f"\n  ✅ Stage 01 complete.  Log → {LOG_DIR}/01_preprocessing.log")

    finally:
        tee.close()


# =============================================================================
# STAGE 02 — BERT CACHE  (unchanged)
# =============================================================================

def _s02_compute_clinicalbert(texts):
    import torch
    from transformers import AutoTokenizer, AutoModel
    print(f"  Loading {CLINICALBERT_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL_ID)
    model     = AutoModel.from_pretrained(CLINICALBERT_MODEL_ID)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device); print(f"  Device: {device}")
    embs      = []
    n_batches = (len(texts) + BERT_BATCH_CLINICAL - 1) // BERT_BATCH_CLINICAL
    for i in range(0, len(texts), BERT_BATCH_CLINICAL):
        batch = texts[i:i+BERT_BATCH_CLINICAL]
        with torch.no_grad():
            inp = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=BERT_MAX_LENGTH)
            inp = {k: v.to(device) for k, v in inp.items()}
            out = model(**inp)
            embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        if (i // BERT_BATCH_CLINICAL) % 50 == 0:
            print(f"    Batch {i//BERT_BATCH_CLINICAL+1}/{n_batches}  ({i:,}/{len(texts):,})")
    return np.vstack(embs)

def _s02_compute_sentencebert(texts):
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {SENTENCEBERT_MODEL_ID}...")
    model = SentenceTransformer(SENTENCEBERT_MODEL_ID)
    return model.encode(texts, show_progress_bar=True, batch_size=BERT_BATCH_SENTENCE)

def _s02_run_task(task_id):
    method, out_filename = S02_TASKS[task_id]
    out_path = os.path.join(BERT_DIR, out_filename)
    log_path = os.path.join(LOG_DIR, f"02_bert_task{task_id}_{method}.log")
    tee = _Tee(log_path); sys.stdout = tee
    try:
        sep(f"TASK {task_id} — {method.upper()} on '{TEXT_COL}'")
        print(f"  Output : {out_path}")
        if os.path.exists(out_path):
            os.remove(out_path); print(f"  Removed existing cache — recomputing fresh.")
        with sqlite3.connect(DB_PATH) as conn:
            df_txt = pd.read_sql(f"SELECT [{TEXT_COL}] FROM {CLEAN_TABLE}", conn)
        texts = df_txt[TEXT_COL].astype(str).tolist()
        print(f"  Loaded {len(texts):,} texts from '{TEXT_COL}'")
        t0   = time.time()
        embs = _s02_compute_clinicalbert(texts) if method == 'clinicalbert' else _s02_compute_sentencebert(texts)
        print(f"\n  Embedding shape : {embs.shape}")
        print(f"  Elapsed         : {(time.time()-t0)/60:.2f} min")
        np.save(out_path, embs)
        print(f"  ✅ Saved → {out_path}")
        print(f"  NOTE: Full {embs.shape[1]}-d embeddings stored. PCA applied per n in FEATURES_PER_COL={FEATURES_PER_COL} fold-wise in Stage 03.")
    finally:
        tee.close()

def run_stage02():
    sep("STAGE 02 — BERT CACHE")
    if all(_s02_task_is_done(tid) for tid in S02_TASKS):
        print(f"  ⏭  All BERT cache files already exist — skipping Stage 02.")
        for tid, (method, fname) in S02_TASKS.items():
            arr = np.load(os.path.join(BERT_DIR, fname))
            print(f"    Task {tid}  {fname:<52}  shape={arr.shape}")
        return
    print()
    for tid, (method, fname) in S02_TASKS.items():
        status = "✅ exists" if _s02_task_is_done(tid) else "❌ missing"
        print(f"  [{tid}]  {method:<16}  {fname:<52}  {status}")
    print()
    print("  [0]  Run ALL missing tasks  [default]")
    raw = input("  Select tasks (e.g. 1 or Enter for all missing): ").strip()
    if raw == '' or raw == '0':
        selected = [tid for tid in S02_TASKS if not _s02_task_is_done(tid)]
    else:
        selected = []
        for part in raw.split(','):
            part = part.strip()
            if part.isdigit() and int(part) in S02_TASKS:
                selected.append(int(part))
            else:
                print(f"  ⚠ Ignored: '{part}'")
    if not selected:
        print("  No tasks to run."); return
    for tid in selected:
        _s02_run_task(tid)
    sep("STAGE 02 COMPLETE")
    for tid in selected:
        _, fname = S02_TASKS[tid]
        arr = np.load(os.path.join(BERT_DIR, fname))
        print(f"  Task {tid}  {fname}  shape={arr.shape}")


# =============================================================================
# STAGE 03 — FOLD ENCODING
# =============================================================================

def _s03_init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encoded_matrices (
            fold       INTEGER,
            split      TEXT,
            encoding   TEXT,
            n_features INTEGER,
            rows       INTEGER,
            cols       INTEGER,
            dtype      TEXT,
            data       BLOB
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encoded_targets (
            fold  INTEGER,
            split TEXT,
            rows  INTEGER,
            dtype TEXT,
            data  BLOB
        )
    """)
    # Index for fast Stage 04 lookups — avoids full table scan per matrix load
    conn.execute("CREATE INDEX IF NOT EXISTS idx_enc ON encoded_matrices (fold, split, encoding, n_features)")
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
        print(f"    [{col}]  type={ptype}  train_NaN={n_tr:,}  val_NaN={n_va:,}")
        if n_tr == 0 and n_va == 0:
            print(f"      → No NaN, skipping."); continue
        num_feats = [c for c in train_df.columns
                     if c not in [col, TARGET, TEXT_COL]
                     and pd.api.types.is_numeric_dtype(train_df[c])]
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
            def predict_col(X, _le=le, _mdl=mdl):
                return _le.inverse_transform(np.round(_mdl.predict(X)).astype(int))
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

def _s03_onehot_fold(train_df, val_df, col, fold):
    train_df, val_df = train_df.copy(), val_df.copy()
    cats = sorted(train_df[col].dropna().unique())
    print(f"    [{col}]  fold {fold} train categories: {cats}")
    for cat in cats:
        cname = f'{col}__{cat}'
        train_df[cname] = (train_df[col] == cat).astype(int)
        val_df[cname]   = (val_df[col]   == cat).astype(int)
    train_df.drop(columns=[col], inplace=True)
    val_df.drop(columns=[col], inplace=True)
    return train_df, val_df

def _s03_onehot_all(train_df, val_df, fold):
    print(f"\n  Fold-wise one-hot (train categories only — no leakage):")
    for col in ['anesthetic_type'] + CATEGORICAL_FOLD_COLS:
        train_df, val_df = _s03_onehot_fold(train_df, val_df, col, fold)
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
    """PCA fit on train fold only — no leakage into val."""
    from sklearn.decomposition import PCA
    n_comp = min(n, emb_tr.shape[1], emb_tr.shape[0])
    pca    = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_bert_tr = pca.fit_transform(emb_tr)
    X_bert_va = pca.transform(emb_va)
    var_exp   = pca.explained_variance_ratio_.sum() * 100
    X_tr = np.hstack([X_struct_tr, X_bert_tr]).astype(np.float32)
    X_va = np.hstack([X_struct_va, X_bert_va]).astype(np.float32)
    return X_tr, X_va, n_comp, var_exp

def run_stage03():
    if _s03_is_done():
        print(f"  ⏭  Stage 03 already complete ({ENCODED_DB} has all expected matrices). Skipping.")
        with sqlite3.connect(ENCODED_DB) as conn:
            for row in conn.execute("SELECT fold,split,encoding,n_features,rows,cols FROM encoded_matrices ORDER BY fold,split,encoding,n_features"):
                print(f"    fold={row[0]} split={row[1]} encoding={row[2]:<20} n={row[3]:>3}  shape=({row[4]},{row[5]})")
        return

    tee = _Tee(f'{LOG_DIR}/03_fold_encoding.log')
    sys.stdout = tee
    try:
        sep("STAGE 03 — FOLD ENCODING")
        print(f"  FEATURES_PER_COL = {FEATURES_PER_COL}")
        print(f"  Classical encodings stored for EACH n value above.")
        print(f"  BERT PCA (train-only, no leakage) stored per fold here — no PCA recomputation in Stage 04.")

        if not _s01_is_done():
            print(f"  ❌ Stage 01 not complete — run Stage 01 first."); return

        # ── Load BERT caches (optional — skipped gracefully if missing) ───────
        bert_cache_s03 = {}
        sep("BERT CACHE CHECK")
        for tid, (method, fname) in S02_TASKS.items():
            path = os.path.join(BERT_DIR, fname)
            if os.path.exists(path):
                bert_cache_s03[method] = np.load(path)
                print(f"  ✅ {fname}  shape={bert_cache_s03[method].shape}")
            else:
                print(f"  ⚠ {fname} not found — BERT encodings skipped (run Stage 02 first)")
        if not bert_cache_s03:
            print(f"  ⚠ No BERT caches available — only classical encodings will be stored.")

        sep("LOAD CLEAN TABLE")
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(f"SELECT * FROM {CLEAN_TABLE}", conn)
        df = df[df[TARGET].notna()].copy().reset_index(drop=True)
        dropped_leak = [c for c in EXCLUDE_COLS if c in df.columns]
        df.drop(columns=dropped_leak, inplace=True)
        print(f"  Loaded   : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Excluded : {dropped_leak}  (intraoperative leakage)")

        if 'surgery_encounter_inpatient' in df.columns:
            nan_count = df['surgery_encounter_inpatient'].isna().sum()
            if nan_count > 0:
                print(f"  ⚠ WARNING: {nan_count} NaN surgery_encounter_inpatient found — should have been dropped in Stage 01. Dropping now but fold indices may be misaligned. Re-run Stage 01.")
                df = df[df['surgery_encounter_inpatient'].notna()].copy()
                df.reset_index(drop=True, inplace=True)

        print(f"  NaN (fold-imputed)    : { {col: int(df[col].isna().sum()) for col in IMPUTE_COLS if col in df.columns} }")
        print(f"  Categorical (fold OHE): {['anesthetic_type'] + CATEGORICAL_FOLD_COLS}")
        print(f"  Text column           : {TEXT_COL}")

        sep("LOAD FOLD INDICES")
        with sqlite3.connect(DB_PATH) as conn:
            fold_df = pd.read_sql(f"SELECT * FROM {FOLD_TABLE}", conn)
        fold_indices = {}
        for fold in range(N_SPLITS):
            tr_idx = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='train')]['row_index'].values
            va_idx = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')]['row_index'].values
            fold_indices[fold] = (tr_idx, va_idx)
            print(f"  Fold {fold}: train={len(tr_idx):,}  val={len(va_idx):,}")

        if os.path.exists(ENCODED_DB):
            os.remove(ENCODED_DB); print(f"\n  Removed existing {ENCODED_DB} — starting fresh.")
        with sqlite3.connect(ENCODED_DB) as conn:
            _s03_init_db(conn)

        sep("ENCODING LOOP — fold → impute → one-hot → encode (all n values) → save")
        for fold, (tr_idx, va_idx) in fold_indices.items():
            sep(f"FOLD {fold}  train={len(tr_idx):,}  val={len(va_idx):,}", char='-')
            train_base = df.iloc[tr_idx].copy().reset_index(drop=True)
            val_base   = df.iloc[va_idx].copy().reset_index(drop=True)

            train_base, val_base = _s03_impute_fold(train_base, val_base)
            train_base, val_base = _s03_onehot_all(train_base, val_base, fold)

            y_train = train_base[TARGET].values.astype(np.float64)
            y_val   = val_base[TARGET].values.astype(np.float64)
            print(f"\n  y_train: mean={y_train.mean():.1f}  std={y_train.std():.1f}  min={y_train.min():.1f}  max={y_train.max():.1f}")
            print(f"  y_val  : mean={y_val.mean():.1f}  std={y_val.std():.1f}  min={y_val.min():.1f}  max={y_val.max():.1f}")
            with sqlite3.connect(ENCODED_DB) as conn:
                _s03_save_target(conn, fold, 'train', y_train)
                _s03_save_target(conn, fold, 'val',   y_val)
                conn.commit()

            # ── only_structured (save once; also reused as struct base for BERT) ──
            X_tr_struct, X_va_struct = _s03_apply_encoding(train_base, val_base, 'only_structured', 0)
            print(f"\n  only_structured  train={X_tr_struct.shape}  val={X_va_struct.shape}  (stored once, no text features)")
            with sqlite3.connect(ENCODED_DB) as conn:
                _s03_save_matrix(conn, fold, 'train', 'only_structured', 0, X_tr_struct)
                _s03_save_matrix(conn, fold, 'val',   'only_structured', 0, X_va_struct)
                conn.commit()

            # ── Classical text encodings ──────────────────────────────────────
            print(f"\n  Classical text encodings (one entry per n value):")
            for n in FEATURES_PER_COL:
                for encoding in CLASSICAL_ENCODINGS_WITH_N:
                    X_tr, X_va = _s03_apply_encoding(train_base, val_base, encoding, n)
                    print(f"    n={n:>3}  {encoding:<8}  train={X_tr.shape}  val={X_va.shape}")
                    with sqlite3.connect(ENCODED_DB) as conn:
                        _s03_save_matrix(conn, fold, 'train', encoding, n, X_tr)
                        _s03_save_matrix(conn, fold, 'val',   encoding, n, X_va)
                        conn.commit()

            # ── BERT PCA encodings (fit PCA on train fold only) ───────────────
            if bert_cache_s03:
                print(f"\n  BERT PCA encodings (PCA fit on train fold only — no leakage):")
            for method, emb in bert_cache_s03.items():
                emb_tr = emb[tr_idx].astype(np.float32)
                emb_va = emb[va_idx].astype(np.float32)
                for n in FEATURES_PER_COL:
                    X_tr, X_va, n_comp, var_exp = _s03_encode_bert_pca(X_tr_struct, X_va_struct, emb_tr, emb_va, n)
                    print(f"    n={n:>3}  {method:<16}  PCA({n_comp}) on {emb_tr.shape[1]}-d → {var_exp:.1f}% var  train={X_tr.shape}  val={X_va.shape}")
                    with sqlite3.connect(ENCODED_DB) as conn:
                        _s03_save_matrix(conn, fold, 'train', method, n, X_tr)
                        _s03_save_matrix(conn, fold, 'val',   method, n, X_va)
                        conn.commit()

        sep("SUMMARY")
        with sqlite3.connect(ENCODED_DB) as conn:
            n_mat = conn.execute("SELECT COUNT(*) FROM encoded_matrices").fetchone()[0]
            n_tgt = conn.execute("SELECT COUNT(*) FROM encoded_targets").fetchone()[0]
            print(f"  encoded_matrices : {n_mat} rows")
            print(f"  encoded_targets  : {n_tgt} rows")
            print(f"\n  {'Fold':<6} {'Split':<8} {'Encoding':<20} {'n_features':>10} {'Rows':>8} {'Cols':>6}")
            print(f"  {'-'*62}")
            for row in conn.execute("SELECT fold,split,encoding,n_features,rows,cols FROM encoded_matrices ORDER BY fold,split,encoding,n_features"):
                print(f"  {row[0]:<6} {row[1]:<8} {row[2]:<20} {row[3]:>10} {row[4]:>8,} {row[5]:>6}")
        print(f"  DB size: {os.path.getsize(ENCODED_DB)/1024/1024:.1f} MB")
        print(f"  ✅ Stage 03 complete.  Log → {LOG_DIR}/03_fold_encoding.log")

    finally:
        tee.close()


# =============================================================================
# STAGE 04 — MODELING
# =============================================================================

def run_stage04():

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
    from tensorflow.keras.optimizers import AdamW   # fix: was Adam — weight_decay now actually applied
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.mixed_precision import set_global_policy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── TF GPU setup ──────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            set_global_policy('mixed_float16')
            print(f"  ✅ TF GPU: {[g.name for g in gpus]}  mixed_float16 enabled")
        except RuntimeError as e:
            print(f"  ⚠ TF GPU config error: {e}")
    else:
        print("  ⚠ No TF GPU detected — MLP will run on CPU")

    # ── XGBoost CUDA runtime check (falls back to CPU — no crash on CPU hosts) ──
    xgb_device = 'cpu'
    try:
        _probe = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1, verbosity=0)
        _probe.fit(np.zeros((10, 2)), np.zeros(10))
        xgb_device = 'cuda'
        print(f"  ✅ XGBoost CUDA available")
    except Exception:
        print(f"  ⚠ XGBoost CUDA unavailable — falling back to CPU")

    tee = _Tee(f'{LOG_DIR}/04_modeling.log')
    sys.stdout = tee
    try:
        sep("STAGE 04 — MODELING")
        sep("CONFIG SNAPSHOT")
        print(f"  FEATURES_PER_COL  = {FEATURES_PER_COL}  (all values in DB — no PCA in Stage 04)")
        print(f"  N_SPLITS          = {N_SPLITS}  /  RANDOM_STATE = {RANDOM_STATE}")
        print(f"  N_TRIALS          = {N_TRIALS}  /  TUNE_SET_FRACTION = {TUNE_SET_FRACTION}")
        print(f"  MLP: AdamW + BatchNorm + variable depth ({OPTUNA_MLP_N_LAYERS_LOW}-{OPTUNA_MLP_N_LAYERS_HIGH} layers, Optuna-tuned)")
        print(f"  MLP training: epochs={MLP_EPOCHS_FINAL}/{MLP_EPOCHS_OPTUNA}  batch={MLP_BATCH_SIZE}  clipnorm={MLP_CLIPNORM}")
        print(f"  EarlyStopping: patience={MLP_PATIENCE_ES}  |  ReduceLROnPlateau: factor={MLP_LR_DECAY_FACTOR}  patience={MLP_PATIENCE_LR}  min_lr={MLP_MIN_LR}")
        print(f"  MLP Optuna subset: {MLP_OPTUNA_SUBSET_SIZE if MLP_OPTUNA_SUBSET_SIZE else 'all rows'} rows per trial")
        print(f"  XGBoost: device={xgb_device}  early_stopping_rounds={XGB_EARLY_STOPPING_ROUNDS}")
        print(f"  RF max_features options: {OPTUNA_RF_MAX_FEATURES}")
        print(f"\n  Per-model artifacts saved to: {LOG_DIR}/<model_name>.log  and  {LOG_DIR}/<model_name>.pdf")

        # ── Verify prerequisites ──────────────────────────────────────────────
        sep("VERIFY PREREQUISITES")
        prereqs_ok = True
        if not _s01_is_done():
            print(f"  ❌ Stage 01 not complete."); prereqs_ok = False
        else:
            print(f"  ✅ {DB_PATH}")
        if not _s03_is_done():
            print(f"  ❌ Stage 03 not complete."); prereqs_ok = False
        else:
            print(f"  ✅ {ENCODED_DB}")
        if not prereqs_ok: return

        with sqlite3.connect(DB_PATH) as conn:
            fold_df = pd.read_sql(f"SELECT * FROM {FOLD_TABLE}", conn)
        fold_indices = {}
        for fold in range(N_SPLITS):
            tr_idx = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='train')]['row_index'].values
            va_idx = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')]['row_index'].values
            fold_indices[fold] = (tr_idx, va_idx)
        print(f"  ✅ Fold indices loaded ({N_SPLITS} folds)")

        # Detect which encodings are actually in the DB
        with sqlite3.connect(ENCODED_DB) as conn:
            db_combos = set(
                (row[0], row[1]) for row in
                conn.execute("SELECT DISTINCT encoding, n_features FROM encoded_matrices").fetchall()
            )
        print(f"  ✅ {len(db_combos)} (encoding, n) combos found in {ENCODED_DB}")

        # ── Model selection ───────────────────────────────────────────────────
        sep("MODEL SELECTION  (always asked — safe for multi-terminal use)")
        MODEL_EMOJIS = {
            'linear':       '📏 Linear Regression',
            'ridge':        '🏔️  Ridge Regression',
            'lasso':        '🪢 Lasso Regression',
            'randomforest': '🌲 Random Forest',
            'xgboost':      '⚡ XGBoost',
            'mlp':          '🧠 MLP Neural Network',
        }
        print(f"  [0]  🚀 Run ALL models  [default]")
        for i, m in enumerate(ALL_MODELS, 1):
            print(f"  [{i}]  {MODEL_EMOJIS[m]}")
        print()
        user_input = input("  Select models (e.g. 1,3,5 or Enter for all): ").strip()
        if user_input == '' or user_input == '0':
            MODEL_LIST = ALL_MODELS[:]
        else:
            MODEL_LIST = []
            for part in user_input.split(','):
                part = part.strip()
                if part.isdigit() and 1 <= int(part) <= len(ALL_MODELS):
                    MODEL_LIST.append(ALL_MODELS[int(part)-1])
                else:
                    print(f"  ⚠ Ignored: '{part}'")
            MODEL_LIST = list(dict.fromkeys(MODEL_LIST))
        if not MODEL_LIST:
            print("  No valid models selected — exiting."); return
        print(f"\n  Running: {MODEL_LIST}")

        # ── Inner helpers ─────────────────────────────────────────────────────

        def load_fold_matrices(fold):
            """Load all encoded matrices for a given fold in a single DB connection."""
            result = {}
            with sqlite3.connect(ENCODED_DB) as conn:
                rows = conn.execute(
                    "SELECT split, encoding, n_features, rows, cols, dtype, data FROM encoded_matrices WHERE fold=?",
                    (fold,)
                ).fetchall()
            for split, encoding, n_features, n_rows, n_cols, dtype, data in rows:
                result[(split, encoding, int(n_features))] = (
                    np.frombuffer(data, dtype=dtype).reshape(n_rows, n_cols).copy()
                )
            return result

        def load_target(fold, split):
            with sqlite3.connect(ENCODED_DB) as conn:
                row = conn.execute(
                    "SELECT rows,dtype,data FROM encoded_targets WHERE fold=? AND split=?",
                    (fold, split)
                ).fetchone()
            if row is None: raise ValueError(f"Target not found: fold={fold} split={split}")
            return np.frombuffer(row[2], dtype=row[1]).copy()

        def save_db(df_save, table):
            with sqlite3.connect(RESULT_DB, timeout=30) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                df_save.to_sql(table, conn, if_exists='append', index=False)

        result_db_is_new = not os.path.exists(RESULT_DB)

        def delete_existing(fold, encoding, n_features, model_name):
            if result_db_is_new: return   # skip on first run — no rows to delete
            with sqlite3.connect(RESULT_DB, timeout=30) as conn:
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
            return {'mse': mse, 'rmse': np.sqrt(mse), 'mae': mae, 'r2': r2, 'smape': smape,
                    'mean_error': mu, 'std_error': sd, 'ci95_low': ci[0], 'ci95_high': ci[1]}

        def build_model(name, params=None):
            p = params or {}
            if name == 'linear':
                return LinearRegression()
            if name == 'ridge':
                return Ridge(alpha=p.get('alpha', 1.0))
            if name == 'lasso':
                return Lasso(alpha=p.get('alpha', 1.0))
            if name == 'randomforest':
                return RandomForestRegressor(
                    n_estimators=p.get('n_estimators', 200),
                    max_depth=p.get('max_depth', 6),
                    max_features=p.get('max_features', 'sqrt'),
                    min_samples_split=p.get('min_samples_split', 2),
                    min_samples_leaf=p.get('min_samples_leaf', 1),
                    random_state=RANDOM_STATE, n_jobs=-1
                )
            if name == 'xgboost':
                return xgb.XGBRegressor(
                    n_estimators=p.get('n_estimators', 200),
                    learning_rate=p.get('learning_rate', 0.05),
                    max_depth=p.get('max_depth', 4),
                    subsample=p.get('subsample', 0.8),
                    colsample_bytree=p.get('colsample_bytree', 0.8),
                    reg_alpha=p.get('reg_alpha', 0.01),
                    reg_lambda=p.get('reg_lambda', 1.0),
                    tree_method=XGB_TREE_METHOD,
                    device=xgb_device,              # runtime-detected, not hardcoded
                    early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,  # constructor in XGBoost >=2.0
                    random_state=RANDOM_STATE, n_jobs=-1
                )
            if name == 'mlp':
                act      = p.get('activation', 'relu')
                lr       = p.get('lr', 1e-3)
                wd       = p.get('weight_decay', 1e-4)
                n_layers = p.get('n_layers', 2)
                opt = AdamW(learning_rate=lr, weight_decay=wd,
                            clipnorm=MLP_CLIPNORM if MLP_CLIPNORM is not None else None)
                layer_sizes = [OPTUNA_MLP_UNITS1_HIGH, OPTUNA_MLP_UNITS2_HIGH, OPTUNA_MLP_UNITS3_HIGH]
                layer_list  = [Input(shape=(p['input_dim'],))]
                for i in range(n_layers):
                    units   = p.get(f'units{i+1}', layer_sizes[i] // 2)
                    dropout = p.get(f'dropout{i+1}', 0.1)
                    layer_list.append(Dense(units, activation=act))
                    layer_list.append(BatchNormalization())  # stabilises mixed-scale tabular inputs
                    layer_list.append(Dropout(dropout))
                layer_list.append(Dense(1, dtype='float32'))
                mdl = Sequential(layer_list)
                mdl.compile(loss='mse', optimizer=opt)
                return mdl
            raise ValueError(f"Unknown model: {name}")

        # ── fit_predict — returns (preds, train_time_s, infer_time_s) ─────────
        def fit_predict(name, mdl, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva=None, final=False):
            t_train_start = time.perf_counter()

            if name == 'mlp':
                Xtr_f = Xtr_sc.astype(np.float32); ytr_f = ytr.astype(np.float32)
                if final and yva is not None:
                    Xva_f = Xva_sc.astype(np.float32); yva_f = yva.astype(np.float32)
                    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr_f, ytr_f))
                                .shuffle(min(10_000, len(ytr_f)), seed=RANDOM_STATE)
                                .batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    val_ds   = (tf.data.Dataset.from_tensor_slices((Xva_f, yva_f))
                                .batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    cbs = [
                        EarlyStopping(monitor='val_loss', patience=MLP_PATIENCE_ES, restore_best_weights=True, verbose=0),
                        ReduceLROnPlateau(monitor='val_loss', factor=MLP_LR_DECAY_FACTOR, patience=MLP_PATIENCE_LR, min_lr=MLP_MIN_LR, verbose=0),
                    ]
                    mdl.fit(train_ds, validation_data=val_ds, epochs=MLP_EPOCHS_FINAL, callbacks=cbs, verbose=0)
                    train_time_s = time.perf_counter() - t_train_start
                    t_infer_start = time.perf_counter()
                    preds = mdl.predict(val_ds, verbose=0).flatten()
                    infer_time_s = time.perf_counter() - t_infer_start
                else:
                    mdl.fit(Xtr_f, ytr_f, epochs=MLP_EPOCHS_OPTUNA, batch_size=MLP_BATCH_SIZE, verbose=0)
                    train_time_s = time.perf_counter() - t_train_start
                    t_infer_start = time.perf_counter()
                    preds = mdl.predict(Xva_sc.astype(np.float32), verbose=0).flatten()
                    infer_time_s = time.perf_counter() - t_infer_start

            elif name == 'xgboost':
                # early_stopping_rounds is set in the constructor (XGBoost >=2.0).
                # eval_set must be provided to fit() for early stopping to activate;
                # fall back to plain fit() if no val labels are available.
                if yva is not None:
                    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                else:
                    mdl.fit(Xtr, ytr)
                train_time_s = time.perf_counter() - t_train_start
                t_infer_start = time.perf_counter()
                preds = mdl.predict(Xva)
                infer_time_s = time.perf_counter() - t_infer_start

            elif name == 'randomforest':
                mdl.fit(Xtr, ytr)
                train_time_s = time.perf_counter() - t_train_start
                t_infer_start = time.perf_counter()
                preds = mdl.predict(Xva)
                infer_time_s = time.perf_counter() - t_infer_start

            else:  # linear, ridge, lasso
                mdl.fit(Xtr_sc, ytr)
                train_time_s = time.perf_counter() - t_train_start
                t_infer_start = time.perf_counter()
                preds = mdl.predict(Xva_sc)
                infer_time_s = time.perf_counter() - t_infer_start

            return preds, train_time_s, infer_time_s

        class _KerasWrapper(BaseEstimator):
            def __init__(self, mdl): self.mdl = mdl
            def fit(self, X, y):
                self.mdl.fit(X.astype(np.float32), y.astype(np.float32),
                             epochs=MLP_EPOCHS_OPTUNA, batch_size=MLP_BATCH_SIZE, verbose=0)
                return self
            def predict(self, X):
                return self.mdl.predict(X.astype(np.float32), verbose=0).flatten()

        # ── Optuna pruning callback for MLP ───────────────────────────────────
        # Reads val_loss directly from Keras logs (no extra predict() call per epoch).
        # Raises TrialPruned inline when the MedianPruner decides to stop the trial.
        class _OptunaPruningCallback(tf.keras.callbacks.Callback):
            def __init__(self, trial):
                super().__init__()
                self._trial = trial
            def on_epoch_end(self, epoch, logs=None):
                val_loss = (logs or {}).get('val_loss')
                if val_loss is None:
                    return
                self._trial.report(float(val_loss), epoch)
                if self._trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        def make_objective(name, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva):
            def obj(trial):
                if name in ('ridge', 'lasso'):
                    p = {'alpha': trial.suggest_float('alpha', OPTUNA_ALPHA_LOW, OPTUNA_ALPHA_HIGH, log=True)}
                elif name == 'randomforest':
                    p = {
                        'n_estimators':      trial.suggest_int('n_estimators', OPTUNA_RF_N_EST_LOW, OPTUNA_RF_N_EST_HIGH),
                        'max_depth':         trial.suggest_int('max_depth', OPTUNA_RF_MAX_DEPTH_LOW, OPTUNA_RF_MAX_DEPTH_HIGH),
                        'max_features':      trial.suggest_categorical('max_features', OPTUNA_RF_MAX_FEATURES),
                        'min_samples_split': trial.suggest_int('min_samples_split', OPTUNA_RF_MIN_SAMPLES_SPLIT_LOW, OPTUNA_RF_MIN_SAMPLES_SPLIT_HIGH),
                        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', OPTUNA_RF_MIN_SAMPLES_LEAF_LOW, OPTUNA_RF_MIN_SAMPLES_LEAF_HIGH),
                        'random_state': RANDOM_STATE, 'n_jobs': -1,
                    }
                elif name == 'xgboost':
                    p = {
                        'n_estimators':     trial.suggest_int('n_estimators', OPTUNA_XGB_N_EST_LOW, OPTUNA_XGB_N_EST_HIGH),
                        'learning_rate':    trial.suggest_float('learning_rate', OPTUNA_XGB_LR_LOW, OPTUNA_XGB_LR_HIGH, log=True),
                        'max_depth':        trial.suggest_int('max_depth', OPTUNA_XGB_MAX_DEPTH_LOW, OPTUNA_XGB_MAX_DEPTH_HIGH),
                        'subsample':        trial.suggest_float('subsample', OPTUNA_XGB_SUBSAMPLE_LOW, OPTUNA_XGB_SUBSAMPLE_HIGH),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', OPTUNA_XGB_COLSAMPLE_LOW, OPTUNA_XGB_COLSAMPLE_HIGH),
                        'reg_alpha':        trial.suggest_float('reg_alpha', OPTUNA_XGB_REG_ALPHA_LOW, OPTUNA_XGB_REG_ALPHA_HIGH, log=True),
                        'reg_lambda':       trial.suggest_float('reg_lambda', OPTUNA_XGB_REG_LAMBDA_LOW, OPTUNA_XGB_REG_LAMBDA_HIGH, log=True),
                        'tree_method': XGB_TREE_METHOD, 'device': xgb_device,
                        'early_stopping_rounds': XGB_EARLY_STOPPING_ROUNDS,  # constructor in XGBoost >=2.0
                        'random_state': RANDOM_STATE, 'n_jobs': -1,
                    }
                elif name == 'mlp':
                    tf.keras.backend.clear_session()
                    n_layers = trial.suggest_int('n_layers', OPTUNA_MLP_N_LAYERS_LOW, OPTUNA_MLP_N_LAYERS_HIGH)
                    units_bounds = [
                        (OPTUNA_MLP_UNITS1_LOW, OPTUNA_MLP_UNITS1_HIGH),
                        (OPTUNA_MLP_UNITS2_LOW, OPTUNA_MLP_UNITS2_HIGH),
                        (OPTUNA_MLP_UNITS3_LOW, OPTUNA_MLP_UNITS3_HIGH),
                    ]
                    p = {
                        'input_dim':    Xtr_sc.shape[1],
                        'n_layers':     n_layers,
                        'activation':   trial.suggest_categorical('activation', OPTUNA_MLP_ACTIVATIONS),
                        'lr':           trial.suggest_float('lr', OPTUNA_MLP_LR_LOW, OPTUNA_MLP_LR_HIGH, log=True),
                        'weight_decay': trial.suggest_float('weight_decay', OPTUNA_MLP_WEIGHT_DECAY_LOW, OPTUNA_MLP_WEIGHT_DECAY_HIGH, log=True),
                    }
                    # Always suggest ALL layer slots up to OPTUNA_MLP_N_LAYERS_HIGH.
                    # This makes the search space static so TPESampler(multivariate=True)
                    # can model all parameters jointly — eliminates the RandomSampler
                    # fallback warning. build_model() ignores slots beyond n_layers
                    # because it iterates range(n_layers) only.
                    for i in range(OPTUNA_MLP_N_LAYERS_HIGH):
                        lo_u, hi_u = units_bounds[i]
                        p[f'units{i+1}']   = trial.suggest_int(f'units{i+1}', lo_u, hi_u)
                        p[f'dropout{i+1}'] = trial.suggest_float(f'dropout{i+1}', OPTUNA_MLP_DROPOUT_LOW, OPTUNA_MLP_DROPOUT_HIGH)
                else:
                    p = {}

                if name == 'mlp':
                    mdl   = build_model('mlp', p)
                    Xtr_f = Xtr_sc.astype(np.float32)
                    ytr_f = ytr.astype(np.float32)
                    Xva_f = Xva_sc.astype(np.float32)
                    yva_f = yva.astype(np.float32)
                    # Cap rows per Optuna trial for speed on large datasets.
                    # A random subsample is sufficient to rank architectures;
                    # the final fit always uses the full training fold.
                    if MLP_OPTUNA_SUBSET_SIZE and len(Xtr_f) > MLP_OPTUNA_SUBSET_SIZE:
                        rng   = np.random.default_rng(RANDOM_STATE)
                        idx   = rng.choice(len(Xtr_f), MLP_OPTUNA_SUBSET_SIZE, replace=False)
                        Xtr_f = Xtr_f[idx]
                        ytr_f = ytr_f[idx]
                    # Single .fit() call with validation_data — eliminates the old
                    # per-epoch fit(epochs=1) + predict() loop overhead (was 1500
                    # separate TF graph entries across 50 trials × 30 epochs).
                    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr_f, ytr_f))
                                .shuffle(min(10_000, len(ytr_f)), seed=RANDOM_STATE)
                                .batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    val_ds   = (tf.data.Dataset.from_tensor_slices((Xva_f, yva_f))
                                .batch(MLP_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
                    history = mdl.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=MLP_EPOCHS_OPTUNA,
                        callbacks=[_OptunaPruningCallback(trial)],
                        verbose=0,
                    )
                    # Return best val_loss seen (consistent with restore_best_weights
                    # in the final fit — avoids penalising trials that overshot late).
                    return float(min(history.history.get('val_loss', [np.inf])))
                else:
                    mdl = build_model(name, p)
                    # Pass yva so XGBoost uses early stopping inside fit_predict
                    preds, _, _ = fit_predict(name, mdl, Xtr, Xtr_sc, Xva, Xva_sc, ytr, yva=yva)
                    return mean_squared_error(yva, preds)
            return obj

        # ── per-model log + PDF ───────────────────────────────────────────────
        def save_model_artifacts(model_name, model_rows):
            df_m = pd.DataFrame(model_rows)
            grp  = df_m.groupby(['encoding', 'n_features'])[['mae', 'smape', 'r2', 'rmse', 'train_time_s', 'infer_time_s']].agg(['mean', 'std'])
            grp.columns = ['_'.join(c) for c in grp.columns]
            grp = grp.reset_index()

            log_path = os.path.join(LOG_DIR, f'{model_name}.log')
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*80}\n  MODEL: {model_name.upper()}\n{'='*80}\n\n")
                f.write("PER-FOLD RESULTS\n")
                f.write(f"  {'Encoding':<18} {'n':>6} {'Fold':>5} {'MAE':>8} {'SMAPE':>8} {'R²':>8} {'RMSE':>8} {'Train(s)':>10} {'Infer(s)':>10}\n")
                f.write(f"  {'-'*87}\n")
                for _, row in df_m.sort_values(['encoding', 'n_features', 'fold']).iterrows():
                    n_str = str(int(row['n_features'])) if row['n_features'] > 0 else 'struct'
                    f.write(f"  {row['encoding']:<18} {n_str:>6} {int(row['fold']):>5} {row['mae']:>8.3f} {row['smape']:>8.3f} {row['r2']:>8.4f} {row['rmse']:>8.3f} {row['train_time_s']:>10.2f} {row['infer_time_s']:>10.4f}\n")
                f.write(f"\n\nAGGREGATED SUMMARY  (mean ± std across {N_SPLITS} folds)\n")
                f.write(f"  {'Encoding':<18} {'n':>6} {'MAE mean':>10} {'±':>4} {'SMAPE mean':>12} {'±':>4} {'R² mean':>9} {'±':>4} {'Train(s)':>10} {'Infer(s)':>10}\n")
                f.write(f"  {'-'*97}\n")
                for _, row in grp.sort_values('mae_mean').iterrows():
                    n_str = str(int(row['n_features'])) if row['n_features'] > 0 else 'struct'
                    f.write(f"  {row['encoding']:<18} {n_str:>6} {row['mae_mean']:>10.3f} {row['mae_std']:>4.3f} {row['smape_mean']:>12.3f} {row['smape_std']:>4.3f} {row['r2_mean']:>9.4f} {row['r2_std']:>4.4f} {row['train_time_s_mean']:>10.2f} {row['infer_time_s_mean']:>10.4f}\n")
            print(f"  📝 {model_name}.log  → {log_path}")

            plots_cfg = [
                ('mae_mean',          'mae_std',          'MAE (minutes)',  'MAE',        False),
                ('smape_mean',        'smape_std',        'SMAPE (%)',      'SMAPE',      False),
                ('r2_mean',           'r2_std',           'R²',             'R²',         True),
                ('train_time_s_mean', 'train_time_s_std', 'Train Time (s)', 'Train Time', False),
            ]

            enc_order = ['only_structured'] + CLASSICAL_ENCODINGS_WITH_N + BERT_ENCODINGS
            enc_order = [e for e in enc_order if e in grp['encoding'].unique()]
            n_order   = [0] + FEATURES_PER_COL
            n_order   = [n for n in n_order if n in grp['n_features'].unique()]

            enc_palette = {enc: plt.cm.tab10.colors[i % 10] for i, enc in enumerate(enc_order)}
            n_labels    = {n: ('struct' if n == 0 else str(n)) for n in n_order}
            n_palette   = {n: plt.cm.Set2.colors[i % 8]       for i, n in enumerate(n_order)}

            lookup = {(r['encoding'], int(r['n_features'])): r for _, r in grp.iterrows()}

            def _get(enc, n, col_mean, col_std):
                row = lookup.get((enc, int(n)))
                if row is None: return (np.nan, np.nan)
                return (row[col_mean], row[col_std])

            pdf_path = os.path.join(LOG_DIR, f'{model_name}.pdf')
            with PdfPages(pdf_path) as pdf:

                # Page 1: grouped by encoding, coloured by n
                n_groups_p1 = len(enc_order)
                n_series_p1 = len(n_order)
                bar_w_p1    = min(0.8 / max(n_series_p1, 1), 0.25)
                offsets_p1  = (np.arange(n_series_p1) - (n_series_p1 - 1) / 2.0) * bar_w_p1

                fig1, axes1 = plt.subplots(2, 2, figsize=(max(14, n_groups_p1 * 1.8), 12))
                fig1.suptitle(f'Model: {model_name.upper()}  —  Page 1: Grouped by Encoding Method\n(colours = feature count n;  error bars = ±1 std;  red border = best per metric)', fontsize=11, fontweight='bold', y=0.98)
                legend_handles_p1 = [plt.Rectangle((0, 0), 1, 1, color=n_palette[nk], alpha=0.85, label=f'n={n_labels[nk]}') for nk in n_order]
                fig1.legend(handles=legend_handles_p1, loc='upper center', ncol=len(n_order), fontsize=10, frameon=True, title='Feature count (n)', title_fontsize=10, bbox_to_anchor=(0.5, 0.96))

                for ax, (mean_col, std_col, ylabel, title, higher_better) in zip(axes1.flat, plots_cfg):
                    best_val, best_bar = None, None
                    for si, nk in enumerate(n_order):
                        means = np.array([_get(ek, nk, mean_col, std_col)[0] for ek in enc_order])
                        stds  = np.array([_get(ek, nk, mean_col, std_col)[1] for ek in enc_order])
                        x_pos = np.arange(n_groups_p1) + offsets_p1[si]
                        valid = ~np.isnan(means)
                        if not valid.any(): continue
                        bars = ax.bar(x_pos[valid], means[valid], width=bar_w_p1 * 0.92, yerr=stds[valid], capsize=2, color=n_palette[nk], alpha=0.85, edgecolor='black', linewidth=0.4, error_kw={'elinewidth': 0.9})
                        for bar, val in zip(bars, means[valid]):
                            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.005, f'{val:.2f}', ha='center', va='bottom', fontsize=5.5, rotation=90)
                        for bar, val in zip(bars, means[valid]):
                            if best_val is None or (higher_better and val > best_val) or (not higher_better and val < best_val):
                                best_val, best_bar = val, bar
                    if best_bar is not None:
                        best_bar.set_edgecolor('red'); best_bar.set_linewidth(2.0)
                    ax.set_xticks(np.arange(n_groups_p1))
                    ax.set_xticklabels(enc_order, fontsize=9, rotation=25, ha='right')
                    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11, fontweight='bold')
                    ax.grid(axis='y', linestyle='--', alpha=0.4)
                plt.tight_layout(rect=[0, 0, 1, 0.91]); pdf.savefig(fig1, bbox_inches='tight'); plt.close(fig1)

                # Page 2: grouped by n, coloured by encoding
                n_groups_p2 = len(n_order)
                n_series_p2 = len(enc_order)
                bar_w_p2    = min(0.8 / max(n_series_p2, 1), 0.25)
                offsets_p2  = (np.arange(n_series_p2) - (n_series_p2 - 1) / 2.0) * bar_w_p2
                n_xlabels   = [f'n={n_labels[nk]}' for nk in n_order]

                fig2, axes2 = plt.subplots(2, 2, figsize=(max(14, n_groups_p2 * 2.2), 12))
                fig2.suptitle(f'Model: {model_name.upper()}  —  Page 2: Grouped by Feature Count (n)\n(colours = encoding method;  error bars = ±1 std;  red border = best per metric)', fontsize=11, fontweight='bold', y=0.98)
                legend_handles_p2 = [plt.Rectangle((0, 0), 1, 1, color=enc_palette[ek], alpha=0.85, label=ek) for ek in enc_order]
                fig2.legend(handles=legend_handles_p2, loc='upper center', ncol=min(len(enc_order), 6), fontsize=10, frameon=True, title='Encoding method', title_fontsize=10, bbox_to_anchor=(0.5, 0.96))

                for ax, (mean_col, std_col, ylabel, title, higher_better) in zip(axes2.flat, plots_cfg):
                    best_val, best_bar = None, None
                    for si, ek in enumerate(enc_order):
                        means = np.array([_get(ek, nk, mean_col, std_col)[0] for nk in n_order])
                        stds  = np.array([_get(ek, nk, mean_col, std_col)[1] for nk in n_order])
                        x_pos = np.arange(n_groups_p2) + offsets_p2[si]
                        valid = ~np.isnan(means)
                        if not valid.any(): continue
                        bars = ax.bar(x_pos[valid], means[valid], width=bar_w_p2 * 0.92, yerr=stds[valid], capsize=2, color=enc_palette[ek], alpha=0.85, edgecolor='black', linewidth=0.4, error_kw={'elinewidth': 0.9})
                        for bar, val in zip(bars, means[valid]):
                            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.005, f'{val:.2f}', ha='center', va='bottom', fontsize=5.5, rotation=90)
                        for bar, val in zip(bars, means[valid]):
                            if best_val is None or (higher_better and val > best_val) or (not higher_better and val < best_val):
                                best_val, best_bar = val, bar
                    if best_bar is not None:
                        best_bar.set_edgecolor('red'); best_bar.set_linewidth(2.0)
                    ax.set_xticks(np.arange(n_groups_p2))
                    ax.set_xticklabels(n_xlabels, fontsize=10, rotation=0, ha='center')
                    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11, fontweight='bold')
                    ax.grid(axis='y', linestyle='--', alpha=0.4)
                plt.tight_layout(rect=[0, 0, 1, 0.91]); pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)

            print(f"  📊 {model_name}.pdf  → {pdf_path}  (2 pages)")

        # ── Main fold loop ────────────────────────────────────────────────────
        sep("MAIN LOOP — fold → load encodings (all n values) → model")
        all_metrics   = []
        model_buckets = {m: [] for m in MODEL_LIST}

        for fold in range(N_SPLITS):
            train_idx, val_idx = fold_indices[fold]
            sep(f"FOLD {fold}  train={len(train_idx):,}  val={len(val_idx):,}", char='-')

            y_train = load_target(fold, 'train')
            y_val   = load_target(fold, 'val')
            print(f"  y_train: mean={y_train.mean():.1f}  std={y_train.std():.1f}  min={y_train.min():.1f}  max={y_train.max():.1f}")
            print(f"  y_val  : mean={y_val.mean():.1f}  std={y_val.std():.1f}  min={y_val.min():.1f}  max={y_val.max():.1f}")

            # Single DB connection per fold — avoids N open/close cycles
            print(f"\n  Loading all matrices for fold {fold} (one DB connection)...")
            raw_mats = load_fold_matrices(fold)

            # Build fold_encoded: (enc, n) → (X_tr, X_tr_sc, X_va, X_va_sc)
            fold_encoded = {}
            for (split, enc, n), arr in raw_mats.items():
                if split == 'train':
                    fold_encoded.setdefault((enc, n), {})['train'] = arr
                else:
                    fold_encoded.setdefault((enc, n), {})['val'] = arr

            # Scale each combo (scaler fit on train only)
            for key in list(fold_encoded.keys()):
                d = fold_encoded[key]
                if 'train' not in d or 'val' not in d:
                    del fold_encoded[key]; continue
                scaler  = MinMaxScaler()
                X_tr_sc = scaler.fit_transform(d['train'])
                X_va_sc = scaler.transform(d['val'])
                fold_encoded[key] = (d['train'], X_tr_sc, d['val'], X_va_sc)

            # Report what was loaded
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

                for model_name in MODEL_LIST:
                    delete_existing(fold, encoding, n, model_name)
                    print(f"    🔧 {model_name}")
                    try:
                        best_p = {}
                        if model_name != 'linear':
                            idx_all  = np.arange(len(y_train))
                            idx_tr2, idx_tune = train_test_split(idx_all, test_size=TUNE_SET_FRACTION, random_state=RANDOM_STATE)
                            X_tr2,  X_tune  = X_train[idx_tr2],  X_train[idx_tune]
                            y_tr2,  y_tune  = y_train[idx_tr2],  y_train[idx_tune]
                            # Refit scaler on sub-train only — fixes alpha leakage for Ridge/Lasso
                            sc_inner   = MinMaxScaler()
                            X_tr2_sc   = sc_inner.fit_transform(X_tr2)
                            X_tune_sc  = sc_inner.transform(X_tune)

                            sampler = optuna.samplers.TPESampler(
                                n_startup_trials=OPTUNA_N_STARTUP_TRIALS,
                                multivariate=True,
                                seed=RANDOM_STATE,
                            )
                            pruner = optuna.pruners.MedianPruner(
                                n_startup_trials=OPTUNA_N_STARTUP_TRIALS,
                                n_warmup_steps=5,
                            )
                            study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
                            study.optimize(
                                make_objective(model_name, X_tr2, X_tr2_sc, X_tune, X_tune_sc, y_tr2, y_tune),
                                n_trials=N_TRIALS
                            )
                            best_p = study.best_trial.params
                            if model_name == 'mlp':
                                best_p['input_dim'] = X_train_sc.shape[1]
                                tf.keras.backend.clear_session()
                            print(f"      Optuna best MSE={study.best_trial.value:.2f}  params={best_p}")

                        mdl = build_model(model_name, best_p)
                        preds, train_time_s, infer_time_s = fit_predict(
                            model_name, mdl,
                            X_train, X_train_sc, X_val, X_val_sc,
                            y_train, yva=y_val, final=True
                        )

                        metrics = compute_metrics(y_val, preds)
                        metrics.update({
                            'fold':         fold,
                            'encoding':     encoding,
                            'n_features':   n,
                            'model':        model_name,
                            'train_time_s': round(train_time_s, 4),
                            'infer_time_s': round(infer_time_s, 4),
                        })
                        all_metrics.append(metrics)
                        model_buckets[model_name].append(metrics)
                        print(f"      ✅ MAE={metrics['mae']:>7.2f}  SMAPE={metrics['smape']:>6.2f}%  R²={metrics['r2']:>6.4f}  RMSE={metrics['rmse']:>7.2f}  train={train_time_s:.2f}s  infer={infer_time_s:.4f}s")

                        save_db(pd.DataFrame([metrics]), 'metrics')

                        val_case_ids = fold_df[(fold_df['fold']==fold) & (fold_df['split']=='val')]['case_id'].values
                        pred_df = pd.DataFrame({
                            'fold':       fold,
                            'encoding':   encoding,
                            'n_features': n,
                            'model':      model_name,
                            'case_id':    val_case_ids,
                            'actual':     y_val,
                            'predicted':  preds,
                        })
                        save_db(pred_df, 'predictions')

                        # Feature importance — gradient saliency for MLP (fast),
                        # native importances for tree models, coefficients for linear
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

                        save_db(pd.DataFrame([{'fold': fold, 'encoding': encoding, 'n_features': n, 'model': model_name, 'importances': str(imps)}]), 'feature_importance')
                        params_clean = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in best_p.items()}
                        save_db(pd.DataFrame([{'fold': fold, 'encoding': encoding, 'n_features': n, 'model': model_name, 'params': str(params_clean)}]), 'hyperparameter')

                    except Exception as e:
                        print(f"      🚫 Error: {e}")

        # ── Per-model .log and .pdf ───────────────────────────────────────────
        sep("PER-MODEL ARTIFACTS  (.log + .pdf)")
        for model_name in MODEL_LIST:
            rows = model_buckets[model_name]
            if rows:
                print(f"\n  Generating artifacts for: {model_name}")
                save_model_artifacts(model_name, rows)
            else:
                print(f"\n  ⚠ No results for {model_name} — skipping artifacts.")

        # ── Final aggregated summary ──────────────────────────────────────────
        sep("FINAL AGGREGATED SUMMARY  (mean ± std across folds)")
        if all_metrics:
            df_m    = pd.DataFrame(all_metrics)
            grouped = df_m.groupby(['encoding', 'n_features', 'model'])[['mae', 'smape', 'r2', 'rmse', 'train_time_s', 'infer_time_s']].agg(['mean', 'std'])
            grouped.columns = ['_'.join(c) for c in grouped.columns]
            grouped = grouped.reset_index().sort_values('mae_mean')
            print(f"\n  {'Encoding':<18} {'n':>5} {'Model':<14} {'MAE mean':>10} {'MAE std':>9} {'SMAPE mean':>12} {'R² mean':>9} {'RMSE mean':>11} {'Train(s)':>10} {'Infer(s)':>10}")
            print(f"  {'-'*110}")
            for _, row in grouped.iterrows():
                n_str = str(int(row['n_features'])) if row['n_features'] > 0 else 'struct'
                print(f"  {row['encoding']:<18} {n_str:>5} {row['model']:<14} {row['mae_mean']:>10.3f} {row['mae_std']:>9.3f} {row['smape_mean']:>12.3f} {row['r2_mean']:>9.4f} {row['rmse_mean']:>11.3f} {row['train_time_s_mean']:>10.2f} {row['infer_time_s_mean']:>10.4f}")

            sep("BEST CONFIGURATION PER METRIC")
            for metric, direction in [('mae_mean', 'min'), ('smape_mean', 'min'), ('r2_mean', 'max'), ('train_time_s_mean', 'min'), ('infer_time_s_mean', 'min')]:
                idx  = grouped[metric].idxmin() if direction == 'min' else grouped[metric].idxmax()
                best = grouped.loc[idx]
                n_str = str(int(best['n_features'])) if best['n_features'] > 0 else 'struct_only'
                print(f"  Best {metric:<22}: encoding={best['encoding']}  n_features={n_str}  model={best['model']}  value={best[metric]:.4f}")

        print(f"\n  Results → {RESULT_DB}")
        print(f"  Log     → {LOG_DIR}/04_modeling.log")
        print(f"  ✅ Stage 04 complete.")

    finally:
        tee.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    sep("pipeline.py  —  Stages 01 · 02 · 03 · 04")
    print()
    print("  Stages 01–03 auto-skip if already complete.")
    print("  Stage 04 always asks which model(s) to run.")
    print("  All settings are in the CONFIG block at the top of this file.")
    print()
    print("  [1]  Stage 01 — Pre-processing     CSV → surgical_data.db + fold indices")
    print("  [2]  Stage 02 — BERT Cache          full embeddings → .npy  (skips if exists)")
    print("  [3]  Stage 03 — Fold Encoding       impute → one-hot → BERT PCA → encode (all n) → fold_encoded.db")
    print("  [4]  Stage 04 — Modeling            tune → fit → evaluate (all n) → result.db")
    print("  [0]  Run ALL stages in order  [default]")
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