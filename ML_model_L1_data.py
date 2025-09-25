# ============================================================
# Predict future microprice with (Linear, MLP, LSTM) compared to baseline models: either last midprice or microprice
# Dataset: tsla_orderbook_ml_dataset_L1_top5.csv
# Features: micro_now, mid_now, spread, multi-level OBI, sizes, ladder distances
# Target: microprice[t+H]. Train on delta, add back micro_now.
# Fracdiff toggle + head alignment for fair A/B comparisons.
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import os, random
from typing import Dict, Tuple

# sklearn for scaling & metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

from tqdm.auto import tqdm

# ------------------ CONFIG ------------------
data_path = Path(r"C:\Users\Fabia\OneDrive\Desktop\TSLA_2015-01-01_2015-03-31_10\output-2015\0\0\3\tsla_orderbook_ml_dataset_L1_top5.csv")

# Prediction setup
H = 5                 # horizon: predict microprice at t+H
LOOKBACK = 20         # sequence length (events of context)
TRAIN_RATIO = 0.70    # chronological split
BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3       #learning rate
WEIGHT_DECAY = 1e-5
PATIENCE = 6
KERAS_VERBOSE = 1     # show epoch progress

# Order book imbalance (multi-level) params (tunable)
OBI_ALPHA = 0.5           # exponential decay per tick distance
OBI_MAX_LEVELS = 5        # use up to 5 levels from dataset
OBI_MAX_DISTANCE_TICKS = 20
OBI_MIN_WEIGHT = None     # e.g., 0.01 to drop far levels
TICK_SIZE = 0.01          # dollars (1 cent)

# Fracdiff toggle (OFF by default) + strict head alignment for A/B fairness
USE_FRACDIFF = True
FD_D = 0.4           # fractional differencing order
FD_TOL = 1e-4
ALIGN_HEAD_FOR_COMPARISON = True   # drop same head even if USE_FRACDIFF=False

# Repro & TF housekeeping
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

# ------------------ LOAD ------------------
df = pd.read_csv(data_path)
# Expected columns: ask_p1..5, ask_s1..5, bid_p1..5, bid_s1..5, microprice, mid
assert {"microprice", "mid"}.issubset(df.columns), "CSV missing microprice/mid columns."

# ------------------ (Optional) fractional differencing helpers ------------------
def fracdiff_weights(d: float, tol: float = 1e-4, max_size: int = 10000) -> np.ndarray:
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - (k - 1)) / k
        if abs(w_k) < tol:
            break
        w.append(w_k); k += 1
    return np.array(w, dtype=float)

def fracdiff_series(x: pd.Series, d: float, tol: float = 1e-4) -> pd.Series:
    x = pd.Series(x).astype(float)
    w = fracdiff_weights(d, tol)
    K = len(w) - 1
    if K < 1:
        return x.copy()
    vals = x.values
    y = np.full_like(vals, np.nan, dtype=float)
    for t in range(K, len(vals)):
        y[t] = np.dot(w, vals[t-K:t+1][::-1])  # causal
    return pd.Series(y, index=x.index)

def add_fracdiff_features(df_in: pd.DataFrame, cols, d=0.4, tol=1e-4, suffix=None) -> pd.DataFrame:
    out = df_in.copy()
    tag = (suffix if suffix is not None else f"fd{d}")
    for c in cols:
        s = np.log(out[c].clip(lower=1e-12))
        out[f"{c}_{tag}"] = fracdiff_series(s, d=d, tol=tol)
    return out

# Decide head drop to keep windows identical across runs
FD_HEAD_LOSS = (len(fracdiff_weights(FD_D, FD_TOL)) - 1) if ALIGN_HEAD_FOR_COMPARISON else 0
if FD_HEAD_LOSS > 0:
    df = df.iloc[FD_HEAD_LOSS:].reset_index(drop=True)

# ------------------ Multi-level order book imbalance ------------------
def order_book_imbalance(bid_prices, bid_volumes, ask_prices, ask_volumes,
                         alpha_decay, tick_size=0.01, max_levels=5,
                         max_distance_ticks=20, min_weight=None):
    """
    Generalized OBI: distance-weighted volumes with exponential decay.
    Returns a tanh-shrunk, tick-scaled value in $-units.
    """
    bid_prices  = np.asarray(bid_prices, dtype=float)
    bid_volumes = np.asarray(bid_volumes, dtype=float)
    ask_prices  = np.asarray(ask_prices, dtype=float)
    ask_volumes = np.asarray(ask_volumes, dtype=float)

    # Sort to inside and cap depth
    bid_order = np.argsort(bid_prices)[::-1]   # bids high->low
    ask_order = np.argsort(ask_prices)         # asks low->high
    bid_prices, bid_volumes = bid_prices[bid_order], bid_volumes[bid_order]
    ask_prices, ask_volumes = ask_prices[ask_order], ask_volumes[ask_order]

    if max_levels and max_levels > 0:
        bid_prices, bid_volumes = bid_prices[:max_levels], bid_volumes[:max_levels]
        ask_prices, ask_volumes = ask_prices[:max_levels], ask_volumes[:max_levels]

    best_bid = float(bid_prices[0])
    best_ask = float(ask_prices[0])

    # Distances in ticks (from best on that side)
    if tick_size and tick_size > 0:
        bid_distances = np.maximum(np.round((best_bid - bid_prices) / tick_size), 0)
        ask_distances = np.maximum(np.round((ask_prices - best_ask) / tick_size), 0)
    else:
        bid_distances = np.zeros_like(bid_prices)
        ask_distances = np.zeros_like(ask_prices)

    # Optional distance gate
    if max_distance_ticks is not None:
        b_keep = bid_distances <= max_distance_ticks
        a_keep = ask_distances <= max_distance_ticks
        bid_volumes, bid_distances = bid_volumes[b_keep], bid_distances[b_keep]
        ask_volumes, ask_distances = ask_volumes[a_keep], ask_distances[a_keep]

    # Exponential decay weights
    bid_weights = np.exp(-alpha_decay * bid_distances)
    ask_weights = np.exp(-alpha_decay * ask_distances)

    # Optional min-weight gate
    if min_weight is not None:
        b_keep = bid_weights >= min_weight
        a_keep = ask_weights >= min_weight
        bid_volumes, bid_weights = bid_volumes[b_keep], bid_weights[b_keep]
        ask_volumes, ask_weights = ask_volumes[a_keep], ask_weights[a_keep]

    # Effective volumes
    eff_bid_vol = float(np.sum(bid_weights * bid_volumes))
    eff_ask_vol = float(np.sum(ask_weights * ask_volumes))
    denom = eff_bid_vol + eff_ask_vol
    if denom <= 0:
        return 0.0

    book_imb = (eff_bid_vol - eff_ask_vol) / denom  # [-1,1]
    # shrink small values, keep monotone, map to $ scale with tick_size
    k = 5.0
    adj = np.tanh(k * (book_imb ** 3))
    return float(adj * tick_size)

# ------------------ FEATURE ENGINEERING ------------------
levels = 5
ask_p = np.stack([df[f"ask_p{i}"].values for i in range(1, levels+1)], axis=1)
ask_s = np.stack([df[f"ask_s{i}"].values for i in range(1, levels+1)], axis=1)
bid_p = np.stack([df[f"bid_p{i}"].values for i in range(1, levels+1)], axis=1)
bid_s = np.stack([df[f"bid_s{i}"].values for i in range(1, levels+1)], axis=1)

# Spread & current levels
spread_now   = df["ask_p1"] - df["bid_p1"]
micro_now_sr = df["microprice"]
mid_now_sr   = df["mid"]

# Multi-level OBI with tqdm
obi_vals = np.empty(len(df), dtype=np.float32)
for i in tqdm(range(len(df)), desc="Computing OBI", unit="rows"):
    obi_vals[i] = order_book_imbalance(
        bid_p[i], bid_s[i], ask_p[i], ask_s[i],
        alpha_decay=OBI_ALPHA, tick_size=TICK_SIZE,
        max_levels=OBI_MAX_LEVELS,
        max_distance_ticks=OBI_MAX_DISTANCE_TICKS,
        min_weight=OBI_MIN_WEIGHT
    )

# Optional fracdiff features (add alongside raw)
if USE_FRACDIFF:
    price_cols = [f"ask_p{i}" for i in range(1,6)] + [f"bid_p{i}" for i in range(1,6)] + ["mid","microprice"]
    df_fd = add_fracdiff_features(df, price_cols, d=FD_D, tol=FD_TOL)
else:
    df_fd = pd.DataFrame(index=df.index)  # empty

# Assemble features
feat: Dict[str, pd.Series] = {}
feat["micro_now"] = micro_now_sr
feat["mid_now"]   = mid_now_sr
feat["spread"]    = spread_now
feat["obi_ml"]    = pd.Series(obi_vals)

# Raw L1-5 sizes
for i in range(1, 6):
    feat[f"ask_s{i}"] = df[f"ask_s{i}"]
    feat[f"bid_s{i}"] = df[f"bid_s{i}"]

# Ladder distances (prices relative to top)
for i in range(2, 6):
    feat[f"d_ask_p{i}"] = df[f"ask_p{i}"] - df["ask_p1"]
    feat[f"d_bid_p{i}"] = df["bid_p1"]   - df[f"bid_p{i}"]

# Add fdiff features if enabled
if USE_FRACDIFF:
    for c in price_cols:
        feat[f"{c}_fd"] = df_fd[f"{c}_fd{FD_D}"]

X_all = pd.DataFrame(feat).astype(np.float32)
# Drop any initial NaNs (shouldn't exist if head aligned, but safe)
X_all = X_all.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# Align targets after potential drop
drop_n = len(df) - len(X_all)
if drop_n > 0:
    df = df.iloc[drop_n:].reset_index(drop=True)

micro_all = df["microprice"].astype(np.float32).values
mid_all   = df["mid"].astype(np.float32).values
X_all     = X_all.values

T, F = X_all.shape
print(f"Samples after prep: {T}, features: {F}")

# ------------------ BUILD SEQUENCES ------------------
def build_sequences(X: np.ndarray,
                    micro: np.ndarray,
                    mid: np.ndarray,
                    L: int, H: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      X_seq   : (N, L, F)
      y_delta : (N,) = micro[t+H] - micro[t]
      m_now   : (N,) = micro[t]
      baselines: dict of absolute baselines (persistence & mid)
    """
    T_local = len(X)
    N = T_local - (L - 1) - H
    X_seq   = np.zeros((N, L, F), dtype=np.float32)
    y_delta = np.zeros(N, dtype=np.float32)
    m_now   = np.zeros(N, dtype=np.float32)
    base_persist = np.zeros(N, dtype=np.float32)
    base_mid     = np.zeros(N, dtype=np.float32)

    for i in tqdm(range(N), desc="Building sequences", unit="samples"):
        t0, t1 = i, i + L
        tl = t1 - 1
        X_seq[i] = X[t0:t1]
        m_now[i]   = micro[tl]
        y_delta[i] = micro[tl + H] - m_now[i]
        base_persist[i] = m_now[i]
        base_mid[i]     = mid[tl]

    baselines = {"persist_micro": base_persist, "mid_as_micro": base_mid}
    return X_seq, y_delta, m_now, baselines

X_seq, y_delta, micro_now_last, baselines = build_sequences(
    X_all, micro_all, mid_all, LOOKBACK, H
)

N = X_seq.shape[0]
split = int(TRAIN_RATIO * N)
val_cut = int(0.85 * split)

tr_idx = slice(0, val_cut)
va_idx = slice(val_cut, split)
te_idx = slice(split, N)

X_tr, X_va, X_te = X_seq[tr_idx], X_seq[va_idx], X_seq[te_idx]
y_tr, y_va, y_te = y_delta[tr_idx], y_delta[va_idx], y_delta[te_idx]
m_now_tr, m_now_va, m_now_te = micro_now_last[tr_idx], micro_now_last[va_idx], micro_now_last[te_idx]
base_persist_te = baselines["persist_micro"][te_idx]
base_mid_te     = baselines["mid_as_micro"][te_idx]
micro_next_true_te = m_now_te + y_te  # absolute target

# ------------------ SCALING ------------------
scaler_X = StandardScaler()
X_tr_flat = X_tr.reshape(X_tr.shape[0]*X_tr.shape[1], X_tr.shape[2])
scaler_X.fit(X_tr_flat)

def transform_sequences(X_seq):
    N_, L_, F_ = X_seq.shape
    Xf = X_seq.reshape(N_*L_, F_)
    Xf = scaler_X.transform(Xf)
    return Xf.reshape(N_, L_, F_)

X_tr_s = transform_sequences(X_tr)
X_va_s = transform_sequences(X_va)
X_te_s = transform_sequences(X_te)

# Flattened for Linear/MLP
X_tr_flat_s = X_tr_s.reshape(X_tr_s.shape[0], -1)
X_va_flat_s = X_va_s.reshape(X_va_s.shape[0], -1)
X_te_flat_s = X_te_s.reshape(X_te_s.shape[0], -1)

# ------------------ MODELS ------------------
def build_linear(in_dim, l2=WEIGHT_DECAY):
    m = keras.Sequential([
        keras.Input(shape=(in_dim,)),
        layers.Dense(1, activation="linear",
                     kernel_regularizer=regularizers.l2(l2) if l2>0 else None)
    ])
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
              loss="mse", metrics=["mse","mae"])
    return m

def build_mlp(in_dim, hidden=(256,128), dropout=0.1, l2=WEIGHT_DECAY):
    x = keras.Input(shape=(in_dim,))
    h = x
    for hdim in hidden:
        h = layers.Dense(hdim, activation="relu",
                         kernel_regularizer=regularizers.l2(l2) if l2>0 else None)(h)
        if dropout and dropout > 0:
            h = layers.Dropout(dropout)(h)
    y = layers.Dense(1, activation="linear")(h)
    m = keras.Model(x, y)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
              loss="mse", metrics=["mse","mae"])
    return m

def build_lstm(feat_dim, units=64, layers_n=1, dropout=0.1, l2=WEIGHT_DECAY):
    inp = keras.Input(shape=(LOOKBACK, feat_dim))
    z = inp
    if layers_n == 1:
        z = layers.LSTM(units, dropout=dropout,
                        kernel_regularizer=regularizers.l2(l2) if l2>0 else None)(z)
    else:
        for _ in range(layers_n-1):
            z = layers.LSTM(units, return_sequences=True, dropout=dropout,
                            kernel_regularizer=regularizers.l2(l2) if l2>0 else None)(z)
        z = layers.LSTM(units, dropout=dropout,
                        kernel_regularizer=regularizers.l2(l2) if l2>0 else None)(z)
    out = layers.Dense(1, activation="linear")(z)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
              loss="mse", metrics=["mse","mae"])
    return m

es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                             restore_best_weights=True, verbose=1)

# ---- Linear
linear = build_linear(in_dim=X_tr_flat_s.shape[1])
linear.fit(X_tr_flat_s, y_tr, validation_data=(X_va_flat_s, y_va),
           epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=KERAS_VERBOSE)
y_pred_lin_delta = linear.predict(X_te_flat_s, verbose=0).ravel()
pred_lin_abs = m_now_te + y_pred_lin_delta

# ---- MLP
mlp = build_mlp(in_dim=X_tr_flat_s.shape[1], hidden=(256,128), dropout=0.1)
mlp.fit(X_tr_flat_s, y_tr, validation_data=(X_va_flat_s, y_va),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=KERAS_VERBOSE)
y_pred_mlp_delta = mlp.predict(X_te_flat_s, verbose=0).ravel()
pred_mlp_abs = m_now_te + y_pred_mlp_delta

# ---- LSTM
lstm = build_lstm(feat_dim=X_tr_s.shape[2], units=64, layers_n=1, dropout=0.1)
lstm.fit(X_tr_s, y_tr, validation_data=(X_va_s, y_va),
         epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=KERAS_VERBOSE)
y_pred_lstm_delta = lstm.predict(X_te_s, verbose=0).ravel()
pred_lstm_abs = m_now_te + y_pred_lstm_delta

# ------------------ METRICS (absolute microprice target) ------------------
def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred))
    }

rows = []
rows.append(("Linear (TF)",         *metrics(micro_next_true_te, pred_lin_abs).values()))
rows.append(("MLP ReLU (TF)",       *metrics(micro_next_true_te, pred_mlp_abs).values()))
rows.append(("LSTM (TF)",           *metrics(micro_next_true_te, pred_lstm_abs).values()))
rows.append(("Baseline: persist",   *metrics(micro_next_true_te, base_persist_te).values()))
rows.append(("Baseline: mid_as_micro", *metrics(micro_next_true_te, base_mid_te).values()))

print("\n=== Test metrics (target = future microprice) ===")
print(f"{'Model':28s}  {'RMSE':>10s}  {'MAE':>10s}  {'R2':>8s}")
for name, rmse, mae, r2 in rows:
    print(f"{name:28s}  {rmse:10.6f}  {mae:10.6f}  {r2:8.4f}")

# ------------------ Plot: predicted vs actual (last 300 test samples) ------------------
import matplotlib.pyplot as plt

seg = 300
n_test = len(micro_next_true_te)
seg = min(seg, n_test)
start = max(0, n_test - seg)
end = n_test
x = np.arange(end - start)

plt.figure(figsize=(12,5))
plt.plot(x, micro_next_true_te[start:end], label="Actual microprice", linewidth=2)
plt.plot(x, pred_lstm_abs[start:end],      label="LSTM prediction", alpha=0.85)
plt.plot(x, pred_mlp_abs[start:end],       label="MLP prediction",  alpha=0.85)
plt.plot(x, pred_lin_abs[start:end],       label="Linear prediction", alpha=0.85)
plt.plot(x, base_persist_te[start:end],    label="Baseline: persistence", linestyle="--", alpha=0.7)
plt.xlabel("Test sample index")
plt.ylabel("Price ($)")
plt.title(f"Predicted vs. Actual Microprice — last {end-start} test samples (H={H})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
