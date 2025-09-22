import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Config
# =========================
ORDERBOOK_PATH = Path(r"C:/Users/Fabia/OneDrive/Desktop/TSLA_2015-01-01_2015-03-31_10/output-2015/0/0/3/TSLA_2015-01-05_34200000_57600000_orderbook_10.csv")
OUT_DIR = ORDERBOOK_PATH.parent
N_EVENTS = None              # set an int to truncate, or None to use all rows
TICK_SIZE = 100              # LOBSTER equities: $ * 10,000 => 1 cent = 100
MAX_LEVELS_PARSE = 5         # only read first 5 levels from file
ALPHA_DECAY = 0.5            # chosen upfront
DOLLAR_SCALE = 1 / 10000.0   # convert price-like columns to dollars

# =========================
# Microprice (old version) — renamed to microprice
# =========================
def microprice_old_core(
    bid_prices, bid_volumes, ask_prices, ask_volumes,
    alpha_decay, tick_size=0.1,
    max_levels=5, clamp_inside=True,
    max_distance_ticks=None, min_weight=None
):
    bid_prices  = np.asarray(bid_prices,  dtype=float)
    bid_volumes = np.asarray(bid_volumes, dtype=float)
    ask_prices  = np.asarray(ask_prices,  dtype=float)
    ask_volumes = np.asarray(ask_volumes, dtype=float)

    # Ensure sorted best→worse
    bid_sort = np.argsort(bid_prices)[::-1]
    ask_sort = np.argsort(ask_prices)
    bid_prices, bid_volumes = bid_prices[bid_sort], bid_volumes[bid_sort]
    ask_prices, ask_volumes = ask_prices[ask_sort], ask_volumes[ask_sort]

    # Limit depth
    if max_levels is not None and max_levels > 0:
        bid_prices, bid_volumes = bid_prices[:max_levels], bid_volumes[:max_levels]
        ask_prices, ask_volumes = ask_prices[:max_levels], ask_volumes[:max_levels]

    best_bid = bid_prices[0]
    best_ask = ask_prices[0]

    # Distances in ticks
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
        bid_prices, bid_volumes, bid_distances = bid_prices[b_keep], bid_volumes[b_keep], bid_distances[b_keep]
        ask_prices, ask_volumes, ask_distances = ask_prices[a_keep], ask_volumes[a_keep], ask_distances[a_keep]

    # Exponential decay weights
    bid_weights = np.exp(-alpha_decay * bid_distances)
    ask_weights = np.exp(-alpha_decay * ask_distances)

    # Optional minimum weight filter
    if min_weight is not None:
        bid_keep = bid_weights >= min_weight
        ask_keep = ask_weights >= min_weight
        bid_prices, bid_volumes, bid_weights = bid_prices[bid_keep], bid_volumes[bid_keep], bid_weights[bid_keep]
        ask_prices, ask_volumes, ask_weights = ask_prices[ask_keep], ask_volumes[ask_keep], ask_weights[ask_keep]

    # Effective volumes
    eff_bid_vol = np.sum(bid_weights * bid_volumes)
    eff_ask_vol = np.sum(ask_weights * ask_volumes)

    if eff_bid_vol <= 0 or eff_ask_vol <= 0:
        fair_value = 0.5 * (best_bid + best_ask)
        return float(np.clip(fair_value, best_bid, best_ask) if clamp_inside else fair_value)

    # Effective prices
    eff_bid_price = np.sum(bid_weights * bid_prices * bid_volumes) / eff_bid_vol
    eff_ask_price = np.sum(ask_weights * ask_prices * ask_volumes) / eff_ask_vol

    # Microprice-like blend
    fair_value = (eff_bid_price * eff_ask_vol + eff_ask_price * eff_bid_vol) / (eff_bid_vol + eff_ask_vol)
    if clamp_inside:
        fair_value = min(max(fair_value, best_bid), best_ask)

    return float(fair_value)

def microprice(bp, bv, ap, av, alpha_decay, tick_size, max_levels):
    """Thin wrapper to match call sites."""
    return microprice_old_core(bp, bv, ap, av, alpha_decay, tick_size, max_levels)

# =========================
# Load & transform (top 5 levels only)
# =========================
df = pd.read_csv(ORDERBOOK_PATH, header=None)
if N_EVENTS is not None:
    df = df.iloc[:N_EVENTS].copy()

n_levels = df.shape[1] // 4
use_levels = min(MAX_LEVELS_PARSE, n_levels)

# Extract arrays: [AskP, AskS, BidP, BidS] per level
ask_prices  = df.iloc[:, [4*i    for i in range(n_levels)]].to_numpy(float)[:, :use_levels]
ask_sizes   = df.iloc[:, [4*i+1  for i in range(n_levels)]].to_numpy(float)[:, :use_levels]
bid_prices  = df.iloc[:, [4*i+2  for i in range(n_levels)]].to_numpy(float)[:, :use_levels]
bid_sizes   = df.iloc[:, [4*i+3  for i in range(n_levels)]].to_numpy(float)[:, :use_levels]

# Top-of-book mid (in LOBSTER units for now)
best_bid = bid_prices[:, 0]
best_ask = ask_prices[:, 0]
mid = 0.5 * (best_bid + best_ask)

# Microprice series (old version), for L=1 and L=2
micro_L1 = np.array([microprice(bp, bv, ap, av, ALPHA_DECAY, TICK_SIZE, max_levels=1)
                     for bp, bv, ap, av in zip(bid_prices, bid_sizes, ask_prices, ask_sizes)])
micro_L2 = np.array([microprice(bp, bv, ap, av, ALPHA_DECAY, TICK_SIZE, max_levels=2)
                     for bp, bv, ap, av in zip(bid_prices, bid_sizes, ask_prices, ask_sizes)])

# =========================
# Build dataframes with top-5 Px/Vol + microprice, mid LAST
# =========================
def build_df_with_levels(levels_to_include, micro_col):
    data = {}
    # Ask side (p1..pK, s1..sK)
    for i in range(levels_to_include):
        data[f"ask_p{i+1}"] = ask_prices[:, i]
        data[f"ask_s{i+1}"] = ask_sizes[:, i]
    # Bid side (p1..pK, s1..sK)
    for i in range(levels_to_include):
        data[f"bid_p{i+1}"] = bid_prices[:, i]
        data[f"bid_s{i+1}"] = bid_sizes[:, i]
    # Microprice (single column) and mid LAST
    data["microprice"] = micro_col
    data["mid"] = mid

    cols_order = (
        [f"ask_p{i+1}" for i in range(levels_to_include)] +
        [f"ask_s{i+1}" for i in range(levels_to_include)] +
        [f"bid_p{i+1}" for i in range(levels_to_include)] +
        [f"bid_s{i+1}" for i in range(levels_to_include)] +
        ["microprice", "mid"]
    )
    return pd.DataFrame(data)[cols_order]

df_lvl1 = build_df_with_levels(use_levels, micro_L1)
df_lvl2 = build_df_with_levels(use_levels, micro_L2)

# =========================
# Convert price columns to dollars (sizes remain unscaled)
# =========================
def scale_prices_in_df(df_in, levels_to_include):
    price_cols = (
        [f"ask_p{i+1}" for i in range(levels_to_include)] +
        [f"bid_p{i+1}" for i in range(levels_to_include)] +
        ["microprice", "mid"]
    )
    df_out = df_in.copy()
    for c in price_cols:
        df_out[c] = df_out[c].astype(float) * DOLLAR_SCALE
    return df_out

df_lvl1 = scale_prices_in_df(df_lvl1, use_levels)
df_lvl2 = scale_prices_in_df(df_lvl2, use_levels)

# =========================
# Save CSVs
# =========================
out1 = OUT_DIR / "tsla_orderbook_ml_dataset_L1_top5.csv"
out2 = OUT_DIR / "tsla_orderbook_ml_dataset_L2_top5.csv"
df_lvl1.to_csv(out1, index=False)
df_lvl2.to_csv(out2, index=False)

print(f"Saved:\n  - {out1}\n  - {out2}")
