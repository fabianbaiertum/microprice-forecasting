import numpy as np

def microprice(
    bid_prices, bid_volumes,
    ask_prices, ask_volumes,
    alpha_decay, tick_size=0.1,
    max_levels=5, clamp_inside=True,
    max_distance_ticks=None, min_weight=None
):
    """
    Microprice ≈ mid + (spread/2) * (4/pi) * arctan( (qb-qa)/(qb+qa) )

    qb, qa are *effective* (distance-weighted) volumes built with an
    exponential decay over level distance in ticks.
    """

    bid_prices  = np.asarray(bid_prices,  dtype=float)
    bid_volumes = np.asarray(bid_volumes, dtype=float)
    ask_prices  = np.asarray(ask_prices,  dtype=float)
    ask_volumes = np.asarray(ask_volumes, dtype=float)

    # Ensure sorted best→worse (bids high→low, asks low→high)
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

    # Distances in ticks from the *best on that side*
    if tick_size and tick_size > 0:
        bid_distances = np.maximum(np.round((best_bid - bid_prices) / tick_size), 0.0)
        ask_distances = np.maximum(np.round((ask_prices - best_ask) / tick_size), 0.0)
    else:
        bid_distances = np.zeros_like(bid_prices)
        ask_distances = np.zeros_like(ask_prices)

    # Optional distance gate
    if max_distance_ticks is not None:
        b_keep = bid_distances <= max_distance_ticks
        a_keep = ask_distances <= max_distance_ticks
        bid_prices, bid_volumes, bid_distances = bid_prices[b_keep], bid_volumes[b_keep], bid_distances[b_keep]
        ask_prices, ask_volumes, ask_distances = ask_prices[a_keep], ask_volumes[a_keep], ask_distances[a_keep]

    # Exponential distance decay weights
    bid_weights = np.exp(-alpha_decay * bid_distances)
    ask_weights = np.exp(-alpha_decay * ask_distances)

    # Optional minimum weight filter
    if min_weight is not None:
        b_keep = bid_weights >= min_weight
        a_keep = ask_weights >= min_weight
        bid_volumes, bid_weights = bid_volumes[b_keep], bid_weights[b_keep]
        ask_volumes, ask_weights = ask_volumes[a_keep], ask_weights[a_keep]

    # Effective volumes (qb, qa)
    qb = float(np.sum(bid_weights * bid_volumes))
    qa = float(np.sum(ask_weights * ask_volumes))

    # Mid & spread from top of book
    m = 0.5 * (best_bid + best_ask)
    s = best_ask - best_bid

    # Handle degenerate case (no effective liquidity after filters)
    if qb + qa <= 0:
        micro = m
    else:
        I = (qb - qa) / (qb + qa)    #order book imbalance
        micro = m + (s / 2.0) * (4.0 / np.pi) * np.arctan(I)   #approximation as in the Baruch MFE lecture notes.

    if clamp_inside:
        micro = min(max(micro, best_bid), best_ask)

    return float(micro)



