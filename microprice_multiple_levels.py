import numpy as np

# we are assuming bid_price[0] is the best bid, etc.
# we are defining a multiple level distance weighted microprice

def microprice(bid_prices, bid_volumes,ask_prices, ask_volumes,alpha_decay,tick_size=0.1,
               max_levels=5,clamp_inside=True,max_distance_ticks=None, min_weight=None):

    bid_prices = np.asarray(bid_prices, dtype=float)
    bid_volumes = np.asarray(bid_volumes, dtype=float)
    ask_prices = np.asarray(ask_prices, dtype=float)
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

    best_bid = bid_prices[0]  # we measure the distance of the bid levels compared to the best bid, not the mid price here!
    best_ask = ask_prices[0]

    # Distances in ticks
    if tick_size and tick_size > 0:
        bid_distances = np.maximum(np.round((best_bid - bid_prices) / tick_size), 0)
        ask_distances = np.maximum(np.round((ask_prices - best_ask) / tick_size), 0)

    # optional distance gate, if e.g. it is set to 20 we don't consider distances further away as 20 ticks from the corresponding best bid/ask
    if max_distance_ticks is not None:
        b_keep = bid_distances <= max_distance_ticks
        a_keep = ask_distances <= max_distance_ticks
        bid_prices, bid_volumes, bid_distances = bid_prices[b_keep], bid_volumes[b_keep], bid_distances[b_keep]
        ask_prices, ask_volumes, ask_distances = ask_prices[a_keep], ask_volumes[a_keep], ask_distances[a_keep]


    #weighting of each level with exponential decay
    bid_weights=np.exp(-alpha_decay * bid_distances)  #having an exponential decay, such that the influence of far out quotes are mostly ignored
    ask_weights=np.exp(-alpha_decay * ask_distances)

    # Optional weight gate, deciding if we keep the weight (drop it if it is too small)
    if min_weight is not None:
        bid_keep = bid_weights >= min_weight
        ask_keep = ask_weights >= min_weight
        bid_prices, bid_volumes, bid_weights = bid_prices[bid_keep], bid_volumes[bid_keep], bid_weights[bid_keep]
        ask_prices, ask_volumes, ask_weights = ask_prices[ask_keep], ask_volumes[ask_keep], ask_weights[ask_keep]


    #effective volume on each side
    effective_bid_volume=np.sum(bid_weights*bid_volumes)
    effective_ask_volume=np.sum(ask_weights*ask_volumes)


    #effective prices (on each side a volume-distance-weighted price)
    effective_bid_price=np.sum(bid_weights*bid_prices*bid_volumes)/effective_bid_volume
    effective_ask_price=np.sum(ask_weights*ask_prices*ask_volumes)/effective_ask_volume

    #now we make it microprice similar: so fair_value= (bid_volume*ask_price+ask_volume*bid_price)/(ask_volume+bid_volume) just generalized

    fair_value=(effective_bid_price*effective_ask_volume + effective_ask_price*effective_bid_volume)/(effective_bid_volume+effective_ask_volume)

    if clamp_inside:
        fair_value=min(max(fair_value, best_bid), best_ask)   #want the microprice to be between [best_bid,best_ask]

    return fair_value



