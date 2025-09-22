import numpy as np


#generalization of the top of the book order book imbalance= (bid_volume-ask_volume)/(bid_volume+ask_volume)  it is between [-1,1]
def order_book_imbalance(bid_prices, bid_volumes, ask_prices, ask_volumes, alpha_decay, tick_size=0.1, max_levels=5,
                         max_distance_ticks=20,min_weight=None):
    #need best bid and best ask for the distance based weighting scheme.
    bid_prices = np.asarray(bid_prices, float)
    bid_volumes = np.asarray(bid_volumes, float)
    ask_prices = np.asarray(ask_prices, float)
    ask_volumes = np.asarray(ask_volumes, float)

    # Sort to inside and cap depth
    bid_order = np.argsort(bid_prices)[::-1]  # bids high→low
    ask_order = np.argsort(ask_prices)  # asks low→high
    bid_prices, bid_volumes = bid_prices[bid_order], bid_volumes[bid_order]
    ask_prices, ask_volumes = ask_prices[ask_order], ask_volumes[ask_order]

    if max_levels and max_levels > 0:
        bid_prices, bid_volumes = bid_prices[:max_levels], bid_volumes[:max_levels]
        ask_prices, ask_volumes = ask_prices[:max_levels], ask_volumes[:max_levels]

    best_bid = float(bid_prices[0])
    best_ask = float(ask_prices[0])

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


    book_imbalance= (effective_bid_volume - effective_ask_volume)/(effective_bid_volume + effective_ask_volume)

    #make sure that it is between [-1,1]
    book_imbalance=max(min(book_imbalance,1),-1)


    ### Now we want that if the order imbalance is small, it should be smaller than linear values, e.g. f(o.i.=0.1) < 0.1, to avoid overcorrection
    ###of our algorithm

    #The function should map -1,1 to -1,1, be f(0) =0 and should be monotone increasing
    order_imbalance=book_imbalance
    k=5
    order_imbalance=np.tanh(k* (order_imbalance**3))   #between -1 and 1 for abs(f(x))<abs(x), abs(x)<0.5 and  for larger absolute values it is larger than linear abs value.
    order_imbalance=order_imbalance*tick_size

    return  order_imbalance

