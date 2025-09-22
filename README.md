# microprice-forecasting
## multi-level microprice
The top of the book microprice is a concept discussed in a paper by Sasha Stoikov, and the book Algorithmic and High Frequency Trading by Cartea et al., which resembles a better estimate of the fair value of an asset than its midprice, which is a low frequency trading signal. My main interest in the microprice is in the context of market making, as the MM is trying to find a fair value, around which he wants to place his bid and ask prices.
Here, I extended this concept such that it uses multiple levels of LOB data, making it a more robust estimate of the fair value.

Consider the following example:
ask @ 101$/1000 shares,   bid @ 100$/10 shares. When using the traditional microprice, you would skew your prices towards the bid price as there is higher pressure to sell  (microprice would be =100.01$).
But what if there is a large volume available at the 2nd best bid @ 99$/ 1M shares? 
That scenario isn't considered by the top of the book microprice, but in my extension with an exponential weighting scheme.









As features, I use spread, multi-level order book imbalance, and the top 5 bid/ask prices and volumes.
The label is the microprice of the next time step, given the last 20 time steps of the LOB data.

![Trained Microprice Forecasts](micropriceforecast.png)
