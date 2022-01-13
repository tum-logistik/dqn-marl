import numpy as np
import matplotlib.pyplot as plt


# zeta is profit
A = -3/2
C = 3
n_players = 2
N = n_players

eq_price = -C/(2*A)

prices = np.linspace(0.0, 2.9, num=200)
same_prices = np.tile(prices, (N, 1)).T

player_prices = same_prices.copy()
player_prices[:, 1] = player_prices[:, 1] * 0.8

def demand_calc(prices_t): # min method
    # demand_driver = np.mean(prices_t)
    # demands = np.array([A * demand_driver + C for x in prices_t])
    demands = np.array([A * x + C for x in prices_t])
    return np.clip(demands, 0, np.Inf)

def profit_func(prices_t):
    demands = demand_calc(prices_t)
    theo_profits = np.multiply(demands, prices_t)
    return np.clip(theo_profits, 0, np.Inf)

def multi_profit_func(prices_t):
    # profits = np.zeros(N)
    denom = np.sum(np.array([np.exp(x) for x in prices_t ]))
    win_probs = np.array([np.exp(x) / denom for x in prices_t ])
    profits = profit_func(prices_t)
    return np.multiply(win_probs, profits), win_probs


global_profits = [ np.sum(multi_profit_func(y)[0]) for y in same_prices ]
avg_profits = [ np.sum(multi_profit_func(y)[0])/N for y in same_prices ]
profits_0 = [ multi_profit_func(y)[0][0] for y in same_prices ]

global_profits_u = [ np.sum(multi_profit_func(y)[0]) for y in player_prices ]
avg_profits_u = [ np.sum(multi_profit_func(y)[0])/N for y in player_prices ]
profits_0_u = [ multi_profit_func(y)[0][0] for y in player_prices ]


# plt.plot(prices, global_profits)
# plt.plot(prices, global_profits_u)

# plt.plot(prices, avg_profits)
# plt.plot(prices, avg_profits_u)

# plt.plot(prices, profits_0)
# plt.plot(prices, profits_0_u)
