import pandas as pd
from RSI_Strategy import Strategy

def objective_function(rsi_crosslevel):
    RSI_Strategy = Strategy(market_data, indicator_threshold=rsi_crosslevel, stoploss_pct=0.01, target_pct=0.01, capital=100000, leverage=5, entry_type="sell", max_trades_per_day=2)
    trade_log = RSI_Strategy.run_backtest()
    pnl = trade_log['Pnl'].sum()
    print("RSI level: ", rsi_crosslevel, "PnL: ", pnl)
    return -pnl # Returning negative pnl for minimization

from pyswarm import pso

market_data = pd.read_csv("RELIANCE_sample_data.csv")
lower_bound = [32]  # Lower bound for RSI buy level
upper_bound = [37]  # Upper bound for RSI buy level

best_params, best_profit = pso(objective_function, lower_bound, upper_bound, swarmsize=10, maxiter=50)

print("Best RSI buy level:", best_params[0])
print("Best profit:", -best_profit)  # Convert back to positive profit