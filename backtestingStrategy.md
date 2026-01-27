Pipeline:

Model predicts next price for each timestep
Generate trading signals from predictions (e.g., "buy if predicted price > current price")
Simulate trades over historical data
Calculate P&L and risk metrics from the simulated returns
Example:

Then compute:

Sharpe Ratio = Mean return / Std dev of returns
Sortino Ratio = Mean return / Std dev of downside returns
Max Drawdown = Largest peak-to-trough decline
VaR = Worst 5th percentile loss