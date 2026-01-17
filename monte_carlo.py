import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. Set up historical estimates (S&P 500)
# -----------------------------------------

# Typical long-term estimates for the S&P 500
annual_mean_return = 0.07      # 7% per year
annual_volatility = 0.15       # 15% annual volatility

# Convert annual values to daily values (252 trading days per year)
days = 252
dt = 1 / days
mu = annual_mean_return
sigma = annual_volatility

# -----------------------------------------
# 2. Simulation settings
# -----------------------------------------

num_paths = 1_000              # number of Monte Carlo simulations
initial_price = 100            # starting portfolio value

# Matrix to store simulated price paths
paths = np.zeros((days, num_paths))
paths[0] = initial_price

# -----------------------------------------
# 3. Simulate price paths using GBM
# -----------------------------------------

for t in range(1, days):
    z = np.random.normal(0, 1, num_paths)

    paths[t] = paths[t - 1] * np.exp(
        (mu - 0.5 * sigma ** 2) * dt
        + sigma * np.sqrt(dt) * z
    )

# -----------------------------------------
# 4. Calculate returns after 1 year
# -----------------------------------------

final_prices = paths[-1]
returns = (final_prices - initial_price) / initial_price

# -----------------------------------------
# 5. Plot distribution of returns
# -----------------------------------------

plt.hist(returns, bins=50)
plt.title("Monte Carlo Simulation of Portfolio Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# -----------------------------------------
# 6. Print summary statistics
# -----------------------------------------

print("Expected return (mean):", np.mean(returns))
print("Volatility of returns:", np.std(returns))
print("5% Value at Risk (VaR):", np.percentile(returns, 5))
