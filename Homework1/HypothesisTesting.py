import numpy as np

n_A = 3250
n_B = 2870
x_A = 1320
x_B = 1250

p_A = x_A / n_A
p_B = x_B / n_B
observed_rate_diff = np.abs(p_A - p_B)

simulated_diff = (x_A + x_B) / (n_A + n_B)
n_trial = 500000

# Draw sample counts from Binomial(n, p_pool)
simulated_conversion_rate_A = np.random.binomial(n=n_A, p=simulated_diff, size=n_trial)
simulated_conversion_rate_B = np.random.binomial(n=n_B, p=simulated_diff, size=n_trial)

simulated_p_A = simulated_conversion_rate_A / n_A
simulated_p_B = simulated_conversion_rate_B / n_B
simulated_rate_diff = simulated_p_A - simulated_p_B

p_value = np.mean(np.abs(simulated_rate_diff) >= observed_rate_diff)

print(f"Sample Conversion Rate A: {p_A:.4f}")
print(f"Sample Conversion Rate B: {p_B:.4f}")
print(f"Observed Difference (B - A): {observed_rate_diff:.4f}")
print(f"simulated_conversion_rate_A: {simulated_conversion_rate_A[:50000]}")
print(f"simulated_conversion_rate_B: {simulated_conversion_rate_B[:5]}")
print(f"simulated_rate_diff: {simulated_rate_diff[:5]}")
print(f"Simulated p-value: {p_value:.4f}")