import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dt = pd.read_csv('nfl_elo.csv')
dt_filtered = dt[(dt['season'] >= 2010) & (dt['season'] <= 2020) & (dt['score1'] != dt['score2'])].copy()
game_cnt = len(dt_filtered)

print(f"Total number of games from 2010 to 2020: {game_cnt}")


dt_filtered['actual_outcome'] = np.where(
    dt_filtered['score1'] > dt_filtered['score2'], 1.0, 0.0
    )
dt_filtered['predicted_outcome'] = np.where(dt_filtered['elo_prob1'] >= 0.5, 1.0, 0.0)

 #Calculating accuracy and Brier score
correct_predictions = (dt_filtered['predicted_outcome'] == dt_filtered['actual_outcome']).sum()

accuracy = correct_predictions / game_cnt

print(f"Accuracy of predictions from 2010 to 2020: {accuracy:.4f}")

brier_score = np.mean((dt_filtered['elo_prob1'] - dt_filtered['actual_outcome']) ** 2)
    
print(f"Brier Score from 2010 to 2020: {brier_score:.4f}")

#Decomposing Brier score into reliability, resolution, and uncertainty

win_cnt = (dt_filtered['actual_outcome'] == 1.0).sum()
base_rate = win_cnt / game_cnt
#print(f"Base rate of winning from 2010 to 2020: {base_rate:.4f}")

uncertainty = base_rate * (1 - base_rate)
print(f"Uncertainty from 2010 to 2020: {uncertainty:.4f}")

#Calculating resolution. Assuming 10 bins with 0.1 increments from 0 to 1 for elo_prob1
resolution = 0.0
for i in range(10):
    bin_lower = i * 0.1
    bin_upper = (i + 1) * 0.1
    if(i == 9):
        bin_data = dt_filtered[(dt_filtered['elo_prob1'] >= bin_lower) & (dt_filtered['elo_prob1'] <= bin_upper)]
    else:
        bin_data = dt_filtered[(dt_filtered['elo_prob1'] >= bin_lower) & (dt_filtered['elo_prob1'] < bin_upper)]
    if len(bin_data) > 0:
        bin_base_rate = (bin_data['actual_outcome'] == 1.0).sum() / len(bin_data)
        resolution += (bin_base_rate - base_rate) ** 2 * len(bin_data) / game_cnt
        
print(f"Resolution from 2010 to 2020: {resolution:.4f}")

#Calculating reliability and generating reliability curve. Assuming 10 bins with 0.1 increments from 0 to 1 for elo_prob1
reliability = 0.0
avg_probs = []
observed_probs = []
for i in range(10):
    bin_lower = i * 0.1
    bin_upper = (i + 1) * 0.1
    if(i == 9):
        bin_data = dt_filtered[(dt_filtered['elo_prob1'] >= bin_lower) & (dt_filtered['elo_prob1'] <= bin_upper)]
    else:
        bin_data = dt_filtered[(dt_filtered['elo_prob1'] >= bin_lower) & (dt_filtered['elo_prob1'] < bin_upper)]
    if len(bin_data) > 0:
        avg_prob_for_bin = bin_data['elo_prob1'].mean()
        base_rate_for_bin = (bin_data['actual_outcome'] == 1.0).sum() / len(bin_data)
        reliability += ((avg_prob_for_bin - base_rate_for_bin) ** 2) * len(bin_data) / game_cnt
        avg_probs.append(avg_prob_for_bin)
        observed_probs.append(base_rate_for_bin)
print(f"Reliability from 2010 to 2020: {reliability:.4f}")

Brier_score_decomposed = reliability - resolution + uncertainty
print(f"Decomposed Brier Score from 2010 to 2020: {Brier_score_decomposed:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(
    avg_probs,
    observed_probs,
    marker='o',
    label='Elo predictions'
)

# Perfect calibration line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    label='Perfect calibration'
)

plt.xlabel('Mean Predicted Probability')
plt.ylabel('Observed Win Rate')
plt.title('Reliability Curve (NFL Elo Predictions, 2010–2020)')
plt.savefig('reliability_curve.pdf', format='pdf', bbox_inches='tight')

plt.legend()
plt.grid(True)
plt.show()