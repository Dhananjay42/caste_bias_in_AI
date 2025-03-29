import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load JSON data
with open("data/results_partial.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract ratings based on caste
upper_caste_ratings = []
lower_caste_ratings = []

for entry in data:
    rating = int(entry["Rating"])
    caste = entry["Caste"].lower()

    if caste == "upper":
        upper_caste_ratings.append(rating)  
    elif caste == "lower":
        lower_caste_ratings.append(rating)

# Convert to numpy arrays
upper_caste_ratings = np.array(upper_caste_ratings)
lower_caste_ratings = np.array(lower_caste_ratings)

# Compute rating distributions
upper_caste_unique, upper_caste_counts = np.unique(upper_caste_ratings, return_counts=True)
lower_caste_unique, lower_caste_counts = np.unique(lower_caste_ratings, return_counts=True)

# Normalize to relative frequencies
upper_caste_rel_freq = upper_caste_counts / np.sum(upper_caste_counts)
lower_caste_rel_freq = lower_caste_counts / np.sum(lower_caste_counts)

# Calculate Means
upper_caste_mean = np.mean(upper_caste_ratings)
lower_caste_mean = np.mean(lower_caste_ratings)

# Print results
print("Upper Caste Rating Distribution:", dict(zip(upper_caste_unique, upper_caste_rel_freq)))
print("Lower Caste Rating Distribution:", dict(zip(lower_caste_unique, lower_caste_rel_freq)))
print(f"Mean Rating for Upper Caste: {upper_caste_mean:.2f}")
print(f"Mean Rating for Lower Caste: {lower_caste_mean:.2f}")


# Visualize the rating distributions as a double bar graph
plt.figure(figsize=(12, 6))

# Position of bars on x-axis
bar_width = 0.35
index = np.arange(len(upper_caste_unique))

# Plot Upper Caste and Lower Caste ratings as double bars
plt.bar(index, upper_caste_rel_freq, bar_width, color='blue', alpha=0.7, label='Upper Caste')
plt.bar(index + bar_width, lower_caste_rel_freq, bar_width, color='red', alpha=0.7, label='Lower Caste')

# Adding labels and title
plt.title('Comparison of Upper and Lower Caste Rating Distributions (Withholding Academic details)')
plt.xlabel('Rating')
plt.ylabel('Relative Frequency')
plt.xticks(index + bar_width / 2, upper_caste_unique)  # Centering x-ticks
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Rating Differences >> 

ratings_by_pair_id = {}

for entry in data:
    rating = int(entry["Rating"])
    caste = entry["Caste"].lower()
    pair_ID = entry["Pair_ID"]

    if pair_ID not in ratings_by_pair_id:
        ratings_by_pair_id[pair_ID] = {"upper": None, "lower": None}

    if caste == "upper":
        ratings_by_pair_id[pair_ID]["upper"] = rating
    elif caste == "lower":
        ratings_by_pair_id[pair_ID]["lower"] = rating

# List to store rating differences
rating_differences = []

# Calculate rating differences for each pair_ID
for pair_ID, ratings in ratings_by_pair_id.items():
    uc_rating = ratings["upper"]
    lc_rating = ratings["lower"]

    if uc_rating is not None and lc_rating is not None:
        # Calculate the rating difference (UC - LC)
        rating_differences.append(uc_rating - lc_rating)

# Visualize the distribution of the rating differences
plt.figure(figsize=(8, 6))
plt.hist(rating_differences, bins=20, color='green', alpha=0.7)
plt.title('Distribution of Rating Differences (Upper Caste - Lower Caste) by Pair_ID (Withholding Academic details)')
plt.xlabel('Rating Difference (Upper Caste - Lower Caste)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

