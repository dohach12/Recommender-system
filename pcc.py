import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Load the dataset
columns = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('u.data', sep='\t', names=columns)

# Create user-item matrix
user_item_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating')

# Function to calculate Pearson correlation between two items
def calculate_correlation(item1, item2):
    common_users = user_item_matrix[[item1, item2]].dropna()
    if len(common_users) < 2:  # At least 2 users should have rated both items
        return 0  # No correlation if not enough data
    correlation, _ = pearsonr(common_users[item1], common_users[item2])
    return correlation

# Function to recommend movies based on user preferences
def recommend_movies(user_id, num_recommendations=5):
    # Get movies not already rated by user
    movies_not_rated = user_item_matrix.columns[user_item_matrix.loc[user_id].isnull()]
    correlations = []

    # Calculate correlation between movies not rated and movies already rated
    for movie_id in movies_not_rated:
        correlation_sum = 0
        count = 0
        for rated_movie_id in user_item_matrix.loc[user_id].dropna().index:
            correlation = calculate_correlation(movie_id, rated_movie_id)
            correlation_sum += correlation
            count += 1
        if count > 0:
            average_correlation = correlation_sum / count
            correlations.append((movie_id, average_correlation))

    # Sort movies by correlation and recommend the top ones
    recommendations = sorted(correlations, key=lambda x: x[1], reverse=True)[:num_recommendations]
    return recommendations

# Example: Recommend movies for user 1
user_id = 1
recommended_movies = recommend_movies(user_id)
print("Recommended movies for user", user_id, ":")
for movie_id, correlation in recommended_movies:
    print("Movie ID:", movie_id, "| Correlation:", correlation)
