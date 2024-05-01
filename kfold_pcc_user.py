import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

# Load the datasets for each fold
data_files = ['u1.data', 'u2.data', 'u3.data', 'u4.data', 'u5.data']
data_sets = [pd.read_csv(file, sep='\t', names=columns_data) for file in data_files]

# Define columns and item data
columns_data = ['user_id', 'item_id', 'rating', 'timestamp']
columns_item = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 
                'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
item_data = pd.read_csv('u.item', sep='|', encoding='latin-1', names=columns_item)

# Define user-item matrix creation function
def create_user_item_matrix(data):
    return data.pivot_table(index='user_id', columns='item_id', values='rating')

# Function to calculate Pearson correlation between two users
def calculate_correlation(user1, user2, user_item_matrix):
    common_items = user_item_matrix.loc[user1].notna() & user_item_matrix.loc[user2].notna()
    num_common_items = sum(common_items)
    if num_common_items < 2:  # If less than 2 common items rated by both users
        return 0  # No correlation if not enough data
    correlation, _ = pearsonr(user_item_matrix.loc[user1, common_items], user_item_matrix.loc[user2, common_items])
    return correlation

# Function to recommend movies based on user preferences
def recommend_movies(user_id, user_item_matrix, num_recommendations=100):
    user_ratings = user_item_matrix.loc[user_id].dropna()
    similar_users = []

    # Find users similar to the target user
    for other_user_id in user_item_matrix.index:
        if other_user_id != user_id:
            correlation = calculate_correlation(user_id, other_user_id, user_item_matrix)
            similar_users.append((other_user_id, correlation))

    # Sort similar users by correlation
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

    # Get movies rated by similar users but not by the target user
    recommendations = []
    for other_user_id, correlation in similar_users:
        other_user_ratings = user_item_matrix.loc[other_user_id].dropna()
        not_rated_by_user = other_user_ratings.index.difference(user_ratings.index)
        for item_id in not_rated_by_user:
            recommendations.append((item_id, other_user_ratings[item_id], correlation))

    # Sort recommendations by weighted rating
    recommendations = sorted(recommendations, key=lambda x: x[1] * x[2], reverse=True)[:num_recommendations]

    # Map movie IDs to movie titles and include correlation coefficient
    recommended_movies_with_correlation = []
    for item_id, _, correlation in recommendations:
        movie_title = item_data[item_data['movie_id'] == item_id]['movie_title'].values[0]
        recommended_movies_with_correlation.append((movie_title, correlation))

    return recommended_movies_with_correlation

# Perform K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(kf.split(data_sets)):
    print(f"\nFold {fold + 1}:")

    # Merge training data from four folds
    train_data = pd.concat([data_sets[i] for i in train_index])

    # Create user-item matrix for training data
    train_user_item_matrix = create_user_item_matrix(train_data)

    # Evaluate on the remaining fold
    test_data = data_sets[test_index[0]]
    test_user_id = test_data['user_id'].unique()[0]

    # Recommend movies for the test user
    recommended_movies = recommend_movies(test_user_id, train_user_item_matrix)
    print("Recommended movies for user", test_user_id, ":")
    for movie_title, correlation in recommended_movies:
        print("Movie:", movie_title, "| Correlation:", correlation)
