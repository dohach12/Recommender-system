import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Chargement des ensembles de données
colonnes_data = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('u.data', sep='\t', names=colonnes_data)

colonnes_item = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 
                'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
item_data = pd.read_csv('u.item', sep='|', encoding='latin-1', names=colonnes_item)

# Création de la matrice utilisateur-film
user_item_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating')

# Fonction pour calculer la corrélation de Pearson entre deux utilisateurs
def calculate_correlation(user1, user2):
    # Articles notés par les deux utilisateurs
    common_items = user_item_matrix.loc[user1].notna() & user_item_matrix.loc[user2].notna()
    num_common_items = sum(common_items)
    if num_common_items < 2:  # Si moins de 2 articles communs notés par les deux utilisateurs
        return 0  # Pas de corrélation s'il n'y a pas assez de données
    correlation, _ = pearsonr(user_item_matrix.loc[user1, common_items], user_item_matrix.loc[user2, common_items])
    return correlation

# Fonction pour recommander des films basée sur les préférences de l'utilisateur
def recommend_movies(user_id, num_recommendations=100):
    user_ratings = user_item_matrix.loc[user_id].dropna()
    similar_users = []

    # Trouver les utilisateurs similaires à l'utilisateur cible
    for other_user_id in user_item_matrix.index:
        if other_user_id != user_id:
            correlation = calculate_correlation(user_id, other_user_id)
            similar_users.append((other_user_id, correlation))

    # Trier les utilisateurs similaires par corrélation
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

    # Obtenir les films notés par les utilisateurs similaires mais non par l'utilisateur cible
    recommendations = []
    for other_user_id, correlation in similar_users:
        other_user_ratings = user_item_matrix.loc[other_user_id].dropna()
        not_rated_by_user = other_user_ratings.index.difference(user_ratings.index)
        for item_id in not_rated_by_user:
            recommendations.append((item_id, other_user_ratings[item_id], correlation))

    # Trier les recommandations par note pondérée
    recommendations = sorted(recommendations, key=lambda x: x[1] * x[2], reverse=True)[:num_recommendations]

    # Associer les identifiants de film aux titres de film et inclure le coefficient de corrélation
    recommended_movies_with_correlation = []
    for item_id, _, correlation in recommendations:
        movie_title = item_data[item_data['movie_id'] == item_id]['movie_title'].values[0]
        recommended_movies_with_correlation.append((movie_title, correlation))

    return recommended_movies_with_correlation

# Exemple : Recommander des films pour l'utilisateur 1
user_id = 1
recommended_movies = recommend_movies(user_id)
print("Films recommandés pour l'utilisateur", user_id, ":")
for movie_title, correlation in recommended_movies:
    print("Film:", movie_title, "| Corrélation:", correlation)
