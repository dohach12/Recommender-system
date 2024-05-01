import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Chargement des ensembles de données
# Colonnes des données de notation des films
colonnes_data = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('u.data', sep='\t', names=colonnes_data)

# Colonnes des informations sur les films
colonnes_item = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 
                'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
item_data = pd.read_csv('u.item', sep='|', encoding='latin-1', names=colonnes_item)

# Création de la matrice utilisateur-film
user_item_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating')

# Fonction pour calculer la corrélation de Pearson entre deux films
def calculate_correlation(item1, item2):
    # Utilisateurs ayant noté les deux films
    common_users = user_item_matrix[[item1, item2]].dropna()
    if len(common_users) < 2:  # Au moins 2 utilisateurs doivent avoir noté les deux films
        return 0  # Pas de corrélation s'il n'y a pas assez de données

    # Variances des notations
    rating_variances = common_users.var()
    if rating_variances[item1] == 0 or rating_variances[item2] == 0:
        return 0  # Gérer la variance nulle

    # Calcul de la corrélation de Pearson
    correlation, _ = pearsonr(common_users[item1], common_users[item2])
    if np.isnan(correlation):
        return 0  # Gérer la division par zéro
    return correlation

# Fonction pour recommander des films en fonction des préférences de l'utilisateur
def recommend_movies(user_id, num_recommendations=5):
    # Récupérer les films non encore notés par l'utilisateur
    movies_not_rated = user_item_matrix.columns[user_item_matrix.loc[user_id].isnull()]
    correlations = []
    # Calcul de la corrélation entre les films non notés et les films déjà notés
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
    # Trier les films par corrélation et recommander les premiers
    recommendations = sorted(correlations, key=lambda x: x[1], reverse=True)[:num_recommendations]
    # Associer les identifiants de film aux titres de film
    recommended_movies_with_correlation = []
    for movie_id, correlation in recommendations:
        movie_title = item_data[item_data['movie_id'] == movie_id]['movie_title'].values[0]
        recommended_movies_with_correlation.append((movie_title, correlation))
    
    return recommended_movies_with_correlation

# Exemple : Recommander des films pour l'utilisateur 1
user_id = 1
recommended_movies = recommend_movies(user_id)
print("Films recommandés pour l'utilisateur", user_id, ":")
for movie_title, correlation in recommended_movies:
    print("Film:", movie_title, "| Corrélation:", correlation)
