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
def calculate_jaccard_similarity(item1, item2):
    # Utilisateurs ayant noté chaque film
    users_item1 = set(user_item_matrix[item1].dropna().index)
    users_item2 = set(user_item_matrix[item2].dropna().index)
    
    # Intersection des ensembles d'utilisateurs
    intersection = len(users_item1.intersection(users_item2))
    
    # Union des ensembles d'utilisateurs
    union = len(users_item1.union(users_item2))
    
    # Calcul de la similarité de Jaccard
    if union == 0:
        return 0
    else:
        similarity = intersection / union
        return similarity

# Fonction pour recommander des films en fonction des préférences de l'utilisateur
def recommend_movies(user_id, num_recommendations=5):
    movies_not_rated = user_item_matrix.columns[user_item_matrix.loc[user_id].isnull()]
    similarities = []
    
    for movie_id in movies_not_rated:
        similarity_sum = 0
        count = 0
        for rated_movie_id in user_item_matrix.loc[user_id].dropna().index:
            similarity = calculate_jaccard_similarity(movie_id, rated_movie_id)
            similarity_sum += similarity
            count += 1
        if count > 0:
            average_similarity = similarity_sum / count
            similarities.append((movie_id, average_similarity))
    
    recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    recommended_movies_with_similarity = []
    for movie_id, similarity in recommendations:
        movie_title = item_data[item_data['movie_id'] == movie_id]['movie_title'].values[0]
        recommended_movies_with_similarity.append((movie_title, similarity))
    
    return recommended_movies_with_similarity


# Exemple : Recommander des films pour l'utilisateur 1
user_id = 1
recommended_movies = recommend_movies(user_id)
print("Films recommandés pour l'utilisateur", user_id, ":")
for movie_title, correlation in recommended_movies:
    print("Film:", movie_title, "| Corrélation:", correlation)
