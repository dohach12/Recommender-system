import numpy as np
import pandas as pd

# Chargement des données
colonne_rating = ["ID_uti", "ID_film", "rating", "time stamps"]
data = pd.read_csv(r'u.data', sep="\t", header=None, names=colonne_rating)

colonne_user = ["ID_uti", "age", "occupation", "zip code"]
users = pd.read_csv(r'u.user', sep="|", header=None, names=colonne_user)

colonne_item = ["movie_id", "movie title", "release date", "video release date", 
                "IMDb URL", "unknown", "Action", "Adventure", "Animation", 
                "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
                "Thriller", "War", "Western"]
items = pd.read_csv(r'u.item', sep="|", header=None, names=colonne_item, encoding="ISO-8859-1")


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def user_based_recommendation(ratings_df, users_df, items_df):
    user_id = int(input("Veuillez entrer l'ID de l'utilisateur : "))
    
    # Obtenir les évaluations de l'utilisateur
    user_ratings = ratings_df[ratings_df['ID_uti'] == user_id]
    
    # Trouver les utilisateurs similaires en fonction des évaluations
    similarities = {}
    for index, row in ratings_df.iterrows():
        if row['ID_uti'] != user_id:
            other_user_ratings = ratings_df[ratings_df['ID_uti'] == row['ID_uti']]
            intersection = len(set(user_ratings['ID_film']).intersection(set(other_user_ratings['ID_film'])))
            union = len(set(user_ratings['ID_film']).union(set(other_user_ratings['ID_film'])))
            similarity = intersection / union
            similarities[row['ID_uti']] = similarity
    
    # Trier les utilisateurs similaires par similarité décroissante
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Obtenir les films notés par les utilisateurs similaires
    recommended_movies = set()
    for user, similarity in sorted_similarities:
        similar_user_ratings = ratings_df[ratings_df['ID_uti'] == user]
        recommended_movies.update(similar_user_ratings['ID_film'])
        if len(recommended_movies) >= 10:
            break
    
    # Filtrer les films déjà notés par l'utilisateur
    recommended_movies -= set(user_ratings['ID_film'])
    
    # Afficher les films recommandés
    recommended_movies_data = items_df[items_df['movie_id'].isin(recommended_movies)]
    print("Suggestions de films pour l'utilisateur ID", user_id)
    print(recommended_movies_data[['movie title', 'release date']].head(10))
    
    # Afficher les coefficients de Jaccard calculés
    print("\nCoefficients de Jaccard calculés :")
    for user, similarity in sorted_similarities[:10]:
        print("ID utilisateur:", user, "- Similarité:", similarity)


# Utiliser la fonction pour recommander des films à un utilisateur donné
user_based_recommendation(data, users, items)