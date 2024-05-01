import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

# Définition de la fonction pour limiter le nombre de recommandations à 5
def get_top_recommendations(top_indices, similarities, items_df, limit=5):
    recommendations = []
    for i, indices in enumerate(top_indices):
        user_recommendations = []
        for j in range(limit):
            suggested_index = indices[j]
            similarity = similarities[i][suggested_index]
            user_recommendations.append((items_df.iloc[suggested_index]['title'], similarity))
        recommendations.append(user_recommendations)
    return recommendations

# Définition de l'utilisateur x
x = 1 # exemple pour l'utilisateur dont l'ID est de 1

# Définition des noms de colonnes pour les données de notation
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
# Chargement des données de notation en utilisant read_csv de Pandas
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'ml-100k')
ratings_df = pd.read_csv(os.path.join(data_directory, 'u.data'), sep='\t', names=ratings_cols)
# Filtrer les évaluations pour ne considérer que celles supérieures à 3 et de l'utilisateur x
user_ratings_df = ratings_df[(ratings_df['user_id'] == x) & (ratings_df['rating'] > 3)]

# Définition des noms de colonnes pour les données sur les films
items_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
              'genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7',
              'genre8', 'genre9', 'genre10', 'genre11', 'genre12', 'genre13', 'genre14',
              'genre15', 'genre16', 'genre17', 'genre18', 'genre19']
# Chargement des données sur les films
items_df = pd.read_csv(os.path.join(data_directory, 'u.item'), sep='|', names=items_cols, encoding='latin-1')

# Création d'une liste pour stocker les vecteurs des films évalués par l'utilisateur x
user_film_vectors = []
# Création des vecteurs pour les films évalués par l'utilisateur x
for index, row in user_ratings_df.iterrows():
    film_id = row['item_id']
    film_rating = row['rating']
    if film_rating > 3:  # Ajouter une condition pour ne considérer que les évaluations supérieures à 3
        film_release_date = items_df.loc[items_df['item_id'] == film_id, 'release_date'].str[-4].iloc[0]
        if pd.notna(film_release_date):
            film_release_date = int(film_release_date)
        else:
            film_release_date = 0  # Gérer les valeurs NaN en mettant une valeur par défaut
        film_genres = items_df.loc[items_df['item_id'] == film_id, 'genre1':'genre19'].values.tolist()[0]
        film_vector = [film_rating] + [film_release_date] + film_genres
        user_film_vectors.append(film_vector)

# Convertir la liste de vecteurs en un tableau numpy
user_film_vectors = np.array(user_film_vectors)

# Création d'une liste pour stocker les vecteurs des films suggérés
suggested_film_vectors = []
# Création des vecteurs pour les films suggérés
for index, row in items_df.iterrows():
    film_id = row['item_id']
    if film_id not in user_ratings_df['item_id'].unique():  # Exclure les films évalués par l'utilisateur
        film_genres = row['genre1':'genre19'].values.tolist()
        # Extraction de l'année de réalisation de la date et conversion en nombre
        try:
            film_release_date = int(row['release_date'].split('-')[-1])  # Extraction de l'année de la date
        except Exception as e:
            film_release_date = 0  # Gérer les erreurs de conversion de date en mettant une valeur par défaut
        film_vector = [0] + [film_release_date] + film_genres
        suggested_film_vectors.append(film_vector)

# Convertir la liste de vecteurs en un tableau numpy
suggested_film_vectors = np.array(suggested_film_vectors)

# Supprimer les colonnes non numériques des vecteurs
user_film_vectors_numeric = user_film_vectors[:, 1:].astype(float)
suggested_film_vectors_numeric = suggested_film_vectors[:, 1:].astype(float)

# Calcul de la similarité cosinus entre les vecteurs de films évalués par l'utilisateur et les films suggérés
similarities = cosine_similarity(user_film_vectors_numeric, suggested_film_vectors_numeric)

# Obtenir les indices des films suggérés classés par ordre de similitude décroissante
top_indices = np.argsort(similarities, axis=1)[:, ::-1]

# Utilisation de la fonction pour obtenir les recommandations
recommendations = get_top_recommendations(top_indices, similarities, items_df)

# Affichage des recommandations
for i, user_recommendations in enumerate(recommendations):
    print(f"Recommandations pour l'utilisateur {x} basées sur le film évalué {i+1}:")
    for j, recommendation in enumerate(user_recommendations):
        print(f"{j+1}. {recommendation[0]} (Similarité : {recommendation[1]})")



# Fonction pour entrainer et tester l'approche item-based en utilisant K-fold cross-validation
def item_based_k_fold_cross_validation(ratings_df, items_df, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_scores = []

    for train_index, test_index in kf.split(ratings_df):
        train_ratings = ratings_df.iloc[train_index]
        test_ratings = ratings_df.iloc[test_index]

        # Creation de la matrice user-item pour le training set
        train_matrix = train_ratings.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

        # Creation de la matrice user-item pour le  test set
        test_matrix = test_ratings.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

        # matrice  item-item similarity  en utilisant cosine similarity
        item_similarity = cosine_similarity(train_matrix.T)

        # faire des prédictions
        pred = train_matrix.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis=1)])

        # Flatten predictions  et ground truth
        pred_flatten = pred.values[test_matrix.values.nonzero()]
        test_flatten = test_matrix.values[test_matrix.values.nonzero()]

        # Calculer RMSE
        rmse = np.sqrt(np.mean((pred_flatten - test_flatten)**2))
        rmse_scores.append(rmse)

    avg_rmse = np.mean(rmse_scores)
    return avg_rmse

# Charger la ratings data
current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'ml-100k')
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings_df = pd.read_csv(os.path.join(data_directory, 'u.data'), sep='\t', names=ratings_cols)

# Charger items data
items_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
              'genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7',
              'genre8', 'genre9', 'genre10', 'genre11', 'genre12', 'genre13', 'genre14',
              'genre15', 'genre16', 'genre17', 'genre18', 'genre19']
items_df = pd.read_csv(os.path.join(data_directory, 'u.item'), sep='|', names=items_cols, encoding='latin-1')

# K-fold cross-validation
avg_rmse = item_based_k_fold_cross_validation(ratings_df, items_df)
print("Average RMSE:", avg_rmse)
