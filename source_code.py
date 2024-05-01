#Partie visualisation de la data--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 as sq3
import pandas.io.sql as pds
import pandas as pd

#u.data
colonnes = ["ID_u", "ID_f", "rating", "Timestamp"]
data = pd.read_csv("/content/drive/MyDrive/u.data", sep="\t", header=None, names=colonnes)
data.head(20)

#u.user
colonne_user=["ID_uti","genre","occupation","zip code"]
users = pd.read_csv("/content/drive/MyDrive/u.user", sep="|", header=None, names=colonne_user)
users.head(25)

data.info()

# Calculer les statistiques descriptives de la colonne "rating" du DataFrame data
rating_stats = data["rating"].describe()

# Afficher les statistiques descriptives calculées
print(rating_stats)

# Créer un histogramme pour visualiser la distribution des valeurs de "rating"
# Diviser les valeurs en 10 intervalles (bins) et marquer les bords en noir (edgecolor='black')
plt.hist(data["rating"], bins=10, edgecolor='black')

# Définir le label de l'axe x comme "Rating"
plt.xlabel('Rating')

# Définir le label de l'axe y comme "Nombre de occurrences"
plt.ylabel('Nombre de occurrences')

# Définir le titre de l'histogramme comme "Distribution des ratings dans le DataFrame data"
plt.title('Distribution des ratings dans le DataFrame data')

# Afficher l'histogramme
plt.show()

# Compter le nombre d'occurrences pour chaque occupation
occupation_counts = users.occupation.value_counts()

# Créer un graphique à barres pour visualiser les résultats
plt.figure(figsize=(10, 6))
occupation_counts.plot(kind='bar', color='skyblue')
plt.title('Nombre d\'utilisateurs par occupation')
plt.xlabel('Occupation')
plt.ylabel('Nombre d\'utilisateurs')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Compter les occurrences de chaque valeur dans la colonne "genre"
genre_counts = users['genre'].value_counts()
# Créer un diagramme à barres pour visualiser la distribution des valeurs de "genre"
plt.bar(genre_counts.index, genre_counts.values)
plt.xlabel('Genre')
plt.ylabel('Nombre d\'utilisateurs')
plt.title('Distribution des genres dans le DataFrame users')
plt.show()

ax = plt.axes()
ax.hist(data.ID_f)

ax.set(xlabel='ID_f', 
       ylabel='Fréquence',
       title='Distribution des ID des film');


# Crée une nouvelle figure pour la visualisation avec une taille de 8 pouces de largeur et 6 pouces de hauteur
plt.figure(figsize=(8, 6))
# Trace un box plot pour la colonne 'rating' du DataFrame 'data'
sns.boxplot(x=data['rating'])
# Ajoute une étiquette à l'axe des abscisses avec le label 'Rating'
plt.xlabel('Rating')
# Ajoute un titre à la visualisation avec le titre 'Box Plot des valeurs de Rating'
plt.title('Box Plot des valeurs de Rating')
# Affiche la visualisation
plt.show()

correlation = data['ID_f'].corr(data['rating'])
print(f"La corrélation entre l'id du film  et les rating  vaut  : {correlation}")

# Extraire la colonne "ID_f" du DataFrame data et la mettre dans un vecteur
id_f_vector = data['ID_f']

# Encoder la colonne "genre" du DataFrame users (F=0, M=1) et la mettre dans un vecteur
genre_encoded_vector = users['genre'].map({'F': 0, 'M': 1})

# Calculer la corrélation entre les deux vecteurs
correlation_idf_genre = id_f_vector.corr(genre_encoded_vector)
print("Corrélation entre 'ID_f' et 'genre' (encodé) :", correlation_idf_genre)

# Extraire la colonne "ID_f" du DataFrame data et la mettre dans un vecteur
id_f_vector = data['rating']
# Encoder la colonne "genre" du DataFrame users (F=0, M=1) et la mettre dans un vecteur
genre_encoded_vector = users['genre'].map({'F': 0, 'M': 1})
# Calculer la corrélation entre les deux vecteurs
correlation_idf_genre = id_f_vector.corr(genre_encoded_vector)
print("Corrélation entre 'rating' et 'genre' (encodé) :", correlation_idf_genre)

data.groupby('ID_f')['rating'].mean()

#Méthode de Vector similarity (Cosine) et optimisation k-fold:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#item-based+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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



#user-based+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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




#Méthode de Pearson Correlation Coefficient (PCC):-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#iem-based+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Chargement des ensembles de données
columns_data = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('u.data', sep='\t', names=columns_data)

columns_item = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 
                'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
item_data = pd.read_csv('u.item', sep='|', encoding='latin-1', names=columns_item)

# Création de la matrice utilisateur-article
user_item_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating')

# Fonction pour calculer la corrélation de Pearson entre deux articles
def calculate_correlation(item1, item2):
    common_users = user_item_matrix[[item1, item2]].dropna()
    if len(common_users) < 2:  # Au moins 2 utilisateurs doivent avoir évalué les deux articles
        return 0  # Pas de corrélation si les données ne sont pas suffisantes

    rating_variances = common_users.var()
    if rating_variances[item1] == 0 or rating_variances[item2] == 0:
        return 0  # Gérer la variance nulle

    correlation, _ = pearsonr(common_users[item1], common_users[item2])
    if np.isnan(correlation):
        return 0  # Gérer la division par zéro
    return correlation

# Fonction pour recommander des films en fonction des préférences de l'utilisateur
def recommend_movies(user_id, num_recommendations=5):
    # Obtention des films non encore évalués par l'utilisateur
    movies_not_rated = user_item_matrix.columns[user_item_matrix.loc[user_id].isnull()]
    correlations = []

    # Calcul de la corrélation entre les films non évalués et les films déjà évalués
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
    
    # Associer les identifiants des films à leurs titres
    recommended_movies_with_correlation = []
    for movie_id, correlation in recommendations:
        movie_title = item_data[item_data['movie_id'] == movie_id]['movie_title'].values[0]
        recommended_movies_with_correlation.append((movie_title, correlation))
    
    return recommended_movies_with_correlation

# Exemple : Recommandation de films pour l'utilisateur 1
user_id = 1
recommended_movies = recommend_movies(user_id)
print("Films recommandés pour l'utilisateur", user_id, ":")
for movie_title, correlation in recommended_movies:
    print("Film:", movie_title, "| Corrélation:", correlation)


#user-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Chargement des ensembles de données
columns_data = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('u.data', sep='\t', names=columns_data)

columns_item = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 
                'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
item_data = pd.read_csv('u.item', sep='|', encoding='latin-1', names=columns_item)

# Création de la matrice utilisateur-article
user_item_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating')

# Fonction pour calculer la corrélation de Pearson entre deux utilisateurs
def calculate_correlation(user1, user2):
    common_items = user_item_matrix.loc[user1].notna() & user_item_matrix.loc[user2].notna()
    num_common_items = sum(common_items)
    if num_common_items < 2:  # Si moins de 2 articles communs notés par les deux utilisateurs
        return 0  # Pas de corrélation si les données ne sont pas suffisantes
    correlation, _ = pearsonr(user_item_matrix.loc[user1, common_items], user_item_matrix.loc[user2, common_items])
    return correlation

# Fonction pour recommander des films en fonction des préférences de l'utilisateur
def recommend_movies(user_id, num_recommendations=100):
    user_ratings = user_item_matrix.loc[user_id].dropna()
    similar_users = []

    # Trouver des utilisateurs similaires à l'utilisateur cible
    for other_user_id in user_item_matrix.index:
        if other_user_id != user_id:
            correlation = calculate_correlation(user_id, other_user_id)
            similar_users.append((other_user_id, correlation))

    # Trier les utilisateurs similaires par corrélation
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

    # Obtenir les films notés par les utilisateurs similaires mais pas par l'utilisateur cible
    recommendations = []
    for other_user_id, correlation in similar_users:
        other_user_ratings = user_item_matrix.loc[other_user_id].dropna()
        not_rated_by_user = other_user_ratings.index.difference(user_ratings.index)
        for item_id in not_rated_by_user:
            recommendations.append((item_id, other_user_ratings[item_id], correlation))

    # Trier les recommandations par note pondérée
    recommendations = sorted(recommendations, key=lambda x: x[1] * x[2], reverse=True)[:num_recommendations]

    # Associer les identifiants des films à leurs titres et inclure le coefficient de corrélation
    recommended_movies_with_correlation = []
    for item_id, _, correlation in recommendations:
        movie_title = item_data[item_data['movie_id'] == item_id]['movie_title'].values[0]
        recommended_movies_with_correlation.append((movie_title, correlation))

    return recommended_movies_with_correlation

# Exemple : Recommandation de films pour l'utilisateur 1
user_id = 1
recommended_movies = recommend_movies(user_id)
print("Films recommandés pour l'utilisateur", user_id, ":")
for movie_title, correlation in recommended_movies:
    print("Film:", movie_title, "| Corrélation:", correlation)

#Méthode Jaccard similarity-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#item-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

    
#user-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



import numpy as np
import pandas as pd

# Chargement des données
colonne_rating = ["ID_uti", "ID_film", "rating", "time stamps"]
data = pd.read_csv(r"C:/Users/X360/OneDrive/Bureau/hhh/u.data", sep="\t", header=None, names=colonne_rating)

colonne_user = ["ID_uti", "age", "occupation", "zip code"]
users = pd.read_csv(r"C:/Users/X360/OneDrive/Bureau/hhh/u.user", sep="|", header=None, names=colonne_user)

colonne_item = ["movie_id", "movie title", "release date", "video release date", 
                "IMDb URL", "unknown", "Action", "Adventure", "Animation", 
                "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
                "Thriller", "War", "Western"]
items = pd.read_csv(r"C:/Users/X360/OneDrive/Bureau/hhh/u.item", sep="|", header=None, names=colonne_item, encoding="ISO-8859-1")


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

#Méthode de Neural Links: Approche Used-based-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Charger les données en spécifiant les noms des colonnes
df = pd.read_csv("u.data", sep='\t', names=['user', 'film', 'rating', 'timestamp'])

# Encoder les ID des utilisateurs et des films
user_encoder = LabelEncoder()
film_encoder = LabelEncoder()
user_ids = user_encoder.fit_transform(df['user'])  # Encoder les ID des utilisateurs
film_ids = film_encoder.fit_transform(df['film'])  # Encoder les ID des films
ratings = df['rating']  # Récupérer les évaluations

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(np.vstack((user_ids, film_ids)).T, ratings, test_size=0.2, random_state=42)

# Définir l'architecture du réseau neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(user_encoder.classes_) + 1, output_dim=50),  # Couche d'incorporation pour les utilisateurs et les films
    tf.keras.layers.Flatten(),  # Aplatir les données
    tf.keras.layers.Dense(64, activation='relu'),  # Couche dense avec activation ReLU
    tf.keras.layers.Dense(1)  # Couche de sortie
])

# Compiler le modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraîner le modèle
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Évaluer le modèle
loss = model.evaluate(X_test, y_test)
print("Erreur quadratique moyenne sur l'ensemble de test:", loss)

# Faire des prédictions
predictions = model.predict(X_test)

# Générer des recommandations pour un utilisateur
def generate_recommendations(user_id, num_recommendations=10):
    user_input = np.full((len(film_encoder.classes_), 1), user_id)  # Entrée utilisateur
    film_input = np.arange(len(film_encoder.classes_)).reshape(-1, 1)  # Entrée des films
    interaction_matrix = np.hstack((user_input, film_input))  # Matrice d'interaction utilisateur-film
    predicted_ratings = model.predict(interaction_matrix).flatten()  # Prédire les évaluations
    top_indices = np.argsort(predicted_ratings)[-num_recommendations:][::-1]  # Indices des meilleures évaluations
    recommended_films = film_encoder.inverse_transform(top_indices)  # Films recommandés
    return recommended_films

# Exemple d'utilisation :
user_id = user_encoder.transform(['400'])[0]  # ID utilisateur
recommendations = generate_recommendations(user_id)  # Générer des recommandations
print("Films recommandés pour l'utilisateur", user_id)
for film_id in recommendations:
    print("ID du film:", film_id)




#Partie fine tuning et K-cross validation appliquée sur la similarité par Pearson

#item-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

# Chargement des ensembles de données
columns_data = ['user_id', 'item_id', 'rating', 'timestamp']
columns_item = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 
                'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Fonction pour calculer la corrélation de Pearson entre deux articles
def calculate_correlation(user_item_matrix, item1, item2):
    common_users = user_item_matrix[[item1, item2]].dropna()
    if len(common_users) < 2:  # Au moins 2 utilisateurs doivent avoir évalué les deux articles
        return 0  # Aucune corrélation si les données ne sont pas suffisantes

    rating_variances = common_users.var()
    if rating_variances[item1] == 0 or rating_variances[item2] == 0:
        return 0  # Gérer la variance nulle

    correlation, _ = pearsonr(common_users[item1], common_users[item2])
    if np.isnan(correlation):
        return 0  # Gérer la division par zéro
    return correlation

# Fonction pour recommander des films en fonction des préférences de l'utilisateur
def recommend_movies(user_item_matrix, user_id, num_recommendations=5):
    # Obtenir les films non encore évalués par l'utilisateur
    movies_not_rated = user_item_matrix.columns[user_item_matrix.loc[user_id].isnull()]
    correlations = []

    # Calculer la corrélation entre les films non évalués et les films déjà évalués
    for movie_id in movies_not_rated:
        correlation_sum = 0
        count = 0
        for rated_movie_id in user_item_matrix.loc[user_id].dropna().index:
            correlation = calculate_correlation(user_item_matrix, movie_id, rated_movie_id)
            correlation_sum += correlation
            count += 1
        if count > 0:
            average_correlation = correlation_sum / count
            correlations.append((movie_id, average_correlation))

    # Trier les films par corrélation et recommander les premiers
    recommendations = sorted(correlations, key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    return recommendations

# Réaliser une validation croisée en K-fold
num_splits = 5
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

average_correlation = 0
for fold, (train_index, test_index) in enumerate(kf.split(range(num_splits))):
    print(f"\nFold {fold + 1}:")

    # Charger les ensembles de données pour l'entraînement et le test
    train_data = pd.concat([pd.read_csv(f'u{i+1}.base', sep='\t', names=columns_data) for i in train_index])
    test_data = pd.read_csv(f'u{test_index[0] + 1}.base', sep='\t', names=columns_data)

    # Créer la matrice utilisateur-article pour les données d'entraînement
    train_user_item_matrix = train_data.pivot_table(index='user_id', columns='item_id', values='rating')

    # Évaluer sur les données de test
    fold_average_correlation = 0
    for user_id in test_data['user_id'].unique():
        recommendations = recommend_movies(train_user_item_matrix, user_id)
        fold_average_correlation += sum(correlation for _, correlation in recommendations)
    fold_average_correlation /= len(test_data['user_id'].unique())

    print("Fold Average Correlation:", fold_average_correlation)
    average_correlation += fold_average_correlation

# Calculer la corrélation moyenne globale
average_correlation /= num_splits
print("\nOverall Average Correlation:", average_correlation)


#user-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

# Importation des bibliothèques nécessaires

# Définition des colonnes pour l'ensemble de données
columns_data = ['user_id', 'item_id', 'rating', 'timestamp']

# Chargement des ensembles de données pour chaque pli
data_files = ['u1.base', 'u2.base', 'u3.base', 'u4.base', 'u5.base']
data_sets = [pd.read_csv(file, sep='\t', names=columns_data) for file in data_files]

# Définition des colonnes pour les données d'articles
columns_item = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 
                'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
item_data = pd.read_csv('u.item', sep='|', encoding='latin-1', names=columns_item)

# Définition de la fonction pour créer la matrice utilisateur-article
def create_user_item_matrix(data):
    return data.pivot_table(index='user_id', columns='item_id', values='rating')

# Fonction pour calculer la corrélation de Pearson entre deux utilisateurs
def calculate_correlation(user1, user2, user_item_matrix):
    common_items = user_item_matrix.loc[user1].notna() & user_item_matrix.loc[user2].notna()
    num_common_items = sum(common_items)
    if num_common_items < 2:  # S'il y a moins de 2 articles communs évalués par les deux utilisateurs
        return 0  # Pas de corrélation si les données sont insuffisantes
    correlation, _ = pearsonr(user_item_matrix.loc[user1, common_items], user_item_matrix.loc[user2, common_items])
    return correlation

# Fonction pour recommander des films en fonction des préférences de l'utilisateur
def recommend_movies(user_id, user_item_matrix, num_recommendations=100):
    user_ratings = user_item_matrix.loc[user_id].dropna()
    similar_users = []

    # Recherche des utilisateurs similaires à l'utilisateur cible
    for other_user_id in user_item_matrix.index:
        if other_user_id != user_id:
            correlation = calculate_correlation(user_id, other_user_id, user_item_matrix)
            similar_users.append((other_user_id, correlation))

    # Tri des utilisateurs similaires par corrélation
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

    # Obtention des films évalués par des utilisateurs similaires mais pas par l'utilisateur cible
    recommendations = []
    for other_user_id, correlation in similar_users:
        other_user_ratings = user_item_matrix.loc[other_user_id].dropna()
        not_rated_by_user = other_user_ratings.index.difference(user_ratings.index)
        for item_id in not_rated_by_user:
            recommendations.append((item_id, other_user_ratings[item_id], correlation))

    # Tri des recommandations par note pondérée
    recommendations = sorted(recommendations, key=lambda x: x[1] * x[2], reverse=True)[:num_recommendations]

    # Association des identifiants des films aux titres et inclusion du coefficient de corrélation
    recommended_movies_with_correlation = []
    for item_id, _, correlation in recommendations:
        movie_title = item_data[item_data['movie_id'] == item_id]['movie_title'].values[0]
        recommended_movies_with_correlation.append((movie_title, correlation))

    return recommended_movies_with_correlation

# Réalisation de la validation croisée K-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_index, test_index) in enumerate(kf.split(data_sets)):
    print(f"\nPlis {fold + 1}:")

    # Fusion des données d'entraînement de quatre plis
    train_data = pd.concat([data_sets[i] for i in train_index])

    # Création de la matrice utilisateur-article pour les données d'entraînement
    train_user_item_matrix = create_user_item_matrix(train_data)

    # Évaluation sur le pli restant
    test_data = data_sets[test_index[0]]
    test_user_id = test_data['user_id'].unique()[0]

    # Recommander des films pour l'utilisateur de test
    recommended_movies = recommend_movies(test_user_id, train_user_item_matrix)

    # Évaluation des recommandations
    num_recommended = len(recommended_movies)
    sum_correlation = sum(correlation for _, correlation in recommended_movies)
    avg_correlation = sum_correlation / num_recommended if num_recommended > 0 else 0
    fold_results.append(avg_correlation)

# Calcul de la corrélation moyenne sur tous les plis
average_correlation = np.mean(fold_results)
print("\nCorrélation moyenne sur tous les plis:", average_correlation)

