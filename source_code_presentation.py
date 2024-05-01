#Importation des bibliothèques
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

#Méthode 1: Vector similarity (cosine)------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#item-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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




#user-based+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
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

#Methode 2: Pearson Correlation------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#item-based+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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



#user-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    

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


#Methode 3: Jaccard------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#item-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
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



#user-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    

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



#Méthode 4 : Nerual Links(approche user-based)---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
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


#Optimisation K-fold pour la correlation de Pearson-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

#item-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
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



#user-based++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

