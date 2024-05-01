import pandas as pd
import numpy as np



#Lecture des données
u1_base = pd.read_csv("u1.base", sep="\t", header=None, names=["Utilisateur", "Film", "Note", "Timestamp"])
u1_test = pd.read_csv("u1.test", sep="\t", header=None, names=["Utilisateur", "Film", "Note", "Timestamp"])
u2_base = pd.read_csv("u2.base", sep="\t", header=None, names=["Utilisateur", "Film", "Note", "Timestamp"])
u2_test = pd.read_csv("u2.test", sep="\t", header=None, names=["Utilisateur", "Film", "Note", "Timestamp"])
u3_base = pd.read_csv("u3.base", sep="\t", header=None, names=["Utilisateur", "Film", "Note", "Timestamp"])
u3_test = pd.read_csv("u3.test", sep="\t", header=None, names=["Utilisateur", "Film", "Note", "Timestamp"])
u4_base = pd.read_csv("u4.base", sep="\t", header=None, names=["Utilisateur", "Film", "Note", "Timestamp"])
u4_test = pd.read_csv("u4.test", sep="\t", header=None, names=["Utilisateur", "Film", "Note", "Timestamp"])
u5_base = pd.read_csv("u5.base", sep="\t", header=None, names=["Utilisateur", "Film", "Note", "Timestamp"])
u5_test = pd.read_csv("u5.test", sep="\t", header=None, names=["Utilisateur", "Film", "Note", "Timestamp"])


users = pd.read_csv("u.user", sep="|", header=None, names=["user_ID","age","gender","occupation","zip code"])
movie = pd.read_csv("u.item", sep="|", header=None, names=["movie id", " movie title" , "release date "," video release date ","IMDb URL "," unknown "," Action "," Adventure "," Animation ","enfance "," Comedy "," Crime "," Documentary "," Drama "," Fantasy ","Film-Noir "," Horror "," Musical "," Mystery "," Romance "," Sci-Fi ","Thriller "," War "," Western "], encoding="latin1")
occupation= pd.read_csv("u.occupation", header=None,names=["occupation"])

occupation.replace({"occupation":{"administrator":0,"artist":1,"doctor":2,"educator":3,"engineer":4,"entertainment":5,"executive":6,"healthcare":7,"homemaker":8,"lawyer":9,"librarian":10,"marketing":11,"none":12,"other":13,"programmer":14,"retired":15,"salesman":16,"scientist":17,"student":18,"technician":19,"writer":20}},inplace=True)
users.replace({"occupation":{"administrator":0,"artist":1,"doctor":2,"educator":3,"engineer":4,"entertainment":5,"executive":6,"healthcare":7,"homemaker":8,"lawyer":9,"librarian":10,"marketing":11,"none":12,"other":13,"programmer":14,"retired":15,"salesman":16,"scientist":17,"student":18,"technician":19,"writer":20}},inplace=True)
users.replace({"gender":{"M":0,"F":1}},inplace=True)

#faire les vecteurs de similarité de chaque utilisateur pour chaque base
users_vect_movie_1 = np.zeros((u1_base.shape[0], movie.shape[0]))
users_vect_movie_2 = np.zeros((u2_base.shape[0], movie.shape[0]))
users_vect_movie_3 = np.zeros((u3_base.shape[0], movie.shape[0]))
users_vect_movie_4 = np.zeros((u4_base.shape[0], movie.shape[0]))
users_vect_movie_5 = np.zeros((u5_base.shape[0], movie.shape[0]))

#remplissement de ces vecteurs
for index, row in u1_base.iterrows():
    user_ID = row["Utilisateur"]
    film_ID = row["Film"]
    users_vect_movie_1[user_ID - 1, film_ID - 1] = 1 
  
for index, row in u2_base.iterrows():
    user_ID = row["Utilisateur"]
    film_ID = row["Film"]
    users_vect_movie_2[user_ID - 1, film_ID - 1] = 1 
    
for index, row in u3_base.iterrows():
    user_ID = row["Utilisateur"]
    film_ID = row["Film"]
    users_vect_movie_3[user_ID - 1, film_ID - 1] = 1 
    
for index, row in u4_base.iterrows():
    user_ID = row["Utilisateur"]
    film_ID = row["Film"]
    users_vect_movie_4[user_ID - 1, film_ID - 1] = 1 
    
for index, row in u5_base.iterrows():
    user_ID = row["Utilisateur"]
    film_ID = row["Film"]
    users_vect_movie_5[user_ID - 1, film_ID - 1] = 1 
  

def cosine(vecteur1, vecteur2):
    produit_scalaire = np.dot(vecteur1, vecteur2)
    norme_vecteur1 = np.linalg.norm(vecteur1)
    norme_vecteur2 = np.linalg.norm(vecteur2)
    cosinus = produit_scalaire / (norme_vecteur1 * norme_vecteur2)
    return cosinus


def max_cosinus(vecteur1, list_vect):
    cosinus_max = 0
    vect_max = None
    
    list_vect_sans_v1 = [v for v in list_vect if not np.array_equal(v, vecteur1)]
    
    for vect in list_vect_sans_v1:
        cos = cosine(vecteur1, vect)
        if cos > cosinus_max:
            cosinus_max = cos
            vect_max = vect
    return cosinus_max, vect_max



# Calcul de la similarité entre utilisateurs pour chaque base de données
user_ID = 1
 
nb_pers_cos_similaire=50

#TEST Base 1 ----------------------------------------------------------------------------------------------

user_vect_movie_1 = users_vect_movie_1[user_ID - 1]

cosinus_list_1 = []
for i, user_vect in enumerate(users_vect_movie_1):
    if i != user_ID - 1: 
        cosinus = cosine(user_vect_movie_1, user_vect)
        cosinus_list_1.append((cosinus, i + 1))
sorted_cosinus_1 = sorted(cosinus_list_1, key=lambda x: x[0], reverse=True)[:nb_pers_cos_similaire]

print(f"Utilisateur {user_ID} : Les 50 plus grands cosinus de la base 1 sont :")
liste_user_proche_1=[]
for i, (cos, user) in enumerate(sorted_cosinus_1, 1):
    print(f"{i}. Cosinus = {cos}, Utilisateur = {user}")
    liste_user_proche_1.append(user)
    
#TEST Base 2 ----------------------------------------------------------------------------------------------

user_vect_movie_2 = users_vect_movie_2[user_ID - 1]

cosinus_list_2 = []
for i, user_vect in enumerate(users_vect_movie_2):
    if i != user_ID - 1: 
        cosinus = cosine(user_vect_movie_2, user_vect)
        cosinus_list_2.append((cosinus, i + 1)) 
sorted_cosinus_2 = sorted(cosinus_list_2, key=lambda x: x[0], reverse=True)[:nb_pers_cos_similaire]

print(f"Utilisateur {user_ID} : Les 50 plus grands cosinus de la base 2 sont :")
liste_user_proche_2=[]
for i, (cos, user) in enumerate(sorted_cosinus_2, 1):
    print(f"{i}. Cosinus = {cos}, Utilisateur = {user}")
    liste_user_proche_2.append(user)
    
#TEST Base 3 ----------------------------------------------------------------------------------------------

user_vect_movie_3 = users_vect_movie_3[user_ID - 1]

cosinus_list_3 = []
for i, user_vect in enumerate(users_vect_movie_3):
    if i != user_ID - 1: 
        cosinus = cosine(user_vect_movie_3, user_vect)
        cosinus_list_3.append((cosinus, i + 1)) 
sorted_cosinus_3 = sorted(cosinus_list_3, key=lambda x: x[0], reverse=True)[:nb_pers_cos_similaire]

print(f"Utilisateur {user_ID} : Les 50 plus grands cosinus de la base 3 sont :")
liste_user_proche_3=[]
for i, (cos, user) in enumerate(sorted_cosinus_3, 1):
    print(f"{i}. Cosinus = {cos}, Utilisateur = {user}")
    liste_user_proche_3.append(user)

#TEST Base 4 ----------------------------------------------------------------------------------------------


user_vect_movie_4 = users_vect_movie_4[user_ID - 1]

cosinus_list_4 = []
for i, user_vect in enumerate(users_vect_movie_4):
    if i != user_ID - 1: 
        cosinus = cosine(user_vect_movie_4, user_vect)
        cosinus_list_4.append((cosinus, i + 1)) 
sorted_cosinus_4 = sorted(cosinus_list_4, key=lambda x: x[0], reverse=True)[:nb_pers_cos_similaire]

print(f"Utilisateur {user_ID} : Les 50 plus grands cosinus de la base 4 sont :")
liste_user_proche_4=[]
for i, (cos, user) in enumerate(sorted_cosinus_4, 1):
    print(f"{i}. Cosinus = {cos}, Utilisateur = {user}")
    liste_user_proche_4.append(user)

#TEST Base 5 ----------------------------------------------------------------------------------------------


user_vect_movie_5 = users_vect_movie_5[user_ID - 1]

cosinus_list_5 = []
for i, user_vect in enumerate(users_vect_movie_5):
    if i != user_ID - 1: 
        cosinus = cosine(user_vect_movie_5, user_vect)
        cosinus_list_5.append((cosinus, i + 1)) 
sorted_cosinus_5 = sorted(cosinus_list_5, key=lambda x: x[0], reverse=True)[:nb_pers_cos_similaire]

print(f"Utilisateur {user_ID} : Les 50 plus grands cosinus de la base 5 sont :")
liste_user_proche_5=[]
for i, (cos, user) in enumerate(sorted_cosinus_5, 1):
    print(f"{i}. Cosinus = {cos}, Utilisateur = {user}")
    liste_user_proche_5.append(user)

def moyenne_diff_note(ID_user1,ID_user2,base):
    user1_data = base[base['Utilisateur'] == ID_user1]
    user2_data = base[base['Utilisateur'] == ID_user2]
    films_user1 = set(user1_data['Film'])
    films_user2 = set(user2_data['Film'])
    films_communs = films_user1.intersection(films_user2)
    diff_note=0
    for film_ID in films_communs:
        note1 = base[(base['Utilisateur'] == ID_user1) & (base['Film'] == film_ID)]['Note'].values
        note2 = base[(base['Utilisateur'] == ID_user2) & (base['Film'] == film_ID)]['Note'].values
        diff_note+=abs(note1-note2)
    return diff_note/len(films_communs)
    


# Calcul des recommandations de films pour chaque base de données
#TEST Base 1 ----------------------------------------------------------------------------------------------


liste_moyenne_diff_note_1 = []

for user, (cos, _) in zip(liste_user_proche_1, sorted_cosinus_1):
    moyenne_diff = moyenne_diff_note(user_ID, user,u1_base)
    liste_moyenne_diff_note_1.append((user, cos, moyenne_diff[0])) 
    
print("Utilisateur ID - Cosinus - Moyenne Différence de Note")
for user, cos, diff_note in liste_moyenne_diff_note_1:
    print(f"{user} - {cos} - {diff_note}")


cos_max_1=max_cosinus(user_vect_movie_1, users_vect_movie_1)

recommandations_1 = []
for cos, user in sorted_cosinus_1:
    user_data = u1_base[u1_base['Utilisateur'] == user]
    films_user = set(user_data['Film'])
    films_cibles = set(u1_base[u1_base['Utilisateur'] == user_ID]['Film'])
    films_a_recommander = films_user - films_cibles 
    for film_ID in films_a_recommander:
        note_moyenne_film = user_data[user_data['Film'] == film_ID]['Note'].mean()
        diff_note = moyenne_diff_note(user_ID, user,u1_base)
        note_predite_1 = note_moyenne_film - diff_note
        
        note_predite_1 *= cos/cos_max_1[0]
        
        
        recommandations_1.append((film_ID, note_predite_1))

recommandations_1 = sorted(recommandations_1, key=lambda x: x[1], reverse=True)
nb_reco=100
top_recommandations_1 = recommandations_1[:nb_reco]

print(f"Top {nb_reco} des recommandations base 1 :")
for film_ID, note_predite_1 in top_recommandations_1:
    titre_film = movie[movie['movie id'] == film_ID][' movie title'].values[0]
    print(f"Film ID : {film_ID}, Titre : {titre_film}, Note Prédite : {note_predite_1}")

#TEST Base 2 ----------------------------------------------------------------------------------------------

liste_moyenne_diff_note_2 = []

for user, (cos, _) in zip(liste_user_proche_2, sorted_cosinus_2):
    moyenne_diff = moyenne_diff_note(user_ID, user,u2_base)
    liste_moyenne_diff_note_2.append((user, cos, moyenne_diff[0])) 
    
print("Utilisateur ID - Cosinus - Moyenne Différence de Note")
for user, cos, diff_note in liste_moyenne_diff_note_2:
    print(f"{user} - {cos} - {diff_note}")


cos_max_2=max_cosinus(user_vect_movie_2, users_vect_movie_2)


recommandations_2 = []
for cos, user in sorted_cosinus_2:
    user_data = u2_base[u2_base['Utilisateur'] == user]
    films_user = set(user_data['Film'])
    films_cibles = set(u2_base[u2_base['Utilisateur'] == user_ID]['Film'])
    films_a_recommander = films_user - films_cibles 
    for film_ID in films_a_recommander:
        note_moyenne_film = user_data[user_data['Film'] == film_ID]['Note'].mean()
        diff_note = moyenne_diff_note(user_ID, user,u2_base)
        note_predite_2 = note_moyenne_film - diff_note
        
        note_predite_2 *= cos/cos_max_2[0]
        
        
        recommandations_2.append((film_ID, note_predite_2))

recommandations_2 = sorted(recommandations_2, key=lambda x: x[1], reverse=True)
top_recommandations_2 = recommandations_2[:nb_reco]
print(f"Top {nb_reco} des recommandations base 2 :")
for film_ID, note_predite_2 in top_recommandations_2:
    titre_film = movie[movie['movie id'] == film_ID][' movie title'].values[0]
    print(f"Film ID : {film_ID}, Titre : {titre_film}, Note Prédite : {note_predite_2}")

#TEST Base 3 ----------------------------------------------------------------------------------------------

liste_moyenne_diff_note_3 = []

for user, (cos, _) in zip(liste_user_proche_3, sorted_cosinus_3):
    moyenne_diff = moyenne_diff_note(user_ID, user,u3_base)
    liste_moyenne_diff_note_3.append((user, cos, moyenne_diff[0])) 
    
print("Utilisateur ID - Cosinus - Moyenne Différence de Note")
for user, cos, diff_note in liste_moyenne_diff_note_3:
    print(f"{user} - {cos} - {diff_note}")


cos_max_3=max_cosinus(user_vect_movie_3, users_vect_movie_3)

recommandations_3 = []
for cos, user in sorted_cosinus_3:
    user_data = u3_base[u3_base['Utilisateur'] == user]
    films_user = set(user_data['Film'])
    films_cibles = set(u3_base[u3_base['Utilisateur'] == user_ID]['Film'])
    films_a_recommander = films_user - films_cibles 
    for film_ID in films_a_recommander:
        note_moyenne_film = user_data[user_data['Film'] == film_ID]['Note'].mean()
        diff_note = moyenne_diff_note(user_ID, user,u3_base)
        note_predite_3 = note_moyenne_film - diff_note
        
        note_predite_3 *= cos/cos_max_3[0]
        
        
        recommandations_3.append((film_ID, note_predite_3))

recommandations_3 = sorted(recommandations_3, key=lambda x: x[1], reverse=True)
top_recommandations_3 = recommandations_3[:nb_reco]

print(f"Top {nb_reco} des recommandations base 3 :")
for film_ID, note_predite_3 in top_recommandations_3:
    titre_film = movie[movie['movie id'] == film_ID][' movie title'].values[0]
    print(f"Film ID : {film_ID}, Titre : {titre_film}, Note Prédite : {note_predite_3}")


#TEST Base 4 ----------------------------------------------------------------------------------------------


liste_moyenne_diff_note_4 = []

for user, (cos, _) in zip(liste_user_proche_4, sorted_cosinus_4):
    moyenne_diff = moyenne_diff_note(user_ID, user,u4_base)
    liste_moyenne_diff_note_4.append((user, cos, moyenne_diff[0])) 
    
print("Utilisateur ID - Cosinus - Moyenne Différence de Note")
for user, cos, diff_note in liste_moyenne_diff_note_4:
    print(f"{user} - {cos} - {diff_note}")


cos_max_4=max_cosinus(user_vect_movie_4, users_vect_movie_4)

recommandations_4 = []
for cos, user in sorted_cosinus_4:
    user_data = u4_base[u4_base['Utilisateur'] == user]
    films_user = set(user_data['Film'])
    films_cibles = set(u4_base[u4_base['Utilisateur'] == user_ID]['Film'])
    films_a_recommander = films_user - films_cibles 
    for film_ID in films_a_recommander:
        note_moyenne_film = user_data[user_data['Film'] == film_ID]['Note'].mean()
        diff_note = moyenne_diff_note(user_ID, user,u4_base)
        note_predite_4 = note_moyenne_film - diff_note
        
        note_predite_4 *= cos/cos_max_4[0]
        
        
        recommandations_4.append((film_ID, note_predite_4))

recommandations_4 = sorted(recommandations_4, key=lambda x: x[1], reverse=True)
top_recommandations_4 = recommandations_4[:nb_reco]

print(f"Top {nb_reco} des recommandations base 4 :")
for film_ID, note_predite_4 in top_recommandations_4:
    titre_film = movie[movie['movie id'] == film_ID][' movie title'].values[0]
    print(f"Film ID : {film_ID}, Titre : {titre_film}, Note Prédite : {note_predite_4}")



#TEST Base 5 ----------------------------------------------------------------------------------------------


liste_moyenne_diff_note_5 = []

for user, (cos, _) in zip(liste_user_proche_5, sorted_cosinus_5):
    moyenne_diff = moyenne_diff_note(user_ID, user,u5_base)
    liste_moyenne_diff_note_5.append((user, cos, moyenne_diff[0])) 
    
print("Utilisateur ID - Cosinus - Moyenne Différence de Note")
for user, cos, diff_note in liste_moyenne_diff_note_5:
    print(f"{user} - {cos} - {diff_note}")


cos_max_5=max_cosinus(user_vect_movie_5, users_vect_movie_5)

recommandations_5 = []
for cos, user in sorted_cosinus_5:
    user_data = u5_base[u5_base['Utilisateur'] == user]
    films_user = set(user_data['Film'])
    films_cibles = set(u5_base[u5_base['Utilisateur'] == user_ID]['Film'])
    films_a_recommander = films_user - films_cibles 
    for film_ID in films_a_recommander:
        note_moyenne_film = user_data[user_data['Film'] == film_ID]['Note'].mean()
        diff_note = moyenne_diff_note(user_ID, user,u5_base)
        note_predite_5 = note_moyenne_film - diff_note
        
        note_predite_5 *= cos/cos_max_5[0]
        
        
        recommandations_5.append((film_ID, note_predite_5))

recommandations_5 = sorted(recommandations_5, key=lambda x: x[1], reverse=True)
top_recommandations_5 = recommandations_5[:nb_reco]     
 
print(f"Top {nb_reco} des recommandations base 5 :")
for film_ID, note_predite_5 in top_recommandations_5:
    titre_film = movie[movie['movie id'] == film_ID][' movie title'].values[0]
    print(f"Film ID : {film_ID}, Titre : {titre_film}, Note Prédite : {note_predite_5}")


#calcul de la precision des recommandation en fonction de la note qu'il donne aux films qu'on lui recommande
def precision_recommandations(recommandations, test_data, seuil_note, user_ID):
    liste_film_reco_note = []
    liste_film_bien_note = []
    for film_ID, _ in recommandations:
        if film_ID in test_data[test_data['Utilisateur'] == user_ID]['Film'].values:
            liste_film_reco_note.append(film_ID)
            
            note_film = test_data[(test_data['Utilisateur'] == user_ID) & (test_data['Film'] == film_ID)]['Note'].values
            if len(note_film) > 0 and note_film[0] >= seuil_note:
                liste_film_bien_note.append(film_ID)
            
    precision = len(liste_film_bien_note) / len(liste_film_reco_note)
    return precision


#nous evaluons une bonne recommandation si il note la recommandation à plus de 4
seuil_note = 4  

# Précision des recommandations
precision_1 = precision_recommandations(top_recommandations_1, u1_test, seuil_note,user_ID)
print(f"Précision des recommandations base 1 : {precision_1}")

precision_2 = precision_recommandations(top_recommandations_2, u2_test, seuil_note,user_ID)
print(f"Précision des recommandations base 2 : {precision_2}")

precision_3 = precision_recommandations(top_recommandations_3, u3_test, seuil_note,user_ID)
print(f"Précision des recommandations base 3 : {precision_3}")

precision_4 = precision_recommandations(top_recommandations_4, u4_test, seuil_note,user_ID)
print(f"Précision des recommandations base 4 : {precision_4}")

precision_5 = precision_recommandations(top_recommandations_5, u5_test, seuil_note,user_ID)
print(f"Précision des recommandations base 5 : {precision_5}")

precision_moyenne=(precision_1+precision_2+precision_3+precision_4+precision_5)/5
print(f"Précision des recommandations : {precision_moyenne}")



