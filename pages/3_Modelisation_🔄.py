# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:15:10 2023

@author: morga
"""
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

import scipy.stats as stats
from PIL import Image

import os 
path=os.getcwd()

df = pd.read_csv("data/df_sample.csv", index_col=0)
df=df.drop(['Ernedc', 'Erwltp', 'De', 'FuelCons','Enedc', 'IT', 'Status'], axis=1)


st.title('Rapport de modélisation')
# Pre-processing
st.header('Pré-processing')
st.write('Nous avons réalisé les étapes de pré-processing suivantes :')
st.write('1. Séparation des données entre une base entrainement (80%) et une base test (20%). La fonction train_test_split de sklearn.model_selection a été utilisée.')
st.write('2. Standardisation des données issues de variables numériques. Pour cela, nous avons utilisé la fonction StandardScaler de sklearn.preprocessing.')
st.write('3. Encoding des variables catégorielles, avec la fonction OneHotEncoder de sklearn.preprocessing.')
st.write('4. Reduction de dimension du jeu de données pour accélérer les étapes de calcul. Soit nous réduisions le nombre de colonnes (features selection), soit le nombre de lignes (échantillonnage)')

cars=df
cars.index=cars['grp']+ ' - ' + cars['Cn']
cars=cars.drop(['grp','Cn'], axis=1)
target=cars['Ewltp']
#target_classe=cars['Classe_emission']
#target_30=cars['OK_2030']
data=cars.drop('Ewltp', axis=1)
X_train0, X_test0, y_train, y_test=train_test_split(data, target, test_size=0.2, random_state=204)
X_train_num=X_train0.select_dtypes(include=['float64', 'int64'])
X_test_num=X_test0.select_dtypes(include=['float64', 'int64'])
scaler=StandardScaler().fit(X_train_num)
X_train_num_scaled=scaler.transform(X_train_num)
X_test_num_scaled=scaler.transform(X_test_num)
X_train_num2=pd.DataFrame(X_train_num_scaled, X_train_num.index, X_train_num.columns)
X_test_num2=pd.DataFrame(X_test_num_scaled, X_test_num.index, X_test_num.columns)
X_train_cat=X_train0.select_dtypes(include='object')
X_test_cat=X_test0.select_dtypes(include='object')
enc = OneHotEncoder(handle_unknown='ignore')
X_train_cat_enc=enc.fit_transform(X_train_cat).toarray()
X_test_cat_enc=enc.transform(X_test_cat).toarray()
X_train_cat2=pd.DataFrame(X_train_cat_enc, X_train_cat.index,enc.get_feature_names_out())
X_test_cat2=pd.DataFrame(X_test_cat_enc, X_test_cat.index,enc.get_feature_names_out())
#A utiliser pour modèles sensibles au scaling
X_train_scaled=pd.concat([X_train_num2, X_train_cat2],axis=1 )
X_test_scaled=pd.concat([X_test_num2, X_test_cat2],axis=1 )
# A utiliser pour modèles pour lesquels scaling pas nécessaire
X_train=pd.concat([X_train_num, X_train_cat2],axis=1 )
X_test=pd.concat([X_test_num, X_test_cat2],axis=1 )

st.write('Et voilà ! Nous pouvons à présent commencer à mettre en place nos modèles de Machine Learning pour prédire nos émissions de CO2 !')

st.header('Algorithmes de régression')

st.write('La métrique choisie pour évaluer la qualité des modèles réalisés est la MAE (mean absolute error)')
# Régression Linéaire
st.subheader('Régression Linéaire')
st.write('Nous avons dans un premier temps choisi de réaliser les prédictions de la variable cible, Ewltp, à partir de modèles de régressions linéaires. Ce sont des modèles simples à interpréter, mais sont-ils pertinents ?')
X_train_scaled_num=X_train_scaled[['m','W', 'At1', 'At2', 'ec', 'ep', 'year']]
X_test_scaled_num=X_test_scaled[['m','W', 'At1', 'At2', 'ec', 'ep', 'year']]
lr=LinearRegression()
lr.fit(X_train_scaled_num, y_train)
y_pred=lr.predict(X_test_scaled_num)
pred_train = lr.predict(X_train_scaled_num)
st.write('Nous obtenons les erreurs absolues (MAE) et carrées (RMSE) entre nos prédictions et les valeurs réelles suivantes :')
st.write('MAE test :', round(mean_absolute_error(y_test,y_pred),2),'MAE train :', round(mean_absolute_error(y_train,pred_train),2),'RMSE:', round(np.sqrt(mean_squared_error(y_test, y_pred)),2))
#st.write("R² ensemble entrainement :", round(lr.score(X_train_scaled_num, y_train),3))
#st.write("R² ensemble test :",round(lr.score(X_test_scaled_num, y_test),3))
residus = pred_train - y_train
fig, ax2 = plt.subplots()
ax2=plt.scatter(y_train, residus, color='#980a10', s=15)
plt.plot((y_train.min(), y_train.max()), (0, 0), lw=3, color='#0a5798');
plt.title('représentation des résidus')
plt.xlabel('résidus')
plt.ylabel('y_train')
st.pyplot(fig);
st.write("Les données ne respectent pas l'hypothèse d’homoscédasticité, car les points représentant y_train en fonction des résidus ne sont pas répartis aléatoirement autour de la droite y=0")
residus_norm=(residus-residus.mean())/residus.std()
fig, ax3 = plt.subplots()
ax3=stats.probplot(residus_norm, plot=plt);
st.pyplot(fig);
st.write("L’hypothèse de normalité n’est pas non plus remplie, car lorsqu’on trace le diagramme Quantile-Quantile, les points ne sont pas alignés sur la première bissectrice. La distribution des résidus ne suit donc pas une loi gaussienne normalisée.")
st.write("Conclusion : Les modèles de régression linéaire ne sont pas pertinents pour notre projet. Nous passons donc à présent à des modèles non linéaires.")

###################
## DECISION TREE ##
###################

st.subheader('Decision Tree Regressor')

st.write("Un premier modèle a été créé avec la fonction DecisionTreeRegressor, avec les paramètres par défaut. Ce modèle a ensuite été ajusté sur les données d’entraînement, puis optimisé par validation croisée")
st.write("Les paramètres sélectionnés sont les suivants : _max_depth=30, min_samples_leaf= 5, min_samples_split=7_")
st.write("Les résultats obtenus sont les suivants : ")

DT_res=Image.open('data/DT_res.png')
st.image(DT_res)

st.write("La validation croisée à permis de diminuer l'overfitting, en conservant de bons résultats")
st.write("Si on prend un véhicule qui émet 166 g de CO2 par km, une MAE de 4.62 aurait pour conséquence une erreur inférieure à 3% : le modèle Decision Tree est donc meilleur et plus précis que le modèle de régression linéaire, comme on pouvait s’y attendre. ")

###################
## Random Forest ##
###################

st.subheader('Random Forest Regressor')

st.write("On a ensuite réalisé un test avec l'algorithme RandomForestRegressor.")
st.write("Comme pour le DecisionTreeRegressor, le modèle a d'abord été créé et entraîné avec les paramètres par défaut, puis optimisé.")
st.write("Les paramètres sélectionnés sont les suivants : _n_estimators=1000,min_samples_split=2,min_samples_leaf=1,max_features='auto',max_depth=20")

RF_res=Image.open('data/RF_res.png')
st.image(RF_res)

st.write("La validation croisée n'améliore pas les résultats et ne change rien au niveau de l'overfitting : le modèle par défaut sera donc retenu.")
st.write("Si on prend un véhicule qui émet 166 g de CO2 par km, une MAE de 3.69 g/km aurait pour conséquence une erreur proche de 2% : le modèle Random Forest est donc particulièrement performant. ")

###################
###### KNN ########
###################
st.subheader('KNN Regressor')

st.write("On a finalement fait un test avec l'algorithme des K-plus-proche-voisins (KNN : KNeighborsRegressor).")
st.write("De la même manière qu'avec les deux autres algorithmes, le modèle a d'abord été créé et entraîner avec les paramètres par défaut, puis optimisé.")
st.write("Les paramètres sélectionnés sont les suivants : _n_neighbors= 5, p= 1, weights= 'distance', algorithm= 'auto'_")

st.write("Les résultats obtenus sont les suivants : ")
KNN_res=Image.open('data/KNN_res.png')
st.image(KNN_res)

st.write("La validation croisée a permis d'améliorer la qualité des résultats.")
st.write("Si on prend un véhicule qui émet 166 g de CO2 par km, une MAE de 4.09 aurait pour conséquence une erreur inférieure à 2.50% : le modèle KNN présente donc de bons résultats.")

##################
##Classification##
##################
st.header("Classification à partir des résultats de la régression")
st.subheader("Classification des véhicules par niveau d'émission")

st.write("Afin de mieux comprendre les résultats obtenus, nous avons décidé, à partir de la prédiction des émissions de CO2, de classer les véhicules du jeu de donné en niveau d'émission")
st.write("Les classes utilisées sont celles définies par l'infographie suivante : ")

niveau_emission=Image.open('data/niveau_emission.jpg')
st.image(niveau_emission)

st.write("Les résultats obtenus avec l'algorithme KNN ont été utilisés pour déterminer le niveau d'émission de chaque véhicule du jeu de données.")
st.write("Pour évaluer la qualité de cette classification, nous avons utilisé la métrique f1-score.")
st.write("Voici les résultats obtenus pour chaque classe : ")
classif=Image.open('data/classif.png')
st.image(classif)

st.write("Les résultats obtenus sont très satisfaisants, le modèle KNN mis au point est donc efficace pour une classification multiclasse en niveau d'émissions.")

st.subheader("Loi climat et objectifs pour 2030")


st.write("L’Assemblée nationale et le Sénat ont adopté en 2021 le projet de loi Climat & Résilience. ")
st.write("Un des articles de cette loi annonce la fin de vente des véhicules émettant plus de 95 g de CO2/km (123 g/km en émission WLTP) en 2030.")


st.write("Encore une fois avec l'algorithme KNN, nous avons réalisé la classification des véhicules pour répondre à la question : 'Ce véhicule sera-t-il autorisé à la vente en 2030 ?'")
st.write("Les classes sont donc Oui et Non.")

st.write("Les résultats obtenus sont les suivants : ")
loi_climat=Image.open('data/loi_climat.png')
st.image(loi_climat)

st.write("Les données sont déséquilibrées, et une grande proportion des véhicules est dans la classe _Non_.")
st.write("Le f1 score pour la classe _Non_ est de _0.99_, et de _0.88_ pour la classe _Oui_, ce qui est très satisfaisant.")




