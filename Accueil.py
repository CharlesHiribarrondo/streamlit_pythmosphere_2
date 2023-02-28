##############################################
# IMPORT PACKAGES
##############################################

import streamlit as st
import pandas as pd

from PIL import Image
import os 
path=os.getcwd()

st.title('	:earth_africa: Pytmosphere	:car:')
st.header('Contexte')

st.write("La protection de l'environnement est aujourd’hui un sujet de société primordial. De nos décisions au quotidien en tant qu’individu aux décisions stratégiques des politiques mondiales telles que les accords de Paris, nous sommes tous concernés par ce qui apparaît être l’enjeu de notre siècle. Parmi les grandes missions de protection de notre planète figure la réduction rapide des émissions de CO2. Et lorsqu’il s’agit d’analyser la répartition des émissions de CO2, nous ne pouvons ignorer l’impact considérable du secteur des transports.")

graphe_emissions=Image.open('data/emissions.png')
st.image(graphe_emissions)
st.write('Les véhicules particuliers occupent notamment une part de 37% des émissions totales de CO2 dans le secteur du transport.')


#Objectif

st.header('Objectif')

st.write("Le projet Pytmosphère a pour objectif de recenser, regrouper, traiter, analyser, visualiser et enfin prédire les émissions de CO2 de véhicules particuliers à partir de leurs caractéristiques techniques. Toutes nos analyses sont réalisées à partir des données fournies par l’Agence Européenne de l’Environnement.")

st.write("La variable cible de l'étude Pytmosphere est le Ewlpt, exprimé en g de CO2 par km. Il s'agit des émissions de CO2 mesurées selon la norme WLTP, utilisée depuis 2019.")
st.write("Pour les dates antérieures, c'est la norme NEDC qui était utilisée. Nous avons utilisé un facteur de conversion pour obtenir le Ewltp pour les données de 2010 à 2018.")
st.write("Pour cela, nous utiliserons des modèles de Machine Learning : Decision tree et KNN.")
st.write("Les véhicules pourront, grâce à la valeur d'émission de CO2, être classés en niveau d'émission")

# Import des donnees
st.write('#### Données')
st.write('Afin de rendre les données exploitables, nous avons procédé à une étape de nettoyage de celles-ci. Après avoir gardé un échantillon de 600000 lignes comme indiqué précédemment, nous avons procédé à une opération de nettoyage des données.')
st.write('En effet, de nombreux doublons, ainsi que des valeurs manquantes, étaient présents initialement. Nous avons aussi supprimé des caractéristiques non pertinentes pour notre étude')
st.write('En définitive, nous obtenons le jeu de données exploitable suivant :')

df = pd.read_csv('data/df_sample.csv', index_col=0)
df=df.drop(['Ernedc', 'Erwltp', 'De', 'FuelCons','Enedc', 'IT', 'Status'], axis=1)
st.write(df)
st.write(" [Source : Agence européenne de l'environnement (EEA)](https://www.eea.europa.eu/data-and-maps/data/co2-cars-emission-20)")

st.write('#### Description du jeu de données :')

st.write("**Après nettoyage du jeu de données, les variables suivantes sont retenues pour être utilisées dans l'analyse :**")
st.write("* Ct : catégorie du véhicule (voiture, utilitaire...)")
st.write("* m : masse du véhicule (kg)")
st.write("* Empattement (mm)")
st.write("* At1 : largeur de l'essieur directeur (mm)")
st.write("* At2 : largeur des autres essieux (mm)")
st.write("* Ft : type de carburant")
st.write("* Mode combustible")
st.write("* ec : Capacité du moteur (cm3)")
st.write("* ep : Puissance du moteur (KW)")
st.write("* year : Année")
st.write("* grp : Groupe automobile")

st.write("Les modèles des véhicules ont également été conservé afin de prédire les émissions (Ewltp) pour un modèle précis")

st.write("**Voici une description statistique du jeu de données :**")
st.write(df.describe())
