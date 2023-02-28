# -*- coding: utf-8 -*-
import streamlit as st


st.title("Perspectives du projet")

st.write("Nous pouvons à présent essayer de nous projeter sur la suite de Pytmosphère, et sur ses futures mises à jour.")

st.write("### Utilisation de données complémentaires")

st.write("**Il pourrait être intéressant d'intégrer à l'analyse d'autres caractéristiques (features), en particulier :**")
st.write("* La géométrie du véhicule (largeur, longueur, hauteur, volume total)")
st.write("* Le coefficient de pénétration dans l'air")

st.write("Des paramètres plus techniques, en lien avec le moteur par exemple, pourraient également être intégrés")

st.write("**La base de données dont nous disposons est européenne : or il serait intéressant de prendre en compte les véhicules en circulation sur les autres continents, notamment en Asie et en Amérique du nord.**")

st.write("### Autres variables cible")

st.write("**Une autre piste pour le projet serait de ne pas se limiter aux seules émissions de CO2.**")
st.write("En effet, la pollution atmosphérique des véhicules routiers tient aussi beaucoup aux particules fines, dont le trafic automobile est responsable à 54% de leur présence dans nos agglomérations urbaines.")

st.write("Ces particules fines se retrouvent dans les moteurs des véhicules, mais aussi dans leurs pneumatiques (phases de freinage).")
st.write("**S'ouvrir au sujet des particules fines nous permettrait d’avoir une approche bien plus précise :**")
st.write("* On pourrait visualiser que les moteurs diesel sont plus impactants que les moteurs essence par exemple (si on se contente des émissions de CO2, les deux carburants sont équivalents, voir même le diesel pollue moins)")
st.write("* On pourrait intégrer dans notre réflexion les véhicules électriques, qui ont été exclus de notre réflexion pendant ce projet, car leurs émissions WLTP sont nulles à l'utilisation")

st.write("**Finalement, une approche globale en se servant notamment d'outils tels que l'Analyse du Cycle de vie, permettrait d'évaluer la pollution et les émissions sur tout le cycle de vie (y compris fabrication et fin de vie.**")
st.write("Une telle approche permettrait de comparer de manière plus rigoureuse les véhicules thermiques et électriques")

st.write("L'outil Pytmosphere pourrait être un puissant support à la prise de décision pour la conception de véhicules moins polluants :earth_africa:")