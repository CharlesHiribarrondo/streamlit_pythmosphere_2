# -*- coding: utf-8 -*-

import streamlit as st
import os 
path=os.getcwd()

st.title('About')
st.header("L'équipe Pytmosphere")
st.write('Morgane Andrès : https://www.linkedin.com/in/morgane-andres/')
st.write('Quang Hai Bui : https://www.linkedin.com/in/quang-hai-b-76612672/')
st.write('Charles Hiribarrondo : https://www.linkedin.com/in/charles-hiribarrondo-47354877/')

st.header('Sources')
st.write(" [Agence européenne de l'environnement (EEA)](https://www.eea.europa.eu/data-and-maps/data/co2-cars-emission-20)")
st.write(" [Loi climat et résiliences](https://www.ecologie.gouv.fr/loi-climat-resilience)")
st.write("[Niveau d'émission des véhicules (étiquette énergie)](http://www.vedura.fr/guide/ecolabel/etiquette-energie-CO2-voiture)")