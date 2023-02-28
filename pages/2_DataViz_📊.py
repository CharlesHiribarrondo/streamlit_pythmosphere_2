
# Import packages
import streamlit as st
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

import os 
path=os.getcwd()

df = pd.read_csv('data/df_sample.csv', index_col=0)
df=df.drop(['Ernedc', 'Erwltp', 'De', 'FuelCons','Enedc', 'IT', 'Status'], axis=1)


# Visualisation de données


st.title('Visualisation de données')

# On voit l'évolution grossièrement
# On peut affiner avec des "violin" plots par exemple

st.write('Regardons dans un premier temps la distribution des émissions CO2 de tous les véhicules de notre DataFrame, tous groupes automobiles confondus :')
fig, ax1 = plt.subplots()
df_BMW=df.loc[df['grp']=='BMW GROUP']
ax1=sns.catplot(x='year', y='Ewltp', kind='violin', data=df_BMW, height=5, aspect=2);
locs, labels = plt.xticks();
plt.setp(labels, rotation=30);
ax1.set(ylim=(0, 400))
st.pyplot(ax1)
st.write('A partir de 2017, des bulles se forment en dessous de 50 pour Ewltp. Sans doute un lien avec la venue des véhicules électriques et hybrides')
st.write('En 2021, la valeur médiane était en dessous de 150g/km contre celle en 2010 qui était autour de 200g/km. Ce résultat montre un véritable effort chez les marques automobiles pour réduire le taux des émissions CO2 de leurs voitures.')

st.write('Etudions maintenant plus en détail la distribution de la variable Ewltp en fonction du type de carburant (Fuel type, ou Ft)')

ft_box=sns.catplot(x='Ft', y='Ewltp', kind='box', data=df, height=5, aspect=2, showmeans=True);
locs, labels = plt.xticks();
plt.setp(labels, rotation=30);
ft_box.set(ylim=(0, 400));
st.pyplot(ft_box)
st.write("Comme on pouvait s'y attendre, ce sont les véhicules Hybrides qui émettent le moins de CO2 au km.Les véhicules roulant à l'E85 sont,quant à eux, ceux qui émettent le plus de CO2 au km.")

st.write("Voici une infographie permettant de visualiser le niveau moyen d'émission de CO2 par groupe automobile: ")
infographie=Image.open('data/infographie_ptm.png')
st.image(infographie)

###############################
## Visualisation interactive ##
###############################
st.header('Visualisations interactives')

st.subheader('Matrice de corrélation')
# Matrice de corrélation
df_num=df.select_dtypes(include=['float64', 'int64'])
columns = df_num.columns.tolist()

# Create a multiselect widget to select the variables to include in the correlation matrix
selected_columns = st.multiselect('Selection des variables :', columns)
# Vérifier si le nombre de colonnes sélectionnées > 2
if len(selected_columns) < 2:
    st.write('Veuillez choisir au moins deux variables pour analyser la corrélation')
else:
    # Choisir uniquement les variables sélectionnées
    selected_df = df[selected_columns]

   
    corr = selected_df.corr()

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(corr, annot=True,
                xticklabels=corr.columns,
                yticklabels=corr.columns, ax=ax)


    st.write('Matrice de corrélation :')
    st.pyplot(fig)


#Corrélations en grille 
st.subheader('Grille de corrélation')
# Create a multiselect widget to select the variables to include in the correlation matrix
selected_columns2 = st.multiselect('Selection variables graphe corrélation', columns)
# Vérifier si le nombre de colonnes sélectionnées > 2
if len(selected_columns2) < 2:
    st.write('Veuillez choisir au moins deux variables pour analyser la corrélation')
else:
    # Choisir uniquement les variables sélectionnées
    selected_columns2.append('Ft')
    selected_df2 = df[selected_columns2]

    fig2 = sns.pairplot(selected_df2, hue='Ft')

    st.write('Corrélation variables continues :')
    st.pyplot(fig2)


# Evolution d'Ewltp par marque
st.subheader('Evolution Ewltp par marque entre 2010 et 2021')

def create_plot(df):
    makers = df['grp'].unique()
    maker = st.selectbox('Groupe automobile :', makers)
    #st.write(f"Selected maker: {maker}") # print selected maker
    data = df[df['grp'] == maker]
    chart_data = data.groupby('year')['Ewltp'].mean()
    st.bar_chart(chart_data, height=500)

# Example usage
create_plot(df)



