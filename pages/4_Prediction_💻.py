# Import packages
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os 
path=os.getcwd()
# ______________________________________________________________________________________________________________________
## Définition des niveaux d'émission pour classification : 
nv=['A','B','C','D','E','F','G']
    

# Fonctions
@st.cache_data
def load_data():
    df = pd.read_csv('data/df_sample.csv', index_col=0)
    df=df.drop(['Ernedc', 'Erwltp', 'De', 'FuelCons','Enedc', 'IT', 'Status'], axis=1)
    #df.iloc[0:4] = df.iloc[0:4].style.format({"m": lambda x: '{:.4f}'.format(x)})
    return df

@st.cache_resource
def get_regressor(clf_name):
    if clf_name == "KNN":
        clf = KNeighborsRegressor(  n_neighbors = 5,
                                    p=1,
                                    weights="distance",
                                    algorithm="auto")
    elif clf_name == "Decision Tree" :
        clf= DecisionTreeRegressor( max_depth=30,
                                    min_samples_leaf=5,
                                    min_samples_split = 7)
    else:
        clf= RandomForestRegressor( n_estimators =1000,
                                    max_depth=20,
                                    min_samples_leaf=1,
                                    min_samples_split = 2,
                                    max_features='auto')
    return clf

def separation_train_test(df,clf_name):
    cars = df.copy()
    cars.index = cars['grp'] + ' - ' + cars['Cn']
    cars = cars.drop(['grp', 'Cn'], axis=1)
    target = cars['Ewltp']
    # target_classe=cars['Classe_emission']
    # target_30=cars['OK_2030']
    data = cars.drop('Ewltp', axis=1)
    X_train0, X_test0, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=204)
    X_train_num = X_train0.select_dtypes(include=['float64', 'int64'])
    X_test_num = X_test0.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler().fit(X_train_num)
    X_train_num_scaled = scaler.transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    X_train_num2 = pd.DataFrame(X_train_num_scaled, X_train_num.index, X_train_num.columns)
    X_test_num2 = pd.DataFrame(X_test_num_scaled, X_test_num.index, X_test_num.columns)
    X_train_cat = X_train0.select_dtypes(include='object')
    X_test_cat = X_test0.select_dtypes(include='object')
    enc = OneHotEncoder(handle_unknown='ignore')
    X_train_cat_enc = enc.fit_transform(X_train_cat).toarray()
    X_test_cat_enc = enc.transform(X_test_cat).toarray()
    X_train_cat2 = pd.DataFrame(X_train_cat_enc, X_train_cat.index, enc.get_feature_names_out())
    X_test_cat2 = pd.DataFrame(X_test_cat_enc, X_test_cat.index, enc.get_feature_names_out())
    # A utiliser pour modèles sensibles au scaling
    X_train_scaled = pd.concat([X_train_num2, X_train_cat2], axis=1)
    X_test_scaled = pd.concat([X_test_num2, X_test_cat2], axis=1)
    # A utiliser pour modèles pour lesquels scaling pas nécessaire
    X_train_not_scaled = pd.concat([X_train_num, X_train_cat2], axis=1)
    X_test_not_scaled = pd.concat([X_test_num, X_test_cat2], axis=1)
    # Pour KNN
    X_train_scaled_50 = X_train_scaled.drop(
        columns=['At1', 'Ct_N1', 'Ct_N1G', 'Ft_e85', 'Ft_lpg', 'Ft_ng', 'Ft_petrol', 'Fm_B', 'Fm_F', 'Fm_M'])
    X_test_scaled_50 = X_test_scaled.drop(
        columns=['At1', 'Ct_N1', 'Ct_N1G', 'Ft_e85', 'Ft_lpg', 'Ft_ng', 'Ft_petrol', 'Fm_B', 'Fm_F', 'Fm_M'])
    if clf_name=="KNN":
        return X_train_scaled_50,X_test_scaled_50,y_train,y_test
    else:
        return X_train_not_scaled,X_test_not_scaled,y_train,y_test

@st.cache_data
def get_mae(y_true,y_estimated):
    return round(mean_absolute_error(y_true,y_estimated),3)

@st.cache_data
def get_rmse(y_true,y_estimated):
    return round(np.sqrt(mean_squared_error(y_true, y_estimated)),3)

@st.cache_data
def get_initial_features(df,groupe,marque):
    ct_selected = df.loc[(df.grp == groupe) & (df.Cn == marque)].Ct.mode()
    m_selected = df.loc[(df.grp == groupe) & (df.Cn == marque)].m.mean()
    W_selected = df.loc[(df.grp == groupe) & (df.Cn == marque)].W.mean()
    At1_selected = df.loc[(df.grp == groupe) & (df.Cn == marque)].At1.mean()
    At2_selected = df.loc[(df.grp == groupe) & (df.Cn == marque)].At2.mean()
    Ft_selected = df.loc[(df.grp == groupe) & (df.Cn == marque)].Ft.mode()
    Fm_selected = df.loc[(df.grp == groupe) & (df.Cn == marque)].Fm.mode()
    ec_selected = df.loc[(df.grp == groupe) & (df.Cn == marque)].ec.mean()
    ep_selected = df.loc[(df.grp == groupe) & (df.Cn == marque)].ep.mean()
    year_selected = int(df.loc[(df.grp == groupe) & (df.Cn == marque)].year.mean())
    return [marque,ct_selected, m_selected, W_selected, At1_selected, At2_selected, Ft_selected, Fm_selected,ec_selected, ep_selected, year_selected,groupe]

@st.cache_data
def transformation(X_origin,df,clf_name):
    df100 = df.copy()
    df100.index = df100['grp'] + ' - ' + df100['Cn']
    df100 = df100.drop(['grp', 'Cn','Ewltp'], axis=1)
    df100_num = df100.select_dtypes(include=['float64', 'int64'])
    df100_cat = df100.select_dtypes(include='object')
    scaler = StandardScaler().fit(df100_num)
    df100_num_scaled = scaler.transform(df100_num)
    df100_num2 = pd.DataFrame(df100_num_scaled, df100.index, df100_num.columns)
    enc = OneHotEncoder().fit(df100_cat)
    df100_cat_enc = enc.transform(df100_cat).toarray()
    df100_cat2 = pd.DataFrame(df100_cat_enc, df100.index, enc.get_feature_names_out())
    df_transformed = pd.concat([df100_num2, df100_cat2], axis=1)
    # -----------------------------------------------------
    X_origin_num=X_origin.select_dtypes(include=['float64', 'int64'])
    X_origin_num_scaled=scaler.transform(X_origin_num)
    X_origin_num2 = pd.DataFrame(X_origin_num_scaled, columns=df100_num.columns)
    X_origin_cat=X_origin.select_dtypes(include='object')
    X_origin_cat_enc = enc.transform(X_origin_cat).toarray()
    X_origin_cat2=pd.DataFrame(X_origin_cat_enc, columns=enc.get_feature_names_out())
    if clf_name=="KNN":
        X_origin_scaled = pd.concat([X_origin_num2, X_origin_cat2], axis=1)
        X_origin_scaled_50 = X_origin_scaled.drop(columns=['At1', 'Ct_N1', 'Ct_N1G', 'Ft_e85', 'Ft_lpg', 'Ft_ng', 'Ft_petrol', 'Fm_B', 'Fm_F', 'Fm_M'])
        return X_origin_scaled_50
    else:
        X_origin_not_scaled = pd.concat([X_origin_num, X_origin_cat2], axis=1)
        return X_origin_not_scaled

@st.cache_data
def create_x_initial(df,groupe,marque):
    features_name = df.columns
    initial_features = get_initial_features(df,groupe,marque)
    dic_10 = dict()
    for i in range(len(features_name)):
        dic_10[features_name[i]] = initial_features[i]
    X_initial = pd.DataFrame(dic_10)
    X_initial=X_initial.drop(columns=['grp','Cn'])
    return X_initial

@st.cache_data
def create_x_tuned(df,groupe,marque,m_tuned,W_tuned,At2_tuned,ec_tuned,ep_tuned,year_tuned,Ft_tuned):
    X_tuned=create_x_initial(df,groupe,marque)
    X_tuned['m']=m_tuned
    X_tuned['W'] = W_tuned
    X_tuned['At2']=At2_tuned
    X_tuned['ec']=ec_tuned
    X_tuned['ep']=ep_tuned
    X_tuned['year']=year_tuned
    X_tuned['Ft']=Ft_tuned
    return X_tuned

@st.cache_data
def give_result(y1,y2):
    cl_pred_y1=pd.cut(y1, bins=[0,100,120,140,160,200,250,np.inf], labels=nv)
    cl_pred_y2=pd.cut(y2, bins=[0,100,120,140,160,200,250,np.inf], labels=nv)
    
    loiClimat1=pd.cut(y1, bins=[0,123,np.inf], labels=['OUI','NON'])
    loiClimat2=pd.cut(y2, bins=[0,123,np.inf], labels=['OUI','NON'])
    
    st.write("Le véhicule d'origine émettait {} g/km de CO2".format(round(y1[0], 2)))
    st.write("Sa classe d'émission est : {}".format(cl_pred_y1[0]))
    if loiClimat1[0]=='OUI':
        st.write(":green[**Le véhicule choisi à l'origine aura le droit d'être commercialisé en 2030**]")
    else: 
        st.write(":red[**Le véhicule choisi à l'origine n'aura pas le droit d'être commercialisé en 2030**]")

    
    st.write("Le véhicule modifié par vos soins émet  {} g/km de CO2 ".format(round(y2[0], 2)))
    st.write("Sa classe d'émission est à présent : {}".format(cl_pred_y2[0]))
    if loiClimat2[0]=='OUI':
        st.write(":green[:earth_africa: **Bravo ! Le véhicule modifié par vos soins aura le droit d'être commercialisé en 2030**  :white_check_mark:]")

# _______________________________________________________________________________________________________________

# Introduction
st.title("Prédictions d'émissions CO2")

# Import donnees
df=load_data()
if st.checkbox('Cacher jeu de données',value=True):
    pass
else:
    st.subheader('Jeu de données')
    st.write(df)

# Selection regresseur
st.subheader('Quel régresseur souhaitez-vous ?')
regressor_name = st.selectbox("Sélection de modèle",("KNN","Decision Tree")) # Random Forest fait planter l'app.

# Choix du véhicule par utilisateur

df=df.drop_duplicates(subset='Cn')
st.subheader("Choisissons maintenant un véhicule")
vehicle_grp = st.selectbox("Groupe du véhicule",df.grp.sort_values().unique())
vehicle_cn = st.selectbox("Modèle du véhicule",df.loc[df.grp==vehicle_grp].Cn.sort_values().unique())

# Changement features du véhicules
st.subheader("Amusons-nous maintenant à changer les caractéristiques d'origine du véhicule")
grp,ct_selected, m_selected, W_selected, At1_selected, At2_selected, Ft_selected, Fm_selected,ec_selected, ep_selected, year_selected,mark=get_initial_features(df,vehicle_grp,vehicle_cn)
slider_m = st.slider("Masse du véhicule (kg)", min_value=float(min(df.m)), max_value=float(max(df.m)),value=float(m_selected), step=10.0)
slider_W = st.slider("W", min_value=float(df.W.min()), max_value=float(df.W.max()), value=float(W_selected), step=5.0)
slider_At2 = st.slider("Entraxe essieux (m)", min_value=float(df.At2.min()), max_value=float(df.At2.max()),value=float(At2_selected), step=5.0)
slider_ec = st.slider("Puissance ec", min_value=float(df.ec.min()), max_value=float(df.ec.max()),value=float(ec_selected), step=5.0)
slider_ep = st.slider("Puissance ep", min_value=0.0, max_value=float(df.ep.max()), value=float(ep_selected), step=2.0)
slider_year = st.slider("Année de production", min_value=int(df.year.min()), max_value=int(df.year.max()),  value=int(year_selected), step=1)


# motor_types = df['Ft'].unique()
# dic_motor_type={'Essence':motor_types[0],
#                 "Diesel":motor_types[1],
#                 "Gaz de pétrole liquéfié (GPL)":motor_types[2],
#                 "Hybride":motor_types[3],
#                 "Gaz naturel":motor_types[4],
#                 "Superéthanol E85":motor_types[5]}
# button = st.radio("Type de motorisation :",dic_motor_type.keys())
# motor=dic_motor_type[button]


X_initial=create_x_initial(df=df.drop(columns='Ewltp'),
                           groupe=vehicle_grp,
                           marque=vehicle_cn)
st.write("Caracteristiques initiales du vehicules :\n",X_initial)


X_tuned=create_x_tuned(df.drop(columns='Ewltp'),
                       vehicle_grp,
                       vehicle_cn,
                       slider_m,
                       slider_W,
                       slider_At2,
                       slider_ec,
                       slider_ep,
                       slider_year,
                       Ft_tuned=Ft_selected)
st.write("Caracteristiques modifiées du vehicules :\n",X_tuned)

# Separation Entrainement, Test
X_train,X_test,y_train,y_test = separation_train_test(df,regressor_name)

# Entrainement du regresseur sur base Entrainement
reg=get_regressor(regressor_name)
reg.fit(X_train,y_train)

# Prediction base Test et erreurs du regresseur
y_pred=reg.predict(X_test)
mae=get_mae(y_test,y_pred)
rmse=get_rmse(y_test,y_pred)

# Comparaison des predictions entre vehicule origine et vehicule modifié
st.subheader("Comparons les résultats")
X_initial_transformed=transformation(X_initial,df,regressor_name)
X_tuned_transformed=transformation(X_tuned,df,regressor_name)
y_pred_initial = reg.predict(X_initial_transformed)
y_pred_tuned = reg.predict(X_tuned_transformed)

# Resultats
give_result(y_pred_initial,y_pred_tuned)

