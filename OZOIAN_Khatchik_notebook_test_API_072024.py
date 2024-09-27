import streamlit as st
import pandas as pd
import requests
import json
import numpy as np

st.title("Application de Scoring de Crédit")

# Initialiser les variables de session pour les prédictions
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None

# Chargement des données
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données")
    st.write(data.head())

    # Nettoyage des données
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Conversion des colonnes en types numériques
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Choisir entre prédire pour tous les IDs ou un ID spécifique
    predire_tous = st.checkbox("Prédire pour tous les IDs", value=True)

    if predire_tous:
        # Préparation des données pour l'API (tous les IDs)
        data_json = data.to_dict(orient='records')
        st.write("Données envoyées à l'API:", data_json)  
    else:
        # Liste déroulante pour choisir l'ID
        selected_id = st.selectbox("Choisissez un ID", data['SK_ID_CURR'].unique())
        
        # Préparation des données pour l'API (ID spécifique)
        selected_data = data[data['SK_ID_CURR'] == selected_id]
        data_json = selected_data.to_dict(orient='records')
        st.write("Données envoyées à l'API:", data_json) 
    
    # Bouton pour lancer les prédictions
    if st.button("Prédire"):
        # Appel à l'API
        response = requests.post("http://127.0.0.1:5001/predict", json=data_json, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            st.session_state['predictions'] = response.json()
        else:
            st.write("Erreur dans l'appel à l'API")
            st.write(f"Status Code: {response.status_code}")
            st.write(f"Message: {response.text}")

# Afficher les prédictions si elles existent
if st.session_state['predictions'] is not None:
    st.write("Prédictions")
    st.write(st.session_state['predictions'])

# Visualisation des résultats
if st.button("Visualiser l'importance des features"):
    st.write("Affichage de l'importance des features globales et locales")
    # Assurez-vous que les chemins vers les images sont corrects
    st.image("C:/Users/Pc Portable Michel/Downloads/app/global_importance.png", caption='Importance globale des features')
    st.image("C:/Users/Pc Portable Michel/Downloads/app/local_importance.png", caption='Importance locale des features')
