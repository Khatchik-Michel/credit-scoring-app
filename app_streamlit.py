import streamlit as st
import pandas as pd
import requests
import json

st.title("Application de Scoring de Crédit")

# Chargement des données
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données")
    st.write(data.head())

    # Préparation des données pour l'API
    data_json = data.to_json(orient='records')

    # Bouton pour lancer les prédictions
    if st.button("Prédire"):
        # Appel à l'API
        response = requests.post("http://127.0.0.1:5001/predict", data=json.dumps(data_json), headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            predictions = response.json()
            st.write("Prédictions")
            st.write(predictions)
        else:
            st.write("Erreur dans l'appel à l'API")

# Visualisation des résultats
if st.button("Visualiser l'importance des features"):
    st.write("Affichage de l'importance des features globales et locales")
    # Exemple d'affichage d'une image sauvegardée
    st.image("path_to_shap_summary.png")
