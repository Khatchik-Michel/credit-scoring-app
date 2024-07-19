import streamlit as st
import requests
import pandas as pd
import json
import numpy as np

# Charger le fichier CSV
file = st.file_uploader("Choisir un fichier CSV", type="csv")

if file is not None:
    data = pd.read_csv(file)

    # Afficher les premières lignes du fichier
    st.write("Aperçu des données :")
    st.write(data.head())

    # Convertir les données en JSON
    data_json = data.to_dict(orient='records')

    # Afficher les données JSON
    st.write("Données JSON : ")
    st.write(data_json)

    # Envoyer la requête POST à l'API
    response = requests.post('http://khatchik.pythonanywhere.com/predict', json=data_json)

    # Afficher la réponse de l'API
    if response.status_code == 200:
        st.write("Prédictions :")
        st.write(response.json())
    else:
        st.write("Erreur lors de la requête à l'API :")
        st.write(response.text)
