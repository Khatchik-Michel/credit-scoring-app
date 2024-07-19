import pandas as pd
import requests
import streamlit as st

# Titre de l'application
st.title("Test de l'API de Prédiction")

# Charger le fichier CSV
file = st.file_uploader("Choisir un fichier CSV", type="csv")

if file is not None:
    data = pd.read_csv(file)

    # Afficher les premières lignes du fichier
    st.write("Aperçu des données :")
    st.write(data.head())

    # Convertir les données en JSON
    data_json = data.to_dict(orient='records')

    # Afficher les données JSON pour vérification
    st.write("Données JSON : ")
    st.write(data_json)

    # Envoyer la requête POST à l'API
    try:
        response = requests.post('http://127.0.0.1:5000/predict', json=data_json)

        # Afficher la réponse de l'API
        if response.status_code == 200:
            st.write("Prédictions :")
            st.write(response.json())
        else:
            st.write("Erreur lors de la requête à l'API :")
            st.write(response.text)
    except requests.exceptions.RequestException as e:
        st.write("Erreur lors de la requête à l'API :")
        st.write(str(e))
