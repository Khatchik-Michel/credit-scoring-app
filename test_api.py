import streamlit as st
import requests
import pandas as pd
import json

st.title('API de Scoring de Crédit')

uploaded_file = st.file_uploader("Choisir un fichier CSV pour prédiction", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    if st.button('Faire une prédiction'):
        # Convertir les données en JSON
        try:
            data_json = data.to_dict(orient='records')
            data_json_str = json.dumps(data_json)  # Convertir en chaîne JSON
            st.write("Données JSON : ", data_json_str)
        except Exception as e:
            st.write("Erreur de conversion des données en JSON : ", e)
            st.stop()

        # Envoyer les données à l'API
        try:
            response = requests.post('http://127.0.0.1:5000/predict', json=data_json)
            if response.status_code == 200:
                predictions = response.json()
                st.write(predictions)
            else:
                st.write('Erreur : ', response.status_code)
        except Exception as e:
            st.write("Erreur lors de la requête à l'API : ", e)

st.write("Utilisez ce formulaire pour tester l'API de prédiction de crédit.")
