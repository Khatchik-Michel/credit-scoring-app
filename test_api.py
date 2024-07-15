import streamlit as st
import requests
import pandas as pd
import json
import numpy as np

st.title('API de Scoring de Crédit')

uploaded_file = st.file_uploader("Choisir un fichier CSV pour prédiction", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Remplacer les valeurs infinies et NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    
    # Optimisation des types de données
    for col in data.select_dtypes(include=['float64']).columns:
        data[col] = data[col].astype('float32')
    for col in data.select_dtypes(include(['int64']).columns):
        data[col] = data[col].astype('int32')

    st.write(data)

    if st.button('Faire une prédiction'):
        # Convertir les données en JSON
        try:
            data_json = data.to_dict(orient='records')
            data_json_str = json.dumps(data_json, indent=4)  # Convertir en chaîne JSON
            st.write("Données JSON : ", data_json_str)  # Afficher les données JSON pour vérification
        except Exception as e:
            st.write("Erreur de conversion des données en JSON : ", e)
            st.stop()

        # Utiliser l'URL de l'API déployée sur PythonAnywhere
        try:
            response = requests.post('https://khatchik.pythonanywhere.com/predict', json=data_json)
            if response.status_code == 200:
                predictions = response.json()
                st.write(predictions)
            else:
                st.write('Erreur : ', response.status_code)
                st.write('Contenu de la réponse : ', response.text)
        except Exception as e:
            st.write("Erreur lors de la requête à l'API : ", e)

st.write("Utilisez ce formulaire pour tester l'API de prédiction de crédit.")
