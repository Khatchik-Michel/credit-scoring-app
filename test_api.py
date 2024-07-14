import streamlit as st
import requests
import pandas as pd

st.title('API de Scoring de Crédit')

uploaded_file = st.file_uploader("Choisir un fichier CSV pour prédiction", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    if st.button('Faire une prédiction'):
        response = requests.post('http://127.0.0.1:5000/predict', json=data.to_dict(orient='records'))
        if response.status_code == 200:
            predictions = response.json()
            st.write(predictions)
        else:
            st.write('Erreur : ', response.status_code)

st.write("Utilisez ce formulaire pour tester l'API de prédiction de crédit.")
