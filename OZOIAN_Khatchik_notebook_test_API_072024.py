import pandas as pd
import streamlit as st
import requests
import json
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

    # Choisir un ID spécifique
    selected_id = st.selectbox("Choisissez un ID", data['SK_ID_CURR'].unique())
    selected_data = data[data['SK_ID_CURR'] == selected_id]

    # Choisir les features à inclure dans la prédiction
    features_to_include = st.multiselect("Choisissez des features à inclure dans la prédiction", options=data.columns.tolist(), default=data.columns.tolist())
    selected_data = selected_data[features_to_include]
    data_json = selected_data.to_dict(orient='records')
    st.write("Données envoyées à l'API après sélection des features:", data_json)
    
    # Bouton pour lancer les prédictions
    if st.button("Prédire"):
        # Appel à l'API Flask (URL de ton API déployée sur Render)
        api_url = "https://credit-scoring-app-voah.onrender.com/predict"
        try:
            response = requests.post(api_url, json=data_json, headers={"Content-Type": "application/json"}, timeout=10)
            response.raise_for_status()
            st.session_state['predictions'] = response.json()
        except requests.exceptions.RequestException as e:
            st.write("Erreur dans l'appel à l'API")
            st.write(f"Message: {str(e)}")

# Afficher les prédictions si elles existent
if st.session_state['predictions'] is not None:
    st.write("Prédictions")
    predictions = st.session_state['predictions']
    st.write("Réponse de l'API:", predictions)  # Afficher la réponse brute de l'API pour vérification
    
    # Vérifier si la réponse est une liste ou un dictionnaire
    if isinstance(predictions, list):
        for prediction in predictions:
            score = prediction.get('score', 0)
            if score > 0.5:
                accepted = "Accepté"
                st.write(f"Crédit: {accepted}, Score: {score}")
                # Jauge pour visualiser le score
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={'text': "Score de Crédit"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "green" if score >= 0.5 else "red"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightcoral"},
                            {'range': [0.5, 1], 'color': "lightgreen"}
                        ]
                    }
                ))
                st.plotly_chart(fig)
    elif isinstance(predictions, dict):
        # Si la réponse est un seul dictionnaire
        score = predictions.get('score', 0)
        if score > 0.5:
            accepted = "Accepté"
            st.write(f"Crédit: {accepted}, Score: {score}")
            # Jauge pour visualiser le score
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Score de Crédit"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "green" if score >= 0.5 else "red"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightcoral"},
                        {'range': [0.5, 1], 'color': "lightgreen"}
                    ]
                }
            ))
            st.plotly_chart(fig)

# Feature importance globale
if st.button("Visualiser l'importance des features"):
    st.write("Affichage de l'importance des features globales et locales")
    # Assurez-vous que les chemins vers les images sont corrects
    if os.path.exists("global_importance.png"):
        st.image("global_importance.png", caption='Importance globale des features')
    else:
        st.write("Image de l'importance globale des features introuvable")
    if os.path.exists("local_importance.png"):
        st.image("local_importance.png", caption='Importance locale des features')
    else:
        st.write("Image de l'importance locale des features introuvable")
