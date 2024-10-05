import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
import plotly.graph_objects as go  # Import nécessaire pour la jauge

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
        # Appel à l'API Flask (URL de ton API déployée sur Render)
        api_url = "https://credit-scoring-app-voah.onrender.com/predict"
        response = requests.post(api_url, json=data_json, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            st.session_state['predictions'] = response.json()
        else:
            st.write("Erreur dans l'appel à l'API")
            st.write(f"Status Code: {response.status_code}")
            st.write(f"Message: {response.text}")

# Afficher les prédictions si elles existent
if st.session_state['predictions'] is not None:
    st.write("**Prédictions**")
    st.write(st.session_state['predictions'])

    # Définir le seuil
    threshold = 0.5  # Vous pouvez ajuster le seuil selon vos besoins

    # Si on a prédit pour un seul ID
    if not predire_tous:
        # Extraction de la probabilité de la classe positive (1)
        prediction = st.session_state['predictions'][0]  # On récupère le premier élément
        # Vérifier si les clés sont des chaînes ou des entiers
        if '1' in prediction:
            score = prediction['1']
        elif 1 in prediction:
            score = prediction[1]
        else:
            st.write("Format de prédiction inattendu.")
            score = None

        if score is not None:
            id_curr = selected_id
            st.write(f"**Score pour l'ID {id_curr}: {score:.2f}**")

            # Création de la jauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': f"Score pour l'ID {id_curr}"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, threshold], 'color': "red"},
                        {'range': [threshold, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))

            # Affichage de la jauge
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Si on a prédit pour tous les IDs
        for idx, prediction in enumerate(st.session_state['predictions']):
            # Extraction de la probabilité de la classe positive (1)
            if '1' in prediction:
                score = prediction['1']
            elif 1 in prediction:
                score = prediction[1]
            else:
                st.write(f"Format de prédiction inattendu pour l'index {idx}.")
                continue  # Passer à la prédiction suivante

            id_curr = data['SK_ID_CURR'].iloc[idx]
            st.write(f"**Score pour l'ID {id_curr}: {score:.2f}**")

            # Création de la jauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': f"Score pour l'ID {id_curr}"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, threshold], 'color': "red"},
                        {'range': [threshold, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))

            # Affichage de la jauge
            st.plotly_chart(fig, use_container_width=True)

# Visualisation des résultats
if st.button("Visualiser l'importance des features"):
    st.write("Affichage de l'importance des features globales et locales")
    # Assurez-vous que les chemins vers les images sont corrects
    st.image("global_importance.png", caption='Importance globale des features')
    st.image("local_importance.png", caption='Importance locale des features')
