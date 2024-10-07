import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Application de Scoring de Crédit")

# Initialiser les variables de session pour les prédictions et les importances
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'feature_importances' not in st.session_state:
    st.session_state['feature_importances'] = None

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
    predire_tous = st.checkbox("Prédire pour tous les IDs", value=False)

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
        # Appel à l'API Flask (URL de votre API déployée sur Render)
        api_url = "https://credit-scoring-app-voah.onrender.com/predict"
        response = requests.post(api_url, json=data_json, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            result = response.json()
            st.session_state['predictions'] = result['predictions']
            st.session_state['feature_importances'] = result.get('feature_importances', None)
        else:
            st.write("Erreur dans l'appel à l'API")
            st.write(f"Status Code: {response.status_code}")
            st.write(f"Message: {response.text}")

# Afficher les prédictions si elles existent
if st.session_state['predictions'] is not None:
    st.write("Prédictions")
    predictions_df = pd.DataFrame(st.session_state['predictions'])
    st.write(predictions_df)

    # Si un ID spécifique est sélectionné
    if not predire_tous:
        # Afficher le numéro du client
        client_id = selected_id
        st.write(f"Client ID: {client_id}")
        # Récupérer le score du client
        client_score = predictions_df[predictions_df['SK_ID_CURR'] == client_id]['score'].values[0]
        # Déterminer si le crédit est accepté ou non
        threshold = 0.5  # Vous pouvez ajuster le seuil selon votre modèle
        if client_score >= threshold:
            decision = "Crédit accepté"
        else:
            decision = "Crédit refusé"
        st.write(f"Décision: {decision}")
        st.write(f"Score du client: {client_score}")

        # Afficher une jauge pour le score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = client_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Score du Client"},
            gauge = {'axis': {'range': [0, 1]},
                     'bar': {'color': "darkblue"},
                     'steps' : [
                         {'range': [0, threshold], 'color': "red"},
                         {'range': [threshold, 1], 'color': "green"}],
                     'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': threshold}}))
        st.plotly_chart(fig)

        # Afficher la feature importance locale
        if st.session_state['feature_importances'] is not None:
            feature_importance = st.session_state['feature_importances'][str(client_id)]
            feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            st.write("Feature Importance Locale")
            st.bar_chart(feature_importance_df.set_index('Feature'))

    else:
        st.write("Affichage des résultats pour tous les clients non implémenté pour les visualisations individuelles.")

# Visualisation des résultats
if st.button("Visualiser les graphiques"):
    st.write("Affichage de l'importance des features globales et locales")

    if uploaded_file is not None and st.session_state['predictions'] is not None:
        # Sélection de deux features
        features = [col for col in data.columns if col != 'SK_ID_CURR']
        feature_1 = st.selectbox("Sélectionnez la première feature", features)
        feature_2 = st.selectbox("Sélectionnez la deuxième feature", features)

        # Distribution de la première feature
        fig1, ax1 = plt.subplots()
        sns.kdeplot(data=data, x=feature_1, hue='TARGET', ax=ax1)
        client_value = data[data['SK_ID_CURR'] == selected_id][feature_1].values[0]
        ax1.axvline(client_value, color='red', linestyle='--', label='Valeur du client')
        ax1.legend()
        st.pyplot(fig1)

        # Distribution de la deuxième feature
        fig2, ax2 = plt.subplots()
        sns.kdeplot(data=data, x=feature_2, hue='TARGET', ax=ax2)
        client_value = data[data['SK_ID_CURR'] == selected_id][feature_2].values[0]
        ax2.axvline(client_value, color='red', linestyle='--', label='Valeur du client')
        ax2.legend()
        st.pyplot(fig2)

        # Graphique bi-varié entre les deux features
        fig3, ax3 = plt.subplots()
        scores = predictions_df['score']
        sc = ax3.scatter(data[feature_1], data[feature_2], c=scores, cmap='viridis')
        ax3.scatter(data[data['SK_ID_CURR'] == selected_id][feature_1], data[data['SK_ID_CURR'] == selected_id][feature_2], color='red', label='Client')
        ax3.set_xlabel(feature_1)
        ax3.set_ylabel(feature_2)
        ax3.legend()
        plt.colorbar(sc, label='Score')
        st.pyplot(fig3)

        # Afficher l'importance globale des features
        if st.session_state['feature_importances'] is not None:
            global_feature_importance = st.session_state['feature_importances']['global']
            global_feature_importance_df = pd.DataFrame(list(global_feature_importance.items()), columns=['Feature', 'Importance'])
            global_feature_importance_df = global_feature_importance_df.sort_values(by='Importance', ascending=False)
            st.write("Feature Importance Globale")
            st.bar_chart(global_feature_importance_df.set_index('Feature'))
