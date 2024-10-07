import streamlit as st
import pandas as pd
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
        for idx, prediction_list in enumerate(predictions):
            if isinstance(prediction_list, list):
                for score in prediction_list:
                    if isinstance(score, (int, float)) and score > 0.5:
                        accepted = "Accepté" if score >= 0.5 else "Refusé"
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

                        # Feature importance locale
                        if 'local_importance' in predictions[idx]:
                            local_importance = predictions[idx]['local_importance']
                            fig, ax = plt.subplots()
                            sns.barplot(x=list(local_importance.keys()), y=list(local_importance.values()), ax=ax)
                            ax.set_title(f"Importance locale des features pour le client {selected_id}")
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                            st.pyplot(fig)
                        break  # Afficher un seul graphique, celui avec score > 0.5
    elif isinstance(predictions, dict):
        # Si la réponse est un seul dictionnaire
        score = predictions.get('score', 0)
        if isinstance(score, (int, float)) and score > 0.5:
            accepted = "Accepté" if score >= 0.5 else "Refusé"
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

            # Feature importance locale
            if 'local_importance' in predictions:
                local_importance = predictions['local_importance']
                fig, ax = plt.subplots()
                sns.barplot(x=list(local_importance.keys()), y=list(local_importance.values()), ax=ax)
                ax.set_title(f"Importance locale des features pour le client {selected_id}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig)
    else:
        st.write(f"Format inattendu pour les prédictions: {predictions}")

# Visualisation des résultats
if uploaded_file is not None:
    features = data.columns.tolist()
    selected_features = st.multiselect("Choisissez deux features pour l'analyse", features, default=features[:2])
    
    if len(selected_features) == 2 and 'TARGET' in data.columns:
        # Vérifier si les features sélectionnées existent dans les données
        if all(feature in data.columns for feature in selected_features):
            # Graphiques de distribution des features sélectionnées
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            for i, feature in enumerate(selected_features):
                sns.histplot(data, x=feature, hue='TARGET', kde=True, ax=ax[i])
                ax[i].axvline(x=selected_data[feature].values[0], color='red', linestyle='--', label='Client')
                ax[i].legend()
                ax[i].set_title(f"Distribution de {feature} selon les classes")
            st.pyplot(fig)
            
            # Graphique d'analyse bi-variée entre les deux features
            fig, ax = plt.subplots()
            scatter = ax.scatter(data[selected_features[0]], data[selected_features[1]], c=data['TARGET'], cmap='viridis', alpha=0.6)
            ax.scatter(selected_data[selected_features[0]], selected_data[selected_features[1]], color='red', s=100, label='Client')
            ax.set_xlabel(selected_features[0])
            ax.set_ylabel(selected_features[1])
            ax.set_title("Analyse bi-variée entre les deux features avec score en dégradé de couleur")
            plt.colorbar(scatter, label='Score')
            ax.legend()
            st.pyplot(fig)

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
