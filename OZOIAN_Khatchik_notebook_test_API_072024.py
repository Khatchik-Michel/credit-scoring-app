import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  # Import nécessaire pour les graphiques

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

# Sélection des features pour l'analyse des distributions
if uploaded_file is not None:
    st.subheader("Analyse des Features Sélectionnées")

    # Liste déroulante pour choisir la feature à visualiser
    selected_feature = st.selectbox("Choisissez une feature à visualiser", data.columns)

    # Distribution de la feature en fonction des classes
    if 'TARGET' in data.columns:  # Assurez-vous que la colonne de la classe existe
        fig1 = px.histogram(data, x=selected_feature, color='TARGET', marginal="box", nbins=30,
                            title=f"Distribution de la feature '{selected_feature}' selon les classes")
        st.plotly_chart(fig1, use_container_width=True)

        # Positionnement de la valeur du client
        if not predire_tous:
            client_value = selected_data[selected_feature].values[0]
            fig2 = go.Figure()

            # Ajout de la distribution générale de la feature
            fig2.add_trace(go.Histogram(x=data[selected_feature], nbinsx=30, name='Distribution globale'))

            # Ajout de la valeur du client sous forme de ligne verticale
            fig2.add_trace(go.Scatter(x=[client_value, client_value], y=[0, 10], mode='lines', name=f'Valeur client {client_value}',
                                      line=dict(color='red', width=3)))

            fig2.update_layout(title=f"Positionnement de la valeur client pour '{selected_feature}'",
                               xaxis_title=selected_feature,
                               yaxis_title="Nombre d'occurrences",
                               showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)

# Visualisation des résultats (importance des features)
if st.button("Visualiser l'importance des features"):
    st.write("Affichage de l'importance des features globales et locales")
    # Assurez-vous que les chemins vers les images sont corrects
    st.image("global_importance.png", caption='Importance globale des features')
    st.image("local_importance.png", caption='Importance locale des features')
