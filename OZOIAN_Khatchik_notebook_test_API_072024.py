import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("Dashboard de Scoring de Crédit")

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

    # Liste déroulante pour sélectionner deux features à analyser
    features = data.columns.tolist()
    feature_1 = st.selectbox("Choisissez la première feature", features)
    feature_2 = st.selectbox("Choisissez la deuxième feature", features)

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

    # Graphiques des distributions de features
    if st.session_state['predictions'] is not None:
        st.write("**Distribution des features par classe**")
        
        # Distribution de la première feature par classe
        fig1 = px.histogram(data, x=feature_1, color="TARGET", nbins=50, title=f"Distribution de {feature_1} par classe")
        st.plotly_chart(fig1)

        # Distribution de la deuxième feature par classe
        fig2 = px.histogram(data, x=feature_2, color="TARGET", nbins=50, title=f"Distribution de {feature_2} par classe")
        st.plotly_chart(fig2)

    # Graphique bi-variée avec dégradé de couleur selon le score
    if st.session_state['predictions'] is not None:
        st.write("**Analyse bi-variée des features avec dégradé selon le score**")

        if not predire_tous:
            # Ajout du score aux données pour une analyse bi-variée
            score = st.session_state['predictions'][0].get('1')  # Probabilité pour la classe 1
            selected_data['score'] = score
            
            fig3 = px.scatter(data, x=feature_1, y=feature_2, color=selected_data['score'], title="Analyse bi-variée")
            st.plotly_chart(fig3)

# Tableau de bord d'informations sur le client
if not predire_tous:
    st.write("**Informations sur le client sélectionné**")
    st.write(f"ID Client: {selected_id}")
    st.write(f"Score: {score}")
    st.write(selected_data)

# Autres graphiques pour les autres clients
if predire_tous:
    st.write("**Informations sur les autres clients**")
    st.write(data[['SK_ID_CURR', 'TARGET']].head())
