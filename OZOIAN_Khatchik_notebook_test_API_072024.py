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

    # Choisir entre prédire pour tous les IDs ou un ID spécifique
    predire_tous = st.checkbox("Prédire pour tous les IDs", value=True)

    if predire_tous:
        # Préparation des données pour l'API (tous les IDs)
        data_json = data.to_dict(orient='records')
        st.write("Données prêtes à être envoyées à l'API.")
    else:
        # Liste déroulante pour choisir l'ID
        selected_id = st.selectbox("Choisissez un ID", data['SK_ID_CURR'].unique())
        
        # Préparation des données pour l'API (ID spécifique)
        selected_data = data[data['SK_ID_CURR'] == selected_id]
        data_json = selected_data.to_dict(orient='records')
        st.write("Données prêtes à être envoyées à l'API.")

    # Bouton pour lancer les prédictions
    if st.button("Prédire"):
        # Appel à l'API Flask (URL de ton API déployée sur Render)
        api_url = "https://credit-scoring-app-voah.onrender.com/predict"
        response = requests.post(api_url, json=data_json, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            # Stocker les prédictions dans l'état de la session
            st.session_state['predictions'] = response.json()
            st.write("Prédictions effectuées avec succès.")
            
            # Afficher la structure des prédictions pour comprendre leur format
            st.write("Structure des prédictions:", st.session_state['predictions'])
            
            # Vérification du format des prédictions pour ajouter la colonne TARGET correctement
            if predire_tous:
                for i, pred in enumerate(st.session_state['predictions']):
                    st.write(f"Prédiction pour ID {i}: {pred}")  # Affichage pour vérifier la structure
                    # Supposons que la prédiction soit un float ou dict avec un score direct
                    if isinstance(pred, dict) and 'score' in pred:  # Ajustez 'score' en fonction de la clé correcte
                        data.loc[i, 'TARGET'] = pred['score']  # Changez 'score' pour la clé correcte si nécessaire
                    elif isinstance(pred, float):  # Si la prédiction est un score direct
                        data.loc[i, 'TARGET'] = pred
                    else:
                        st.write(f"Erreur : La structure des prédictions n'est pas reconnue pour l'ID {i}.")
            else:
                # Pour un seul ID, ajout de la prédiction dans 'TARGET'
                pred = st.session_state['predictions'][0]
                if isinstance(pred, dict) and 'score' in pred:  # Ajustez 'score' selon la clé correcte
                    selected_data.loc[selected_data.index, 'TARGET'] = pred['score']
                elif isinstance(pred, float):
                    selected_data.loc[selected_data.index, 'TARGET'] = pred
                else:
                    st.write("Erreur : La structure des prédictions pour l'ID sélectionné n'est pas reconnue.")
        else:
            st.write("Erreur dans l'appel à l'API")
            st.write(f"Status Code: {response.status_code}")
            st.write(f"Message: {response.text}")

    # Si les prédictions sont effectuées, afficher les graphiques
    if st.session_state['predictions'] is not None and 'TARGET' in data.columns:
        st.write("**Distribution des features par classe après prédiction**")
        
        # Liste déroulante pour sélectionner deux features à analyser
        features = data.columns.tolist()
        feature_1 = st.selectbox("Choisissez la première feature", features)
        feature_2 = st.selectbox("Choisissez la deuxième feature", features)

        # Distribution de la première feature par classe
        fig1 = px.histogram(data, x=feature_1, color="TARGET", nbins=50, title=f"Distribution de {feature_1} par classe")
        st.plotly_chart(fig1)

        # Distribution de la deuxième feature par classe
        fig2 = px.histogram(data, x=feature_2, color="TARGET", nbins=50, title=f"Distribution de {feature_2} par classe")
        st.plotly_chart(fig2)

        # Graphique bi-variée avec dégradé de couleur selon le score
        st.write("**Analyse bi-variée des features avec dégradé selon le score**")
        fig3 = px.scatter(data, x=feature_1, y=feature_2, color="TARGET", title="Analyse bi-variée")
        st.plotly_chart(fig3)
    else:
        st.write("Erreur : La colonne 'TARGET' n'a pas été ajoutée correctement.")
else:
    st.write("Veuillez télécharger un fichier CSV pour commencer.")
