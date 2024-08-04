from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import gc
import warnings

warnings.simplefilter(action='ignore', category=AttributeError)
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

# Charger le modèle pré-entraîné
model = joblib.load('model.pkl')

# Obtenir les noms des caractéristiques du modèle
expected_features = model.feature_names_in_

@app.route('/new_model', methods=['POST'])
def new_model():
    data = request.get_json(force=True)
    model_id = data.get('model_id')
    if model_id is None:
        return jsonify({"error": "model_id is required"}), 400
    # Logique pour changer de modèle
    return jsonify({"message": "Model changed successfully"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        df = pd.DataFrame(data)  # Convertir les données en DataFrame

        # Ajouter des colonnes manquantes avec une valeur par défaut appropriée
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0  # Ou une autre valeur par défaut appropriée

        # Réordonner les colonnes pour correspondre à l'ordre attendu par le modèle
        df = df[expected_features]

        # Convertir toutes les colonnes en types numériques
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        prediction = model.predict_proba(df)
        return jsonify(prediction.tolist())
    except Exception as e:
        print("Erreur lors de la prédiction:", e)  # Loguer les erreurs
        return jsonify({"error": str(e)}), 500

@app.route('/version', methods=['GET'])
def version():
    return jsonify("2.0")

@app.route('/threshold', methods=['GET'])
def threshold():
    return jsonify("0.1")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
