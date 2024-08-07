from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import warnings
import os

warnings.simplefilter(action='ignore', category=AttributeError)
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

# Chemin complet vers le modèle pré-entraîné model.pkl
model_path = 'C:/Users/Pc Portable Michel/Downloads/model.pkl'
model = joblib.load(model_path)

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

@app.route('/global_importance', methods=['GET'])
def global_importance():
    # Chemin vers l'image d'importance globale des features
    filepath = 'C:/Users/Pc Portable Michel/Downloads/Implémentez_un_modèle_de_scoring_OZOIAN_Khatchik/OZOIAN_Khatchik_dossier_code_072024/Dashboard/assets/shap_summary.png'
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    else:
        return "File not found", 404

@app.route('/get_local_importance', methods=['POST'])
def get_local_importance():
    data = request.get_json(force=True)
    # Chemin vers l'image d'importance locale des features
    filepath = 'C:/Users/Pc Portable Michel/Downloads/Implémentez_un_modèle_de_scoring_OZOIAN_Khatchik/OZOIAN_Khatchik_dossier_code_072024/Dashboard/assets/local_importance.html'
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='text/html')
    else:
        return "File not found", 404

if __name__ == '__main__':
    # Port 5001 par défaut
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
