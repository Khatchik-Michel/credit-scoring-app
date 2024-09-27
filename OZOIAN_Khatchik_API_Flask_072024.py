from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import warnings
import os

# Ignorer les warnings
warnings.simplefilter(action='ignore', category=AttributeError)
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

# Obtenir le chemin de travail actuel
base_path = os.path.dirname(os.path.abspath(__file__))

# Utiliser des chemins relatifs pour le modèle et les images
model_path = os.path.join(base_path, 'model.pkl')
global_importance_path = os.path.join(base_path, 'global_importance.png')
local_importance_path = os.path.join(base_path, 'local_importance.png')

# Charger le modèle initial
model = joblib.load(model_path)

# Obtenir les noms des caractéristiques du modèle
expected_features = model.feature_names_in_

@app.route('/new_model', methods=['POST'])
def new_model():
    """
    Route pour charger un nouveau modèle.
    """
    data = request.get_json(force=True)
    model_id = data.get('model_id')
    if model_id is None:
        return jsonify({"error": "model_id is required"}), 400
    
    # Charger un nouveau modèle basé sur l'ID fourni
    new_model_path = os.path.join(base_path, f'model_{model_id}.pkl')
    
    # Vérifier si le fichier du modèle existe
    if os.path.exists(new_model_path):
        global model  # Changer la variable globale 'model'
        model = joblib.load(new_model_path)
        return jsonify({"message": "Model changed successfully"}), 200
    else:
        return jsonify({"error": "Model not found"}), 404

@app.route('/predict', methods=['POST'])
def predict():
    """
    Route pour effectuer des prédictions sur des données envoyées via POST.
    """
    data = request.get_json(force=True)
    try:
        df = pd.DataFrame(data)  # Convertir les données en DataFrame

        # Ajouter des colonnes manquantes avec une valeur par défaut appropriée
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0  # Valeur par défaut

        # Réordonner les colonnes pour correspondre à l'ordre attendu par le modèle
        df = df[expected_features]

        # Convertir toutes les colonnes en types numériques
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Prédire la probabilité
        prediction = model.predict_proba(df)
        return jsonify(prediction.tolist())
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")  # Loguer l'erreur
        return jsonify({"error": str(e)}), 500

@app.route('/version', methods=['GET'])
def version():
    """
    Route pour obtenir la version actuelle de l'API.
    """
    return jsonify("2.0")

@app.route('/threshold', methods=['GET'])
def threshold():
    """
    Route pour obtenir le seuil de prédiction.
    """
    return jsonify("0.1")

@app.route('/global_importance', methods=['GET'])
def global_importance():
    """
    Route pour récupérer l'image d'importance globale des features.
    """
    if os.path.exists(global_importance_path):
        return send_file(global_importance_path, mimetype='image/png')
    else:
        return "File not found", 404

@app.route('/get_local_importance', methods=['POST'])
def get_local_importance():
    """
    Route pour récupérer l'image d'importance locale des features.
    """
    if os.path.exists(local_importance_path):
        return send_file(local_importance_path, mimetype='image/png')
    else:
        return "File not found", 404


if __name__ == '__main__':
    # Port par défaut : 5001, mais peut être configuré via une variable d'environnement
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
