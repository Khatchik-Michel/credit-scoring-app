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

def optimize_int_memory(df):
    for col in df.select_dtypes(include=['int64', 'Int64']).columns:
        max_val = df[col].max()
        min_val = df[col].min()
        if min_val >= 0:
            if max_val <= 255:
                df[col] = df[col].astype('uint8')
            elif max_val <= 65535:
                df[col] = df[col].astype('uint16')
            elif max_val <= 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if min_val >= -128 and max_val <= 127:
                df[col] = df[col].astype('int8')
            elif min_val >= -32768 and max_val <= 32767:
                df[col] = df[col].astype('int16')
            elif min_val >= -2147483648 and max_val <= 2147483647:
                df[col] = df[col].astype('int32')
    return df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    
    # Optimisation de la mémoire
    df = optimize_int_memory(df)
    
    prediction = model.predict(df)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)