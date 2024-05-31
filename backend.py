from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Charger le modèle entraîné
model = tf.keras.models.load_model('model.h5')

# Définir la structure des données d'entrée
class PredictionInput(BaseModel):
    features: list

app = FastAPI()

@app.post("/predict")
def predict(input: PredictionInput):
    features = np.array(input.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": float(prediction[0, 0])}