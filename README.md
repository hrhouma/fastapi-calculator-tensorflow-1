# Commandes

```bash
# Cloner le dépôt
git clone https://github.com/hrhouma/fastapi-calculator-tensorflow-1.git
cd fastapi-calculator-tensorflow-1

# Créer et activer un environnement virtuel
python -m venv myenv
# Sous Windows
myenv\Scripts\activate
# Sous macOS et Linux
source myenv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Entraîner et sauvegarder le modèle
python model.py

# Démarrer le serveur FastAPI
uvicorn backend:app --reload

# Démarrer l'application Streamlit
streamlit run frontend.py
```

# Objectif :
- Le modèle TensorFlow dans cet exemple est un modèle de classification binaire très simple qui prend en entrée un vecteur de 8 caractéristiques et prédit une valeur entre 0 et 1. Plus précisément, il prend des valeurs d'entrée (8 caractéristiques) et prédit si la somme de ces valeurs est supérieure à 4. Voici une description plus détaillée du modèle et de son fonctionnement :

### Entraînement du Modèle

1. **Génération des Données :** Les données d'entraînement sont générées de manière aléatoire. Il y a 1000 échantillons, chacun avec 8 caractéristiques. La sortie (label) est 1 si la somme des caractéristiques de l'échantillon est supérieure à 4, sinon elle est 0.
2. **Définition du Modèle :** Le modèle est un réseau de neurones simple avec deux couches cachées :
   - Une couche dense de 16 neurones avec la fonction d'activation ReLU.
   - Une deuxième couche dense de 8 neurones avec la fonction d'activation ReLU.
   - Une couche de sortie avec un seul neurone et la fonction d'activation sigmoïde pour produire une probabilité (valeur entre 0 et 1).
3. **Compilation et Entraînement du Modèle :** Le modèle est compilé avec l'optimiseur Adam et la perte de binary_crossentropy, puis entraîné sur les données générées pendant 10 époques.

### Prédiction avec le Modèle

Le modèle est utilisé pour prédire si la somme des 8 caractéristiques fournies en entrée est supérieure à 4. L'API FastAPI expose une route `/predict` qui prend en entrée une liste de 8 caractéristiques et renvoie la probabilité prédite par le modèle.

### Exemple de Données

Voici comment les données d'entraînement sont générées et à quoi elles ressemblent :

```python
import numpy as np

# Générer des données d'entraînement factices
def generate_data():
    X = np.random.rand(1000, 8)  # 1000 échantillons, 8 caractéristiques
    y = (X.sum(axis=1) > 4).astype(int)  # 1 si la somme des caractéristiques > 4, sinon 0
    return X, y
```

### Exemple d'Entrée et de Sortie

**Entrée :** Un vecteur de 8 nombres flottants, par exemple : `[0.5, 0.3, 0.8, 0.2, 0.6, 0.9, 0.4, 0.7]`

**Sortie :** Une probabilité, par exemple : `0.85` (ce qui signifie qu'il y a une forte probabilité que la somme des caractéristiques soit supérieure à 4).

### Utilisation dans l'Interface

Dans l'interface Streamlit, l'utilisateur peut entrer les 8 caractéristiques, et l'application affichera la probabilité prédite par le modèle.

Voici un récapitulatif du code du modèle, de l'API backend, et du frontend :

### Code Complet du Modèle (`model.py`)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Générer des données d'entraînement factices
def generate_data():
    X = np.random.rand(1000, 8)  # 1000 échantillons, 8 caractéristiques
    y = (X.sum(axis=1) > 4).astype(int)  # 1 si la somme des caractéristiques > 4, sinon 0
    return X, y

# Créer et entraîner un modèle simple
def train_model():
    X, y = generate_data()
    model = Sequential([
        Dense(16, activation='relu', input_shape=(8,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10)
    model.save('model.h5')

if __name__ == "__main__":
    train_model()
```

### Code Complet du Backend (`backend.py`)

```python
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
```

### Code Complet du Frontend (`frontend.py`)

```python
import streamlit as st
import requests

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000"

st.title("Prédiction avec TensorFlow, FastAPI et Streamlit")

features = [st.number_input(f"Caractéristique {i+1}", format="%f") for i in range(8)]

if st.button("Prédire"):
    response = requests.post(f"{API_URL}/predict", json={"features": features})
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        st.success(f"La prédiction est : {prediction}")
    else:
        st.error("Erreur dans la prédiction")
```

Ce projet simple démontre comment utiliser TensorFlow pour entraîner un modèle, FastAPI pour servir ce modèle via une API, et Streamlit pour créer une interface utilisateur permettant de faire des prédictions.
