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