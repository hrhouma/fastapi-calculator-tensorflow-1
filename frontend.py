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