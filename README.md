# Commandes

```bash
# Cloner le dépôt
git clone https://votre-repo-git.git
cd votre-repo

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
