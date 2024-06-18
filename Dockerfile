# Utiliser une image de base Python officielle
FROM python:3.9-slim

# Installer les outils de compilation nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances Python
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Copier tous les fichiers du projet dans le conteneur
COPY . .

# Exposer le port sur lequel Streamlit va s'exécuter
EXPOSE 8501

# Commande pour exécuter l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
