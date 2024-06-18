
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle

# Définition de la configuration de la page
st.set_page_config(page_title="Dashboard Score de Crédit", layout="wide")

# Fonction pour charger les données clients depuis un fichier CSV
@st.cache
def load_data(filename):
    return pd.read_csv(filename)

# Fonction pour afficher la jauge de score
def jauge_score(proba):
    proba_value = proba if proba is not None else 0

    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=proba_value * 100,
        mode="gauge+number+delta",
        title={'text': "Jauge de score"},
        delta={'reference': 52},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
               'bar': {'color': "MidnightBlue"},
               'steps': [
                   {'range': [0, 20], 'color': "Green"},
                   {'range': [20, 45], 'color': "LimeGreen"},
                   {'range': [45, 52], 'color': "Orange"},
                   {'range': [52, 100], 'color': "Red"}],
               'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1, 'value': 52}}))

    st.plotly_chart(fig)

# Fonction pour récupérer le score de crédit du client à partir de l'API
def get_credit_score(client_data):
    api_url = "https://modelia.azurewebsites.net/predict/"
    response = requests.post(api_url, json=client_data)
    try:
        response_data = response.json()
    except requests.exceptions.JSONDecodeError:
        st.error(f"Erreur lors de l'appel à l'API : {response.status_code} - Réponse non-JSON reçue")
        st.error(response.text)
        return None, None

    if response.status_code == 200:
        prediction = response_data.get('prediction', None)
        if prediction is not None:
            decision = "Refusé" if prediction == 1 else "Accordé"
            return prediction, decision
        else:
            st.error("La prédiction n'est pas disponible dans la réponse de l'API.")
            return None, None
    else:
        st.error(f"Erreur lors de l'appel à l'API : {response.status_code} - {response_data}")
        return None, None

# Fonction pour récupérer la probabilité de défaut du client à partir de l'API
def get_proba(client_data):
    api_url = "https://modelia.azurewebsites.net/predict_proba/"
    response = requests.post(api_url, json=[client_data])
    try:
        response_data = response.json()
    except requests.exceptions.JSONDecodeError:
        st.error(f"Erreur lors de l'appel à l'API : {response.status_code} - Réponse non-JSON reçue")
        st.error(response.text)
        return None

    if response.status_code == 200:
        proba_list = response_data.get('predicted_proba', None)
        if proba_list is not None and len(proba_list) > 0:
            return proba_list[0]
        else:
            st.error("La probabilité n'est pas disponible dans la réponse de l'API.")
            return None
    else:
        st.error(f"Erreur lors de l'appel à l'API : {response.status_code} - {response_data}")
        return None

# Fonction pour afficher le waterfall plot SHAP local
def display_shap_waterfall_plot(shap_values, X, index):
    st.subheader("SHAP Waterfall Plot (Local)")
    fig, ax = plt.subplots()
    num_features = len(X.columns)
    shap.waterfall_plot(shap.Explanation(values=shap_values[index], base_values=explainer2.expected_value, data=X.iloc[index]), max_display=num_features, show=False)
    st.pyplot(fig)

# Fonction pour afficher le SHAP summary plot global
def display_shap_summary_plot(shap_values, X):
    st.subheader("SHAP Summary Plot (Global)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig)

# Fonction pour afficher le nuage de points des clients
def scatter_plot_clients(data, selected_client_id=None):
    st.subheader("Nuage de Points des Clients")
    feature_x = st.selectbox("Choisir la caractéristique pour l'axe X :", data.columns)
    feature_y = st.selectbox("Choisir la caractéristique pour l'axe Y :", data.columns)
    fig, ax = plt.subplots()

    # Tracer les points des clients
    sns.scatterplot(x=data[feature_x], y=data[feature_y], ax=ax)

    # Si un client est sélectionné, tracer ce point en rouge
    if selected_client_id is not None:
        selected_client_data = data[data['SK_ID_CURR'] == selected_client_id]
        sns.scatterplot(x=selected_client_data[feature_x], y=selected_client_data[feature_y], color='red', s=100, ax=ax)

    st.pyplot(fig)

# Chargement des données clients à partir du fichier CSV
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

# Charger la pipeline enregistrée
pipeline_path = "model.pkl"
with open(pipeline_path, 'rb') as file:
    loaded_pipeline = pickle.load(file)

# Extraire le modèle du pipeline pour SHAP
model2 = loaded_pipeline.named_steps['classifier']

# Créer l'explainer SHAP avec le modèle extrait
explainer2 = shap.TreeExplainer(model2)

# Chargement des données clients
data = load_data("test_df.csv")

# Titre de la page
st.title("Dashboard Score de Crédit")
# Ajout de l'image de l'entreprise sur la page d'accueil
st.image("image.png", use_column_width=True)

# Navigation entre les onglets
tabs = st.sidebar.radio("Navigation", ("Accueil", "Prédiction et SHAP", "Visualisation des Clients", "Analyse Bi-Variée et Distribution"))

# Onglet : Prédiction et SHAP
if tabs == "Prédiction et SHAP":
    st.subheader("Prédiction de Crédit et Analyse SHAP")

    # Sélection du type de client : existant ou nouveau
    client_type = st.radio("Choisir le type de client :", ("Client existant", "Nouveau client"))

    # Onglet : Client existant
    if client_type == "Client existant":
        client_id = st.selectbox("ID Client", data['SK_ID_CURR'].unique())
        client_data = data[data['SK_ID_CURR'] == client_id].iloc[0].to_dict()

        # Bouton pour afficher les informations du client sous forme de tableau horizontal
        if st.button("Afficher les informations du client", key="display_client_info_button"):
            st.subheader("Informations du Client")
            st.markdown("<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)
            st.markdown("<style>div.Widget.row-widget.stNumberInput > div{flex-direction:row;}</style>", unsafe_allow_html=True)
            st.markdown("<style>div.Widget.row-widget.stTextInput > div{flex-direction:row;}</style>", unsafe_allow_html=True)

            st.write("<div style='display:flex; flex-wrap: wrap;'>", unsafe_allow_html=True)
            for key, value in client_data.items():
                st.write(f"<div style='margin-right:30px;'><b>{key}</b>: {value}</div>", unsafe_allow_html=True)
            st.write("</div>", unsafe_allow_html=True)

        # Bouton pour envoyer les données du client existant pour prédiction
        if st.button("Envoyer pour Prédiction", key="existing_client_predict_button"):
            prediction, decision = get_credit_score(client_data)
            if prediction is not None:
                st.write(f"Prédiction de défaut du client : {prediction}")
                st.write(f"Décision : {decision}")

                proba = get_proba(client_data)
                if proba is not None:
                    st.write(f"Probabilité de défaut du client : {proba:.2f}")
                    jauge_score(proba)

                    # Calculer les valeurs SHAP pour les features
                    X_val_test = data.drop(columns=['SK_ID_CURR'])
                    shap_values = explainer2.shap_values(X_val_test)
                    selected_index = data.index[data['SK_ID_CURR'] == client_id].tolist()[0]

                    display_shap_waterfall_plot(shap_values, X_val_test, selected_index)

            # Afficher le SHAP summary plot global
            st.subheader("SHAP Summary Plot (Global)")
            display_shap_summary_plot(shap_values, X_val_test)

    # Onglet : Nouveau client
    elif client_type == "Nouveau client":
        st.subheader("Informations du Nouveau Client")

                # Widgets pour saisir les informations du nouveau client
        days_employed = st.number_input("Jours d'emploi", value=0)
        days_birth = st.number_input("Jours de naissance", value=0)
        ext_source_3 = st.number_input("EXT_SOURCE_3", value=0.0)
        days_id_publish = st.number_input("DAYS_ID_PUBLISH", value=0)
        code_gender = st.number_input("CODE_GENDER", value=0)
        flag_own_car = st.number_input("FLAG_OWN_CAR", value=0)
        ext_source_2 = st.number_input("EXT_SOURCE_2", value=0.0)
        ext_source_1 = st.number_input("EXT_SOURCE_1", value=0.0)
        name_education_type_highereducation = st.number_input("NAME_EDUCATION_TYPE_Highereducation", value=0)
        name_contract_type_cashloans = st.number_input("NAME_CONTRACT_TYPE_Cashloans", value=0)
        hour_appr_process_start = st.number_input("HOUR_APPR_PROCESS_START", value=0)
        name_family_status_married = st.number_input("NAME_FAMILY_STATUS_Married", value=0)
        flag_phone = st.number_input("FLAG_PHONE", value=0)
        amt_income_total = st.number_input("AMT_INCOME_TOTAL", value=0.0)
        amt_credit = st.number_input("AMT_CREDIT", value=0.0)
        days_registration = st.number_input("DAYS_REGISTRATION", value=0)
        income_credit_perc = st.number_input("INCOME_CREDIT_PERC", value=0.0)
        flag_document_3 = st.number_input("FLAG_DOCUMENT_3", value=0)
        emergency_state_mode_no = st.number_input("EMERGENCYSTATE_MODE_No", value=0)
        walls_material_mode_panel = st.number_input("WALLSMATERIAL_MODE_Panel", value=0)

        client_data = {
            "SK_ID_CURR": -1,  # Valeur par défaut pour un nouveau client
            "DAYS_EMPLOYED": days_employed,
            "DAYS_BIRTH": days_birth,
            "EXT_SOURCE_3": ext_source_3,
            "DAYS_ID_PUBLISH": days_id_publish,
            "CODE_GENDER": code_gender,
            "FLAG_OWN_CAR": flag_own_car,
            "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_1": ext_source_1,
            "NAME_EDUCATION_TYPE_Highereducation": name_education_type_highereducation,
            "NAME_CONTRACT_TYPE_Cashloans": name_contract_type_cashloans,
            "HOUR_APPR_PROCESS_START": hour_appr_process_start,
            "NAME_FAMILY_STATUS_Married": name_family_status_married,
            "FLAG_PHONE": flag_phone,
            "AMT_INCOME_TOTAL": amt_income_total,
            "AMT_CREDIT": amt_credit,
            "DAYS_REGISTRATION": days_registration,
            "INCOME_CREDIT_PERC": income_credit_perc,
            "FLAG_DOCUMENT_3": flag_document_3,
            "EMERGENCYSTATE_MODE_No": emergency_state_mode_no,
            "WALLSMATERIAL_MODE_Panel": walls_material_mode_panel
        }

        if st.button("Envoyer pour Prédiction", key="new_client_predict_button"):
            prediction, decision = get_credit_score(client_data)
            if prediction is not None:
                st.write(f"Prédiction de défaut du client : {prediction}")
                st.write(f"Décision : {decision}")
                
                proba = get_proba(client_data)
                if proba is not None:
                    st.write(f"Probabilité de défaut du client : {proba:.2f}")
                    jauge_score(proba)

# Onglet : Visualisation des Clients
elif tabs == "Visualisation des Clients":
    st.subheader("Visualisation des Caractéristiques des Clients")

    # Option pour mettre en évidence le client sélectionné dans le nuage de points
    selected_client_id = st.selectbox("Sélectionner un client pour le mettre en évidence :", data['SK_ID_CURR'].unique())
    scatter_plot_clients(data, selected_client_id)

# Onglet : Analyse Bi-Variée et Distribution
elif tabs == "Analyse Bi-Variée et Distribution":
    st.subheader("Analyse Bi-Variée et Distribution")

    # Analyse Bi-Variée
    st.subheader("Analyse Bi-Variée")
    feature_x_bivar = st.selectbox("Choisir la caractéristique pour l'axe X :", data.columns)
    feature_y_bivar = st.selectbox("Choisir la caractéristique pour l'axe Y :", data.columns)
    fig, ax = plt.subplots()
    sns.scatterplot(x=data[feature_x_bivar], y=data[feature_y_bivar], ax=ax)
    st.pyplot(fig)

    # Distribution des Variables
    st.subheader("Distribution des Variables")
    feature_dist = st.selectbox("Choisir la variable à visualiser :", data.columns)
    fig_dist, ax_dist = plt.subplots()
    sns.histplot(data[feature_dist], kde=True, ax=ax_dist)
    st.pyplot(fig_dist)

# Onglet : Accueil
else:
    st.write("Bienvenue sur le Dashboard Score de Crédit.")
    st.write("Utilisez les onglets à gauche pour naviguer et explorer les fonctionnalités.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Réalisé avec Streamlit • par Imane EL HABACHI")


