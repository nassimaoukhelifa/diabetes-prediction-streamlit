# app_diabete_portfolio.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# ===============================
# CONFIGURATION GÉNÉRALE
# ===============================
st.set_page_config(
    page_title="Prédiction du Diabète – Pima Indians",
    page_icon="🩺",
    layout="wide"
)

# ===============================
#  TITRE ET INTRO
# ===============================
st.title(" Application de Prédiction du Diabète")
st.markdown("""
Bienvenue dans cette application de **Machine Learning** basée sur le dataset *Pima Indians Diabetes*.
Entrez les données médicales d’un patient pour **estimer le risque de diabète**.
""")
st.markdown("---")

# ===============================
# 📂 CHARGEMENT DU DATASET
# ===============================

# Chemins possibles
possible_paths = [
    "data/diabetes.csv",
    "diabetes.csv",
    "./diabetes.csv",
    "/app/data/diabetes.csv",   # pour Streamlit Cloud
]

csv_path = None
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    st.error("⚠️ Le fichier `diabetes.csv` est introuvable. "
             "Assure-toi qu'il se trouve dans un dossier `data/` ou dans le même répertoire que ce script.")
    st.stop()

df = pd.read_csv(csv_path, sep=";") if ";" in open(csv_path).readline() else pd.read_csv(csv_path)

st.success(f"✅ Données chargées depuis : `{csv_path}`")

# ===============================
# 🔢 PRÉPARATION DES DONNÉES
# ===============================

# --- Nettoyage automatique des colonnes numériques ---
for col in df.columns:
    if df[col].dtype == 'object':  # si le contenu est du texte (souvent à cause des virgules)
        df[col] = df[col].str.replace(',', '.', regex=False).astype(float)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_scaled, y)

# ===============================
# 🧮 ENTRÉES UTILISATEUR
# ===============================
st.sidebar.header(" Paramètres du patient")

pregnancies = st.sidebar.number_input("Grossesses", 0, 20, 2)
glucose = st.sidebar.slider("Glucose (mg/dL)", 50, 200, 100)
blood_pressure = st.sidebar.slider("Pression artérielle (mm Hg)", 40, 120, 70)
skin_thickness = st.sidebar.slider("Épaisseur du pli cutané (mm)", 10, 99, 20)
insulin = st.sidebar.slider("Insuline (mu U/ml)", 0, 900, 80)
bmi = st.sidebar.slider("BMI (Indice de Masse Corporelle)", 15.0, 67.0, 30.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Âge (années)", 18, 100, 35)

user_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

st.subheader("📋 Données saisies")
st.dataframe(user_data, use_container_width=True)

# ===============================
# 🔮 PRÉDICTION
# ===============================
user_scaled = scaler.transform(user_data)

if st.button(" Lancer la prédiction"):
    pred = model.predict(user_scaled)[0]
    proba = model.predict_proba(user_scaled)[0][1] * 100

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ Risque élevé de **diabète** – probabilité estimée : **{proba:.1f}%**")
    else:
        st.success(f"✅ Aucun signe de diabète détecté – probabilité estimée : **{proba:.1f}%**")

    # ===============================
    # 📊 IMPORTANCE DES VARIABLES
    # ===============================
    st.subheader(" Importance des variables selon le modèle")

    importances = pd.DataFrame({
        "Variable": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Importance", y="Variable", data=importances, palette="YlGn")
    plt.title("Importance des variables – Random Forest")
    st.pyplot(fig)

# ===============================
# 📈 VISUALISATION DU DATASET
# ===============================
with st.expander(" Voir un aperçu statistique du dataset"):
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Distribution du Glucose")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Glucose"], bins=20, kde=True, color="#4C8BF5", ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.write("### Distribution du BMI")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["BMI"], bins=20, kde=True, color="#37C871", ax=ax2)
        st.pyplot(fig2)

st.caption("🧬 Application développée avec Streamlit, Pandas et Scikit-learn – © 2025")

