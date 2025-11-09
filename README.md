# diabetes-prediction-streamlit
Application Streamlit pour la prédiction du diabète (dataset Pima Indians
#  Prédiction du Diabète – Application Streamlit

![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Status](https://img.shields.io/badge/Status-En%20ligne-success)

---

##  Objectif du projet

Cette application interactive permet de **prédire le risque de diabète** à partir du **dataset Pima Indians Diabetes** (Kaggle).  
Elle s’appuie sur un modèle de **Machine Learning (Random Forest)** entraîné sur 768 observations.

---

##  Fonctionnalités

- Entrée manuelle des paramètres médicaux :
  - Grossesses, Glucose, Pression artérielle, Insuline, BMI, Âge, etc.
- Prédiction instantanée du **risque de diabète**
- Affichage de la **probabilité estimée**
- Visualisation des **variables les plus importantes**
- Graphiques de distribution interactifs

---

##  Technologies utilisées

| Outil | Rôle |
|-------|------|
| **Python** | Langage principal |
| **Pandas / Numpy** | Gestion et manipulation des données |
| **Scikit-learn** | Entraînement du modèle de Machine Learning |
| **Matplotlib / Seaborn** | Visualisations |
| **Streamlit** | Création de l’application web |

---

##  Modèle utilisé

- **Type :** Random Forest Classifier  
- **Précision :** 0.75  
- **Variables clés :** Glucose, BMI, Age, DiabetesPedigreeFunction  
- **Objectif :** Identifier les patientes présentant un risque élevé de diabète

---

##  Lancer l’application localement

###  Cloner le dépôt
```bash
git clone https://github.com/<TON_UTILISATEUR>/diabetes-prediction-streamlit.git
cd diabetes-prediction-streamlit

