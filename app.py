import streamlit as st
import streamlit.components.v1 as components # Indispensable pour afficher le SHAP textuel
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
import altair as alt 
import shap # On croise les doigts pour la version du serveur !

# --- 0. CONFIGURATION ---
st.set_page_config(
    page_title="Trustpilot Sentiment IA",
    page_icon="⭐",
    layout="wide"
)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. CHARGEMENT DES RESSOURCES (CACHE) ---
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

download_nltk_resources()

@st.cache_resource
def load_model_assets():
    try:
        # On utilise joblib car tes fichiers s'appellent .pkl mais ont été sauvés avec joblib dans le code précédent
        model = joblib.load('trustpilot_lgbm_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model_assets()

# --- 2. FONCTIONS SPÉCIFIQUES SHAP (Ton code adapté) ---

# Fonction wrapper pour SHAP : prend du texte brut -> sort des probas
def predict_proba_text(texts):
    # vectorizer et model doivent être accessibles ici
    X = vectorizer.transform(texts)
    return model.predict_proba(X)

# On met en cache l'explainer car il est long à initialiser
@st.cache_resource
def build_shap_explainer(_model, _vectorizer):
    # Création du masker basé sur le tokenizer du vectorizer
    # Si build_tokenizer échoue, on utilise un regex simple r"\W+"
    try:
        tokenizer = _vectorizer.build_tokenizer()
        masker = shap.maskers.Text(tokenizer=tokenizer)
    except:
        masker = shap.maskers.Text(tokenizer=r"\W+")
        
    explainer = shap.Explainer(
        predict_proba_text,
        masker,
        output_names=["Négatif", "Neutre", "Positif"]
    )
    return explainer

# --- 3. PIPELINE DE NETTOYAGE (Pour l'affichage propre, mais SHAP utilise le texte brut ici) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def processing_pipeline(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", '', text) # Nettoyage simple pour l'affichage
    return text

# --- 4. INTERFACE UTILISATEUR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
    st.header("🔍 Infos du Modèle")
    st.info("Modèle : LightGBM")
    st.write("Vectorisation : TF-IDF")
    st.metric(label="Précision", value="71.8%")
    st.caption("Projet DataScientest")

st.title("🛍️ Analyse de Sentiment & SHAP")

# --- 5. ZONE DE PRÉDICTION ---
if model is None:
    st.error("⚠️ Fichiers modèles introuvables.")
else:
    user_input = st.text_area("Entrez un avis client :", "This product is amazing and easy to use but the delivery was slow.")

    if st.button("Analyser", type="primary"):
        with st.spinner('Calcul de la prédiction et des valeurs SHAP...'):
            
            # A. Prédiction Classique
            # Note: Pour SHAP textuel, on passe souvent le texte BRUT (ou légèrement nettoyé)
            # car le masker s'occupe du découpage.
            vec_input = vectorizer.transform([user_input])
            pred_class = model.predict(vec_input.toarray())[0]
            pred_proba = model.predict_proba(vec_input.toarray())[0]
            
            labels = {0: ("Négatif 😞", "red"), 1: ("Neutre 😐", "orange"), 2: ("Positif 😃", "green")}
            label_text, color = labels[pred_class]
            
            # Affichage Verdict
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"### Verdict : :{color}[{label_text}]")
                st.metric("Confiance", f"{pred_proba[pred_class]:.1%}")

            # B. SHAP TEXT PLOT (Ton Code Intégré)
            st.subheader("🧠 Analyse d'impact par mot (SHAP)")
            st.write(f"Voici comment chaque mot contribue à la note **{label_text}** :")
            
            try:
                # 1. Récupération de l'explainer
                explainer = build_shap_explainer(model, vectorizer)
                
                # 2. Calcul des valeurs SHAP pour la phrase
                shap_values = explainer([user_input])
                
                # 3. Affichage HTML
                # shap.plots.text génère du HTML. On doit l'extraire pour Streamlit.
                # On affiche l'explication pour la classe prédite (pred_class)
                html_shap = shap.plots.text(shap_values[0, :, pred_class], display=False)
                
                # Utilisation du composant Streamlit pour rendre le HTML
                components.html(html_shap, height=400, scrolling=True)
                
                st.caption("🔴 Rouge : Pousse vers ce sentiment | 🔵 Bleu : Éloigne de ce sentiment")

            except Exception as e:
                st.error(f"Erreur SHAP : {e}")
                st.info("Astuce : Si ça plante, vérifiez les versions de bibliothèques.")

    # --- Onglet Données (Optionnel, simplifié) ---
    with st.expander("Voir les statistiques du dataset"):
        st.write("Dataset : Amazon Reviews Electronics (1.2M avis)")
        st.bar_chart({"Négatif": 190000, "Neutre": 190000, "Positif": 190000})
