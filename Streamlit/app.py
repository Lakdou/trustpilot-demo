import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Trustpilot Sentiment IA",
    page_icon="⭐",
    layout="centered"
)

# --- CHARGEMENT DES RESSOURCES NLTK (CACHE) ---
# Nécessaire pour que ça marche sur n'importe quel ordi ou sur le Cloud
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

download_nltk_resources()

# --- CHARGEMENT DU MODÈLE ET VECTORISEUR ---
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load('trustpilot_lgbm_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model_assets()

# --- FONCTION DE NETTOYAGE (Identique à ton entraînement) ---
# On initialise les outils une seule fois
stop_words = set(stopwords.words('english'))
# Ta liste personnalisée de stopwords/ponctuation
stop_words.update([",", ".", "``", "@", "*", "(", ")", "[","]", "...", "-", "_", ">", "<", ":", "/", "//", "///", "=", "--", "©", "~", ";", "\\", "\\\\", '"', "'","''", '""' "'m", "'ve", "n't","!","?", "'re", "rd", "'s", "%"])
lemmatizer = WordNetLemmatizer()

def processing_pipeline(text):
    if not isinstance(text, str): return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Regex (Tes règles exactes)
    text = re.sub(r"\.+", '', text)      # Points multiples
    text = re.sub(r"/", ' ', text)       # Slashes
    text = re.sub(r"[0-9]+", '', text)   # Chiffres
    
    # 3. Tokenisation
    try:
        tokens = word_tokenize(text, language='english')
    except:
        # Fallback simple si punkt_tab plante
        tokens = text.split()
    
    # 4. Stopwords & Lemmatisation
    cleaned_tokens = []
    seen = set() # Pour éviter les doublons dans la même phrase si tu veux
    
    for token in tokens:
        if token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)
            
    # 5. Rejoin (TF-IDF a besoin d'une string, pas d'une liste)
    return " ".join(cleaned_tokens)

# --- INTERFACE UTILISATEUR ---

st.title("🛍️ Analyse d'Avis Trustpilot")
st.markdown("""
Cette IA analyse le texte d'un commentaire pour prédire si l'expérience client a été :
**Négative** (1-2⭐), **Neutre** (3⭐) ou **Positive** (4-5⭐).
""")

if model is None:
    st.error("⚠️ Erreur : Les fichiers `.pkl` sont introuvables. Vérifiez qu'ils sont bien dans le même dossier que app.py")
else:
    # Zone de saisie
    user_input = st.text_area("Copiez un avis client ici (en anglais) :", height=100, placeholder="Example: The delivery was very fast but the product quality is poor...")

    if st.button("Lancer l'analyse", type="primary"):
        if user_input.strip():
            with st.spinner('Nettoyage et analyse en cours...'):
                
                # 1. Prétraitement
                clean_text = processing_pipeline(user_input)
                
                # 2. Vectorisation
                vec_input = vectorizer.transform([clean_text])
                
                # 3. Prédiction
                pred_class = model.predict(vec_input)[0]
                pred_proba = model.predict_proba(vec_input)[0]
                
                # Mapping des classes (0, 1, 2) vers Labels
                # Rappel: Tu as fait y_train - 1, donc : 0=Neg, 1=Neu, 2=Pos
                labels = {
                    0: ("Négatif 😞", "red"),
                    1: ("Neutre 😐", "orange"),
                    2: ("Positif 😃", "green")
                }
                
                label_text, color = labels[pred_class]
                confidence = pred_proba[pred_class]

                # --- AFFICHAGE DES RÉSULTATS ---
                st.divider()
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"### Verdict :")
                    st.markdown(f":{color}[**{label_text}**]")
                    st.metric("Niveau de confiance", f"{confidence:.1%}")
                
                with col2:
                    st.markdown("#### Probabilités par classe")
                    # Création d'un petit dataframe pour le graph
                    chart_data = pd.DataFrame(
                        pred_proba.reshape(1, 3), 
                        columns=["Négatif", "Neutre", "Positif"]
                    )
                    st.bar_chart(chart_data.T)

                # Section Debug (Toujours sympa pour la démo)
                with st.expander("👀 Voir ce que l'IA a 'lu' (Texte nettoyé)"):
                    st.write(f"**Brut :** {user_input}")
                    st.write(f"**Nettoyé & Lemmatisé :** {clean_text}")

        else:
            st.warning("Veuillez entrer du texte pour analyser.")

# Footer
st.markdown("---")
st.caption("Projet École - Modèle LightGBM + TF-IDF")