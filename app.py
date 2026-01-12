import streamlit as st
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
        model = joblib.load('trustpilot_lgbm_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model_assets()

# --- 2. PIPELINE DE NETTOYAGE ---
stop_words = set(stopwords.words('english'))
stop_words.update([",", ".", "``", "@", "*", "(", ")", "[","]", "...", "-", "_", ">", "<", ":", "/", "//", "///", "=", "--", "©", "~", ";", "\\", "\\\\", '"', "'","''", '""' "'m", "'ve", "n't","!","?", "'re", "rd", "'s", "%"])
lemmatizer = WordNetLemmatizer()

def processing_pipeline(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"\.+", '', text)
    text = re.sub(r"/", ' ', text)
    text = re.sub(r"[0-9]+", '', text)
    try:
        tokens = word_tokenize(text, language='english')
    except:
        tokens = text.split()
    
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)
    return " ".join(cleaned_tokens)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
    st.header("🔍 Infos du Modèle")
    st.info("Modèle : LightGBM")
    st.write("Vectorisation : TF-IDF")
    st.metric(label="Précision (Accuracy)", value="71.8%", delta="+vs Baseline")
    st.markdown("---")
    st.caption("Projet DataScientest\nLakdar & Aurore")

# --- 4. TITRE PRINCIPAL ---
st.title("🛍️ Analyse de Sentiment & Expérience Client")
st.markdown("Application de démonstration pour la prédiction de satisfaction à partir d'avis textuels.")

# --- 5. CRÉATION DES ONGLETS ---
tab_demo, tab_data, tab_model = st.tabs(["🚀 Démo Live", "📊 Jeu de Données", "🤖 Performance Modèle"])

# ==============================================================================
# ONGLET 1 : DÉMO LIVE
# ==============================================================================
with tab_demo:
    if model is None:
        st.error("⚠️ Erreur : Fichiers .pkl introuvables.")
    else:
        if "text_input" not in st.session_state:
            st.session_state.text_input = ""

        def set_text(text):
            st.session_state.text_input = text

        st.subheader("Testez l'IA en temps réel")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("😡 Négatif", on_click=set_text, args=["Horrible service, I waited 2 weeks and the package is broken. Never again!"], use_container_width=True)
        with col2:
            st.button("😐 Neutre", on_click=set_text, args=["The product is okay but shipping was a bit slow. Not bad, not great."], use_container_width=True)
        with col3:
            st.button("😍 Positif", on_click=set_text, args=["Absolutely amazing! Best purchase of the year, highly recommended."], use_container_width=True)

        user_input = st.text_area("Votre commentaire :", value=st.session_state.text_input, height=100)

        if st.button("Lancer l'analyse", type="primary"):
            if user_input.strip():
                with st.spinner('Analyse en cours...'):
                    clean_text = processing_pipeline(user_input)
                    vec_input = vectorizer.transform([clean_text])
                    pred_class = model.predict(vec_input.toarray())[0]
                    pred_proba = model.predict_proba(vec_input.toarray())[0]
                    
                    labels = {0: ("Négatif 😞", "red"), 1: ("Neutre 😐", "orange"), 2: ("Positif 😃", "green")}
                    label_text, color = labels[pred_class]

                    st.divider()
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown("### Verdict :")
                        st.markdown(f":{color}[**{label_text}**]")
                        st.metric("Confiance", f"{pred_proba[pred_class]:.1%}")
                    with c2:
                        df_chart = pd.DataFrame({"Sentiment": ["Négatif", "Neutre", "Positif"], "Probabilité": pred_proba})
                        chart = alt.Chart(df_chart).mark_bar().encode(
                            x=alt.X('Sentiment', sort=None),
                            y='Probabilité',
                            color=alt.Color('Sentiment', scale=alt.Scale(domain=["Négatif", "Neutre", "Positif"], range=["#6D6D6D", "#FFB7B2", "#FF69B4"]), legend=None)
                        )
                        st.altair_chart(chart, use_container_width=True)

# ==============================================================================
# ONGLET 2 : JEU DE DONNÉES (Slide 6 & Preprocessing)
# ==============================================================================
with tab_data:
    st.header("📚 Le Jeu de Données : Amazon Electronics")
    
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.markdown("### Source & Pourquoi ?")
        st.info("**Amazon Reviews Dataset**")
        st.write("- Structure identique à Trustpilot (Texte + Note)")
        st.write("- Focus **Electronics** : Vocabulaire riche et technique")
        st.write("- Période : 2010 - 2018")
    
    with col_d2:
        st.markdown("### Volumétrie & Nettoyage")
        metrics_df = pd.DataFrame({"Métrique": ["Avis bruts", "Avis après filtrage", "Langue"], "Valeur": ["~1.2 Millions", "572 950", "Anglais"]})
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        st.error("❌ **Variable 'Prix'** : Supprimée (trop de valeurs manquantes).")
        st.warning("⚠️ **Variable 'Votes'** : Imputation à 0 (Pas de vote = Inutile).")
        st.info("🖼️ **Variable 'Image'** : Transformée en Booléen (Présence/Absence).")

    st.divider()
    st.subheader("📋 Aperçu des données brutes (Exemple)")
    example_data = {
        "overall": [5, 1, 3, 5, 2],
        "summary": ["Amazing sound", "Waste of money", "Average", "Great service", "Disappointed"],
        "reviewText": ["This headphone is amazing! The bass is deep.", "Terrible quality, stopped working.", "It's okay for the price.", "Works perfectly, fast delivery.", "Poor screen resolution."],
        "brand": ["Bose", "Generic", "Sony", "Samsung", "LG"]
    }
    st.dataframe(pd.DataFrame(example_data), use_container_width=True)

    st.markdown("---")
    st.subheader("Distribution des classes (Équilibrée par Undersampling)")
    chart_balance = alt.Chart(pd.DataFrame({"Sentiment": ["Négatif", "Neutre", "Positif"], "Nombre": [190983, 190983, 190983]})).mark_bar().encode(
        x=alt.X('Sentiment', sort=None), y='Nombre',
        color=alt.Color('Sentiment', scale=alt.Scale(range=["#6D6D6D", "#FFB7B2", "#FF69B4"]), legend=None)
    )
    st.altair_chart(chart_balance, use_container_width=True)

# ==============================================================================
# ONGLET 3 : PERFORMANCE MODÈLE
# ==============================================================================
with tab_model:
    st.header("⚙️ Performance LightGBM")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy Globale", "71.8%")
    m2.metric("F1-Score", "0.72")
    m3.metric("Vocabulaire", "5 000 mots")

    st.subheader("Matrice de Confusion")
    confusion_data = pd.DataFrame(
        [[8303, 2155, 558], [2303, 6734, 1979], [551, 1765, 8700]],
        columns=["Prédit Négatif", "Prédit Neutre", "Prédit Positif"],
        index=["Réel Négatif", "Réel Neutre", "Réel Positif"]
    )
    
    # Sécurité pour éviter l'erreur matplotlib
    try:
        st.dataframe(confusion_data.style.background_gradient(cmap="Blues"), use_container_width=True)
    except:
        st.dataframe(confusion_data, use_container_width=True)
    
    st.info("**Analyse :** Excellente détection des extrêmes. La classe **Neutre** reste la plus complexe à isoler sémantiquement.")
