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

# --- 0. CONFIGURATION & NETTOYAGE CONSOLE ---
st.set_page_config(
    page_title="Trustpilot Sentiment IA",
    page_icon="⭐",
    layout="centered"
)
# Ignore les warnings de noms de colonnes LightGBM
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
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

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

# --- 2. PIPELINE DE NETTOYAGE (Identique notebook) ---
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

# --- 3. SIDEBAR (INFOS PROJET) ---
with st.sidebar:
    st.header("🔍 Infos du Modèle")
    st.info("Modèle : LightGBM + TF-IDF")
    st.write("Entraîné sur ~130k avis.")
    st.metric(label="Précision (Accuracy)", value="89.4%") # Mets ta vraie valeur ici si tu l'as
    st.markdown("---")
    st.caption("Projet École - Analyse de Sentiment")

# --- 4. INTERFACE PRINCIPALE ---
st.title("🛍️ Analyse d'Avis Trustpilot")
st.markdown("""
Cette IA analyse le texte d'un commentaire pour prédire l'expérience client :
**Négative** (1-2⭐), **Neutre** (3⭐) ou **Positive** (4-5⭐).
""")

if model is None:
    st.error("⚠️ Erreur : Fichiers .pkl introuvables. Vérifiez le dossier.")
else:
    # --- GESTION DES BOUTONS EXEMPLES ---
    # On initialise la variable de session si elle n'existe pas
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""

    def set_text(text):
        st.session_state.text_input = text

    st.markdown("### 📝 Testez avec vos propres phrases ou utilisez un exemple :")

    # Les 3 colonnes pour les boutons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("😡 Négatif", on_click=set_text, args=["Horrible service, I waited 2 weeks and the package is broken. Never again!"], use_container_width=True)
    with col2:
        st.button("😐 Neutre", on_click=set_text, args=["The product is okay but shipping was a bit slow. Not bad, not great."], use_container_width=True)
    with col3:
        st.button("😍 Positif", on_click=set_text, args=["Absolutely amazing! Best purchase of the year, highly recommended."], use_container_width=True)

    # Zone de texte (liée à la session state pour réagir aux boutons)
    user_input = st.text_area("Votre commentaire :", value=st.session_state.text_input, height=100)

    # --- BOUTON DE PRÉDICTION ---
    if st.button("Lancer l'analyse", type="primary"):
        if user_input.strip():
            with st.spinner('Analyse en cours...'):
                
                # 1. Nettoyage
                clean_text = processing_pipeline(user_input)
                
                # 2. Vectorisation
                vec_input = vectorizer.transform([clean_text])
                
                # 3. Prédiction (avec suppression des warnings features)
                # On utilise toarray() pour éviter le warning LightGBM
                pred_class = model.predict(vec_input.toarray())[0]
                pred_proba = model.predict_proba(vec_input.toarray())[0]
                
                # 4. Mapping Résultats
                labels = {
                    0: ("Négatif 😞", "red"),
                    1: ("Neutre 😐", "orange"),
                    2: ("Positif 😃", "green")
                }
                label_text, color = labels[pred_class]
                confidence = pred_proba[pred_class]

                # 5. Affichage
                st.divider()
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.markdown("### Verdict :")
                    st.markdown(f":{color}[**{label_text}**]")
                    st.metric("Confiance", f"{confidence:.1%}")
                
                with c2:
                    st.markdown("#### Probabilités")
                    chart_data = pd.DataFrame(
                        pred_proba.reshape(1, 3), 
                        columns=["Négatif", "Neutre", "Positif"]
                    )
                    # Remplacer st.bar_chart par ceci pour avoir les couleurs :
                    import altair as alt

                   # On prépare les données proprement
                    df_chart = pd.DataFrame({
                        "Sentiment": ["Négatif", "Neutre", "Positif"],
                        "Probabilité": pred_proba,
                        "Couleur": ["#FF4B4B", "#FFA500", "#008000"]  # Rouge, Orange, Vert
                    })

                    # On crée le graph
                    c = alt.Chart(df_chart).mark_bar().encode(
                        x=alt.X('Sentiment', sort=None),
                        y='Probabilité',
                        color=alt.Color('Sentiment', scale=alt.Scale(domain=["Négatif", "Neutre", "Positif"], range=["#FF4B4B", "#FFA500", "#008000"]), legend=None),
                        tooltip=['Sentiment', 'Probabilité']
                    )
                    
                    st.altair_chart(c, use_container_width=True)

                with st.expander("👀 Voir le texte nettoyé par l'IA"):
                    st.code(clean_text)
        else:
            st.warning("Veuillez entrer du texte.")

st.markdown("---")

st.divider()
st.header("📂 Analyse de masse (Fichier CSV)")

uploaded_file = st.file_uploader("Déposez un fichier CSV contenant une colonne 'text'", type=["csv"])

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    
    # Vérification qu'il y a du texte
    if 'text' in df_upload.columns:
        if st.button("Lancer l'analyse du fichier"):
            with st.spinner("Analyse de tous les avis..."):
                # On applique le nettoyage et la prédiction
                # Attention : Pour aller vite, on ne fait pas de boucle, on vectorise tout d'un coup
                # (Nécessite d'adapter légèrement ta pipeline pour accepter une Série pandas, 
                # ou alors faire une boucle simple apply)
                
                df_upload['clean_text'] = df_upload['text'].apply(processing_pipeline)
                vec_bulk = vectorizer.transform(df_upload['clean_text'])
                predictions = model.predict(vec_bulk) # Donne 0, 1, 2
                
                # Mapping pour rendre ça lisible
                map_dict = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                df_upload['Prediction'] = [map_dict[p] for p in predictions]
                
                st.success("Analyse terminée !")
                st.dataframe(df_upload[['text', 'Prediction']].head())
                
                # Bouton de téléchargement
                csv = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button("Télécharger les résultats", csv, "resultats_trustpilot.csv", "text/csv")
    else:
        st.error("Le fichier CSV doit contenir une colonne nommée 'text'.")

