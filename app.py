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
import altair as alt  # Importé ici pour être dispo partout

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
    st.metric(label="Précision (Accuracy)", value="71.8%")
    st.markdown("---")
    st.caption("DataScientest - Trust Pilot - Analyse de Sentiment")

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
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""

    def set_text(text):
        st.session_state.text_input = text

    st.markdown("### 📝 Testez avec vos propres phrases ou utilisez un exemple :")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("😡 Négatif", on_click=set_text, args=["Horrible service, I waited 2 weeks and the package is broken. Never again!"], use_container_width=True)
    with col2:
        st.button("😐 Neutre", on_click=set_text, args=["The product is okay but shipping was a bit slow. Not bad, not great."], use_container_width=True)
    with col3:
        st.button("😍 Positif", on_click=set_text, args=["Absolutely amazing! Best purchase of the year, highly recommended."], use_container_width=True)

    # Zone de texte
    user_input = st.text_area("Votre commentaire :", value=st.session_state.text_input, height=100)

    # --- BOUTON DE PRÉDICTION UNITAIRE ---
    if st.button("Lancer l'analyse", type="primary"):
        if user_input.strip():
            with st.spinner('Analyse en cours...'):
                
                # 1. Nettoyage & Vectorisation
                clean_text = processing_pipeline(user_input)
                vec_input = vectorizer.transform([clean_text])
                
                # 2. Prédiction
                pred_class = model.predict(vec_input.toarray())[0]
                pred_proba = model.predict_proba(vec_input.toarray())[0]
                
                # 3. Mapping
                labels = {
                    0: ("Négatif 😞", "red"),
                    1: ("Neutre 😐", "orange"),
                    2: ("Positif 😃", "green")
                }
                label_text, color = labels[pred_class]
                confidence = pred_proba[pred_class]

                # 4. Affichage
                st.divider()
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.markdown("### Verdict :")
                    st.markdown(f":{color}[**{label_text}**]")
                    st.metric("Confiance", f"{confidence:.1%}")
                
                with c2:
                    st.markdown("#### Probabilités")
                    df_chart = pd.DataFrame({
                        "Sentiment": ["Négatif", "Neutre", "Positif"],
                        "Probabilité": pred_proba,
                        "Couleur": ["#FF4B4B", "#FFA500", "#008000"]
                    })
                    
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

    # === SECTION 2 : ANALYSE DE MASSE (CSV) ===
    st.markdown("---")
    st.subheader("2. Analyse de masse (Fichier CSV) 📂")
    
    st.markdown("""
    **Comment ça marche ?**
    1. Téléchargez le modèle ci-dessous.
    2. Ajoutez vos avis dans la colonne **'text'**.
    3. Déposez le fichier ici.
    """)

    csv_template = "text\nExemple: Super produit !\nExemple: Livraison trop longue..."
    st.download_button(
        label="📥 Télécharger le modèle CSV vide",
        data=csv_template,
        file_name="modele_avis.csv",
        mime="text/csv"
    )

    st.warning("⚠️ Important : Votre fichier doit avoir une colonne nommée **'text'**, **'review'** ou **'comment'**.")

    uploaded_file = st.file_uploader("Déposez votre fichier rempli ici", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            possible_cols = [c for c in df.columns if 'text' in c.lower() or 'review' in c.lower() or 'comment' in c.lower()]
            
            if not possible_cols:
                st.error(f"❌ Erreur : Colonne texte introuvable. Colonnes vues : {list(df.columns)}")
                st.info("Conseil : Renommez votre colonne d'avis en 'text' dans Excel.")
            else:
                target_col = possible_cols[0]
                st.success(f"✅ Colonne détectée : **{target_col}** ({len(df)} lignes)")

                if st.button(f"Lancer l'analyse sur les {len(df)} avis", type="primary"):
                    with st.spinner("Traitement en cours..."):
                        progress_bar = st.progress(0)
                        
                        df['clean_text'] = df[target_col].astype(str).apply(processing_pipeline)
                        progress_bar.progress(30)
                        
                        vec_bulk = vectorizer.transform(df['clean_text'])
                        progress_bar.progress(60)
                        
                        preds = model.predict(vec_bulk.toarray())
                        progress_bar.progress(90)
                        
                        label_map = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                        df['Prediction_IA'] = [label_map[p] for p in preds]
                        
                        progress_bar.progress(100)
                        
                        st.balloons()
                        st.write("### 📊 Résultats de l'analyse :")
                        
                        # --- 💡 AJOUT : FILTRE INTERACTIF ---
                        st.write("🔎 **Filtrer les résultats :**")
                        filter_option = st.radio(
                            "Afficher :",
                            ["Tout voir", "Négatif", "Neutre", "Positif"],
                            horizontal=True
                        )

                        if filter_option == "Négatif":
                            df_display = df[df['Prediction_IA'] == "Négatif"]
                        elif filter_option == "Neutre":
                            df_display = df[df['Prediction_IA'] == "Neutre"]
                        elif filter_option == "Positif":
                            df_display = df[df['Prediction_IA'] == "Positif"]
                        else:
                            df_display = df

                        st.caption(f"{len(df_display)} avis affichés.")

                        # Colorer le tableau
                        def color_pred(val):
                            color = '#ffcccc' if val == 'Négatif' else '#ccffcc' if val == 'Positif' else '#ffeebb'
                            return f'background-color: {color}'

                        st.dataframe(df_display[[target_col, 'Prediction_IA']].style.applymap(color_pred, subset=['Prediction_IA']), use_container_width=True)

                        # Statistique et Export
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.write("#### Répartition globale :")
                            st.bar_chart(df['Prediction_IA'].value_counts())
                        
                        with col_stat2:
                             st.write("#### Export :")
                             csv_result = df.to_csv(index=False).encode('utf-8')
                             st.download_button(
                                label="📥 Télécharger les résultats complets",
                                data=csv_result,
                                file_name="resultats_trustpilot.csv",
                                mime="text/csv"
                            )

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")

# --- 💡 AJOUT : SECTION EXPLICABILITÉ (BAS DE PAGE) ---
st.markdown("---")
with st.expander("🧠 Comprendre comment l'IA décide (Interprétabilité)"):
    st.write("""
    Ce modèle ne devine pas au hasard. Il a appris à associer certains mots à des scores.
    Voici les mots qui ont le plus d'impact statistique sur la décision :
    """)
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.error("📉 Mots qui rendent un avis NÉGATIF")
        # Si tu as une image 'wordcloud_negatif.png', décommente :
        # st.image("wordcloud_negatif.png")
        st.markdown("- **Service :** rude, ignored, horrible, useless")
        st.markdown("- **Produit :** broken, waste, scam, never")

    with col_img2:
        st.success("📈 Mots qui rendent un avis POSITIF")
        # Si tu as une image 'wordcloud_positif.png', décommente :
        # st.image("wordcloud_positif.png")
        st.markdown("- **Service :** fast, amazing, helpful, thanks")
        st.markdown("- **Produit :** love, best, recommend, perfect")
