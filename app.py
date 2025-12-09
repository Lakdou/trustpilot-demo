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

# --- 0. CONFIGURATION & NETTOYAGE CONSOLE ---
st.set_page_config(
    page_title="Trustpilot Sentiment IA",
    page_icon="⭐",
    layout="wide" # Layout "wide" pour avoir plus de place pour les onglets
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

# --- 3. SIDEBAR (INFOS PROJET) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150) # Petit clin d'oeil à la source
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
# ONGLET 1 : DÉMO LIVE (Ton code existant)
# ==============================================================================
with tab_demo:
    if model is None:
        st.error("⚠️ Erreur : Fichiers .pkl introuvables. Vérifiez le dossier.")
    else:
        # --- BOUTONS EXEMPLES ---
        if "text_input" not in st.session_state:
            st.session_state.text_input = ""

        def set_text(text):
            st.session_state.text_input = text

        st.subheader("Testez l'IA en temps réel")
        st.markdown("Choisissez un exemple ou écrivez votre propre avis :")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("😡 Négatif", on_click=set_text, args=["Horrible service, I waited 2 weeks and the package is broken. Never again!"], use_container_width=True)
        with col2:
            st.button("😐 Neutre", on_click=set_text, args=["The product is okay but shipping was a bit slow. Not bad, not great."], use_container_width=True)
        with col3:
            st.button("😍 Positif", on_click=set_text, args=["Absolutely amazing! Best purchase of the year, highly recommended."], use_container_width=True)

        user_input = st.text_area("Votre commentaire :", value=st.session_state.text_input, height=100)

        # --- PREDICTION ---
        if st.button("Lancer l'analyse", type="primary"):
            if user_input.strip():
                with st.spinner('Analyse en cours...'):
                    clean_text = processing_pipeline(user_input)
                    vec_input = vectorizer.transform([clean_text])
                    
                    pred_class = model.predict(vec_input.toarray())[0]
                    pred_proba = model.predict_proba(vec_input.toarray())[0]
                    
                    labels = {0: ("Négatif 😞", "red"), 1: ("Neutre 😐", "orange"), 2: ("Positif 😃", "green")}
                    label_text, color = labels[pred_class]
                    confidence = pred_proba[pred_class]

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
                            "Couleur": ["#6D6D6D", "#FFB7B2", "#FF69B4"] 
                        })
                        c = alt.Chart(df_chart).mark_bar().encode(
                            x=alt.X('Sentiment', sort=None),
                            y='Probabilité',
                            color=alt.Color('Sentiment', scale=alt.Scale(domain=["Négatif", "Neutre", "Positif"], range=["#6D6D6D", "#FFB7B2", "#FF69B4"]), legend=None),
                            tooltip=['Sentiment', 'Probabilité']
                        )
                        st.altair_chart(c, use_container_width=True)

                    with st.expander("👀 Voir le texte nettoyé (Lemmatisation)"):
                        st.code(clean_text)
            else:
                st.warning("Veuillez entrer du texte.")

        # --- ANALYSE CSV ---
        st.markdown("---")
        st.subheader("📂 Analyse de masse (Fichier CSV)")
        
        csv_template = "text\nExemple: Super produit !\nExemple: Livraison trop longue..."
        st.download_button("📥 Télécharger le modèle CSV", data=csv_template, file_name="modele_avis.csv", mime="text/csv")

        uploaded_file = st.file_uploader("Déposez votre fichier ici", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                possible_cols = [c for c in df.columns if 'text' in c.lower() or 'review' in c.lower() or 'comment' in c.lower()]
                
                if not possible_cols:
                    st.error("Colonne texte introuvable. Vérifiez votre fichier.")
                else:
                    target_col = possible_cols[0]
                    if st.button(f"Analyser {len(df)} avis"):
                        with st.spinner("Traitement..."):
                            df['clean_text'] = df[target_col].astype(str).apply(processing_pipeline)
                            vec_bulk = vectorizer.transform(df['clean_text'])
                            preds = model.predict(vec_bulk.toarray())
                            label_map = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                            df['Prediction_IA'] = [label_map[p] for p in preds]
                            
                            st.write("### Résultats :")
                            filter_option = st.radio("Filtrer :", ["Tout voir", "Négatif", "Neutre", "Positif"], horizontal=True)
                            
                            if filter_option != "Tout voir":
                                df_display = df[df['Prediction_IA'] == filter_option]
                            else:
                                df_display = df
                                
                            def color_pred(val):
                                color = '#ffcccc' if val == 'Négatif' else '#ccffcc' if val == 'Positif' else '#ffeebb'
                                return f'background-color: {color}'

                            st.dataframe(df_display[[target_col, 'Prediction_IA']].style.applymap(color_pred, subset=['Prediction_IA']), use_container_width=True)
                            
                            col_stat1, col_stat2 = st.columns(2)
                            with col_stat1:
                                st.bar_chart(df['Prediction_IA'].value_counts())
                            with col_stat2:
                                st.download_button("📥 Télécharger résultats", df.to_csv(index=False).encode('utf-8'), "resultats.csv", "text/csv")

            except Exception as e:
                st.error(f"Erreur : {e}")

# ==============================================================================
# ONGLET 2 : DONNÉES D'ENTRAÎNEMENT (Infos du rapport)
# ==============================================================================
with tab_data:
    st.header("📚 Le Jeu de Données d'Entraînement")
    
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        st.markdown("### Source")
        st.info("**Amazon Reviews Dataset**")
        st.markdown("Catégorie : **Electronics**")
        st.markdown("Période : **2010 - 2018**")
        st.markdown("Langue : **Anglais** uniquement")
    
    with col_d2:
        st.markdown("### Volumétrie & Nettoyage")
        st.write("Pour garantir la qualité du modèle, nous avons appliqué un preprocessing strict :")
        
        # Données fictives basées sur ton rapport pour l'affichage
        metrics_df = pd.DataFrame({
            "Métrique": ["Avis bruts", "Avis après nettoyage", "Classes"],
            "Valeur": ["~1.2 Millions", "572 950", "3 (Équilibrées)"]
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        st.warning("⚠️ **Choix forts :** Suppression de la variable 'Prix' (trop de valeurs manquantes) et imputation des votes vides à 0.")

    st.markdown("---")
    st.subheader("Distribution des classes (Après rééquilibrage)")
    # On simule les données du rapport
    chart_data = pd.DataFrame({
        "Sentiment": ["Négatif", "Neutre", "Positif"],
        "Nombre d'avis": [190983, 190983, 190983] 
    })
    st.bar_chart(chart_data.set_index("Sentiment"))
    st.caption("Le dataset a été rééchantillonné (Undersampling) pour éviter que le modèle ne favorise la classe majoritaire (5 étoiles).")

# ==============================================================================
# ONGLET 3 : PERFORMANCES MODÈLE (Infos du rapport)
# ==============================================================================
with tab_model:
    st.header("⚙️ Performance du Modèle LightGBM")
    
    st.write("Le modèle **LightGBM** a été retenu face au Random Forest et aux réseaux de neurones (CNN/LSTM) pour son excellent ratio performance/coût.")

    # 1. Métriques Clés
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy Globale", "71.8%", delta="Meilleur modèle")
    with m2:
        st.metric("F1-Score (Macro)", "0.72")
    with m3:
        st.metric("Taille Vocabulaire", "5 000 mots")

    st.markdown("---")

    # 2. Matrice de Confusion (Reconstituée d'après ton rapport)
    col_conf1, col_conf2 = st.columns([2, 1])
    
    with col_conf1:
        st.subheader("Matrice de Confusion")
        st.write("Capacité du modèle à prédire la bonne classe :")
        
        # Données exactes de ton rapport page 30
        confusion_data = pd.DataFrame(
            [
                [8303, 2155, 558],
                [2303, 6734, 1979],
                [551, 1765, 8700]
            ],
            columns=["Prédit Négatif", "Prédit Neutre", "Prédit Positif"],
            index=["Réel Négatif", "Réel Neutre", "Réel Positif"]
        )
        st.dataframe(confusion_data.style.background_gradient(cmap="Blues"), use_container_width=True)
    
    with col_conf2:
        st.subheader("Analyse")
        st.info("""
        **Points forts :**
        - Excellente détection des avis **Négatifs** et **Positifs**.
        
        **Points faibles :**
        - La classe **Neutre** reste difficile à isoler (confusions fréquentes avec les classes voisines).
        """)

    # 3. Feature Importance (SHAP Global)
    st.markdown("---")
    st.subheader("🧠 Quels mots pèsent le plus ? (Global Feature Importance)")
    st.write("Voici les termes qui influencent le plus la décision de l'IA (basé sur SHAP Values) :")
    
    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        st.error("📉 Mots Négatifs")
        st.write("- **bad, poor, waste, return, money**")
        st.caption("Indiquent souvent un problème de qualité ou de remboursement.")
    with col_feat2:
        st.success("📈 Mots Positifs")
        st.write("- **great, love, good, easy, perfect**")
        st.caption("Indiquent une satisfaction émotionnelle forte.")
