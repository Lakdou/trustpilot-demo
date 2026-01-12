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
import shap
import matplotlib.pyplot as plt

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
# ONGLET 1 : DÉMO LIVE (Avec SHAP dynamique)
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

        # --- PREDICTION ET SHAP ---
        if st.button("Lancer l'analyse", type="primary"):
            if user_input.strip():
                with st.spinner('Analyse et interprétabilité en cours...'):
                    # 1. Pipeline
                    clean_text = processing_pipeline(user_input)
                    vec_input = vectorizer.transform([clean_text])
                    input_array = vec_input.toarray()
                    
                    pred_class = model.predict(input_array)[0]
                    pred_proba = model.predict_proba(input_array)[0]
                    
                    labels = {0: ("Négatif 😞", "red"), 1: ("Neutre 😐", "orange"), 2: ("Positif 😃", "green")}
                    label_text, color = labels[pred_class]
                    confidence = pred_proba[pred_class]

                    # 2. Résultats
                    st.divider()
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        st.markdown("### Verdict :")
                        st.markdown(f":{color}[**{label_text}**]")
                        st.metric("Confiance", f"{confidence:.1%}")
                    
                    with c2:
                        st.markdown("#### Probabilités")
                        df_chart = pd.DataFrame({"Sentiment": ["Négatif", "Neutre", "Positif"], "Probabilité": pred_proba})
                        c = alt.Chart(df_chart).mark_bar().encode(
                            x=alt.X('Sentiment', sort=None),
                            y='Probabilité',
                            color=alt.Color('Sentiment', scale=alt.Scale(domain=["Négatif", "Neutre", "Positif"], range=["#6D6D6D", "#FFB7B2", "#FF69B4"]), legend=None)
                        )
                        st.altair_chart(c, use_container_width=True)

                    # 3. Graphique SHAP Dynamique
                    st.markdown("---")
                    st.subheader("🧠 Pourquoi cette décision ? (Analyse SHAP)")
                    st.write(f"Voici les mots qui ont le plus influencé la prédiction : **{label_text}**")

                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(input_array)
                        feature_names = vectorizer.get_feature_names_out()

                        # Génération du graphique
                        fig, ax = plt.subplots(figsize=(10, 4))
                        shap.plots.bar(
                            shap.Explanation(
                                values=shap_values[pred_class][0], 
                                base_values=explainer.expected_value[pred_class], 
                                data=input_array[0], 
                                feature_names=feature_names
                            ),
                            max_display=12,
                            show=False
                        )
                        st.pyplot(fig)
                        st.caption(f"🟥 Rouge : Pousse vers '{label_text}' | 🟦 Bleu : S'y oppose")

                    except Exception as e:
                        st.warning(f"Graphique SHAP non disponible : {e}")
                    
                    with st.expander("👀 Voir le texte nettoyé"):
                        st.code(clean_text)
            else:
                st.warning("Veuillez entrer du texte.")

        # --- CSV BULK ---
        st.markdown("---")
        st.subheader("📂 Analyse de masse (Fichier CSV)")
        csv_template = "text\nExemple: Super produit !\nExemple: Livraison trop longue..."
        st.download_button("📥 Télécharger modèle CSV", csv_template, "modele_avis.csv", "text/csv")
        
        uploaded_file = st.file_uploader("Déposez votre fichier ici", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                cols = [c for c in df.columns if 'text' in c.lower() or 'review' in c.lower()]
                if cols:
                    if st.button(f"Analyser {len(df)} avis"):
                        target_col = cols[0]
                        df['clean'] = df[target_col].astype(str).apply(processing_pipeline)
                        vecs = vectorizer.transform(df['clean'])
                        preds = model.predict(vecs.toarray())
                        mapping = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                        df['Prediction'] = [mapping[p] for p in preds]
                        
                        st.dataframe(df[[target_col, 'Prediction']].style.applymap(lambda x: 'background-color: #ffcccc' if x=='Négatif' else 'background-color: #ccffcc' if x=='Positif' else 'background-color: #ffeebb', subset=['Prediction']), use_container_width=True)
                        st.download_button("📥 Télécharger résultats", df.to_csv(index=False).encode('utf-8'), "resultats.csv", "text/csv")
            except Exception as e:
                st.error(f"Erreur CSV : {e}")

# ==============================================================================
# ONGLET 2 : JEU DE DONNÉES (Slide 6 & Preprocessing)
# ==============================================================================
with tab_data:
    st.header("📚 Le Jeu de Données : Amazon Electronics")
    
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.markdown("### Source & Pourquoi ?")
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=100)
        st.info("**Amazon Reviews Dataset**")
        st.write("- **Structure identique** : Texte + Note")
        st.write("- **Focus Electronics** : Vocabulaire riche")
        st.write("- **Période** : 2010 - 2018")
    
    with col_d2:
        st.markdown("### Volumétrie & Nettoyage")
        metrics_df = pd.DataFrame({"Métrique": ["Avis bruts", "Avis après filtrage", "Langue"], "Valeur": ["~1.2 Millions", "572 950", "Anglais"]})
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.error("❌ **Prix**\n\nSupprimé (trop de NAs)")
        c2.warning("⚠️ **Votes**\n\nImputé à 0 (NAs)")
        c3.info("🖼️ **Image**\n\nBooléen (Y/N)")

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
    st.subheader("Distribution des classes (Équilibrée)")
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
    # Données issues de ton rapport
    confusion_data = pd.DataFrame(
        [[8303, 2155, 558], [2303, 6734, 1979], [551, 1765, 8700]],
        columns=["Prédit Négatif", "Prédit Neutre", "Prédit Positif"],
        index=["Réel Négatif", "Réel Neutre", "Réel Positif"]
    )
    # Affichage sécurisé sans matplotlib requis pour le style
    st.dataframe(confusion_data, use_container_width=True)
    
    st.success("✅ **Observation :** Très bonne détection des avis positifs et négatifs.")
    st.warning("⚠️ **Limite :** La classe 'Neutre' (au centre) est celle qui génère le plus de confusion.")
    
    st.markdown("---")
    st.subheader("Global Feature Importance")
    st.write("Mots les plus impactants pour le modèle :")
    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        st.error("📉 **Négatif** : bad, poor, waste, return, money")
    with col_feat2:
        st.success("📈 **Positif** : great, love, good, easy, perfect")

