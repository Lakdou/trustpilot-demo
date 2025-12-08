import pandas as pd

# Création de fausses données variées pour bien tester les couleurs
data = {
    'text': [
        "Absolutely amazing! The delivery was super fast and I love the product.", # Positif
        "Terrible experience. The item arrived broken and customer service was rude.", # Négatif
        "It is okay. Not great, not bad. Just an average product.", # Neutre
        "Highly recommended! I will definitely buy again from this shop.", # Positif
        "Total scam! Do not buy from here. Waste of money.", # Négatif
        "The package was a bit late but the item works fine.", # Neutre
        "Five stars! exceptional quality.", # Positif
        "I demanded a refund and they ignored my emails.", # Négatif
        "Reasonable price for the quality. Satisfied.", # Neutre/Positif
        "Horrible. Worst purchase of my life." # Négatif
    ]
}

# Création du DataFrame
df = pd.DataFrame(data)

# Sauvegarde en CSV
df.to_csv("avis_test.csv", index=False)

print("✅ Fichier 'avis_test.csv' créé avec succès ! Tu peux l'utiliser dans Streamlit.")