import pandas as pd
import pickle

# Load the dataset
df = pd.read_csv('data/clean/final_dataset.csv')

# Load model
with open('models/baseline_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('models/baseline_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get some humanized samples
humanized = df[df['label'] == 'AI_HUMANIZED'].sample(5, random_state=42)

print("=" * 80)
print("SAMPLE HUMANIZED TEXTS (What online detectors call 0% AI)")
print("=" * 80)

for idx, row in humanized.iterrows():
    text = row['text']
    
    # Get prediction
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    print(f"\nText: {text[:200]}...")
    print(f"Model says: {'AI' if prediction == 1 else 'HUMAN'} (confidence: {proba[prediction]:.2%})")
    print("-" * 80)

# Save full samples to file
with open('humanized_samples.txt', 'w', encoding='utf-8') as f:
    for idx, row in humanized.iterrows():
        text = row['text']
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        f.write(f"SAMPLE {idx}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{text}\n\n")
        f.write(f"Prediction: {'AI' if prediction == 1 else 'HUMAN'}\n")
        f.write(f"Confidence: {proba[prediction]:.2%}\n")
        f.write("\n" + "=" * 80 + "\n\n")

print("\nFull samples saved to humanized_samples.txt")
