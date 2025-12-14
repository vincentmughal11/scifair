import pandas as pd
import pickle

# Load model
with open('models/baseline_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('models/baseline_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature names and coefficients
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Create a dict of word -> importance
word_scores = dict(zip(feature_names, coefficients))

# Get AI-like words (positive coefficients > 0.5)
ai_words = {word: score for word, score in word_scores.items() if score > 0.5}

def highlight_ai_words(text):
    """Highlight AI-indicator words in text"""
    words = text.lower().split()
    highlighted = []
    
    for word in text.split():
        word_clean = word.lower().strip('.,!?;:')
        if word_clean in ai_words:
            score = ai_words[word_clean]
            highlighted.append(f"**{word}**[{score:.2f}]")
        else:
            highlighted.append(word)
    
    return ' '.join(highlighted)

# Load dataset
df = pd.read_csv('data/clean/final_dataset.csv')

# Get some samples
humanized = df[df['label'] == 'AI_HUMANIZED'].sample(3, random_state=42)
ai_raw = df[df['label'] == 'AI_RAW'].sample(3, random_state=42)
human = df[df['label'] == 'HUMAN'].sample(3, random_state=42)

# Create markdown output
with open('highlighted_samples.md', 'w', encoding='utf-8') as f:
    f.write("# AI-Indicator Words Highlighted in Samples\n\n")
    f.write("**Legend:** Words shown as **bold**[score] are AI-indicators. Higher score = stronger AI signal.\n\n")
    f.write("---\n\n")
    
    f.write("## AI_HUMANIZED Samples (What online detectors call 0% AI)\n\n")
    for idx, row in humanized.iterrows():
        text = row['text'][:500]  # Limit length
        highlighted = highlight_ai_words(text)
        
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        f.write(f"### Sample {idx}\n\n")
        f.write(f"**Your Detector Says:** {'AI' if prediction == 1 else 'HUMAN'} ({proba[prediction]:.1%} confidence)\n\n")
        f.write(f"{highlighted}\n\n")
        f.write("---\n\n")
    
    f.write("## AI_RAW Samples (Original AI text)\n\n")
    for idx, row in ai_raw.iterrows():
        text = row['text'][:500]
        highlighted = highlight_ai_words(text)
        
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        f.write(f"### Sample {idx}\n\n")
        f.write(f"**Your Detector Says:** {'AI' if prediction == 1 else 'HUMAN'} ({proba[prediction]:.1%} confidence)\n\n")
        f.write(f"{highlighted}\n\n")
        f.write("---\n\n")
    
    f.write("## HUMAN Samples\n\n")
    for idx, row in human.iterrows():
        text = row['text'][:500]
        highlighted = highlight_ai_words(text)
        
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        f.write(f"### Sample {idx}\n\n")
        f.write(f"**Your Detector Says:** {'AI' if prediction == 1 else 'HUMAN'} ({proba[prediction]:.1%} confidence)\n\n")
        f.write(f"{highlighted}\n\n")
        f.write("---\n\n")

print("Created highlighted_samples.md")
print("\nYou can now see which words are triggering the AI detection!")
