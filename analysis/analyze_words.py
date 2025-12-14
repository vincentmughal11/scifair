import pickle
import pandas as pd
import numpy as np

# Load the baseline model
with open('models/baseline_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
with open('models/baseline_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Get coefficients (importance of each word)
coefficients = model.coef_[0]

# Sort by coefficient value
# Positive = AI-like, Negative = Human-like
word_importance = list(zip(feature_names, coefficients))
word_importance.sort(key=lambda x: x[1])

print("=" * 80)
print("TOP 30 HUMAN-LIKE WORDS (Negative coefficients)")
print("=" * 80)
for word, coef in word_importance[:30]:
    print(f"{word:<20} {coef:>10.4f}")

print("\n" + "=" * 80)
print("TOP 30 AI-LIKE WORDS (Positive coefficients)")
print("=" * 80)
for word, coef in word_importance[-30:]:
    print(f"{word:<20} {coef:>10.4f}")

# Save to file for easier reading
with open('word_patterns.txt', 'w', encoding='utf-8') as f:
    f.write("HUMAN-LIKE WORDS\n")
    f.write("=" * 50 + "\n")
    for word, coef in word_importance[:50]:
        f.write(f"{word:<20} {coef:>10.4f}\n")
    
    f.write("\n\nAI-LIKE WORDS\n")
    f.write("=" * 50 + "\n")
    for word, coef in word_importance[-50:]:
        f.write(f"{word:<20} {coef:>10.4f}\n")

print("\nResults saved to word_patterns.txt")
