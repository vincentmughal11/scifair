import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze_humanized_detection():
    print("Loading Baseline Model...")
    with open('models/baseline_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/baseline_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    print("Loading Humanized Samples...")
    df = pd.read_csv('data/Balanced_AI_Human_Humanized_UPDATED.csv')
    humanized_df = df[df['generated'] == 2].copy()
    
    print(f"Analyzing {len(humanized_df)} humanized samples...")
    
    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    word_scores = dict(zip(feature_names, coefficients))
    
    # Counter for triggering words
    trigger_words = {}
    
    for text in humanized_df['text']:
        # Simple tokenization matching the vectorizer's style roughly
        words = text.lower().split()
        for word in words:
            word = word.strip('.,!?;:()"\'')
            if word in word_scores:
                score = word_scores[word]
                # Only count if it's a strong AI indicator (> 1.0)
                if score > 1.0:
                    trigger_words[word] = trigger_words.get(word, 0) + 1
    
    # Sort by frequency
    sorted_triggers = sorted(trigger_words.items(), key=lambda x: x[1], reverse=True)
    
    with open('trigger_analysis.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("TOP WORDS TRIGGERING DETECTION IN HUMANIZED TEXT\n")
        f.write("="*60 + "\n")
        f.write(f"{'WORD':<20} {'COUNT':<10} {'AI SCORE':<10}\n")
        f.write("-" * 40 + "\n")
        
        for word, count in sorted_triggers[:30]:
            score = word_scores[word]
            f.write(f"{word:<20} {count:<10} {score:.2f}\n")
            
        f.write("\n" + "="*60 + "\n")
        f.write("INTERPRETATION\n")
        f.write("These are the words that the Baseline model (trained ONLY on AI Raw)\n")
        f.write("is finding inside the Humanized text.\n")
        f.write("If these are common words, it proves the 'Vocabulary Hypothesis'.\n")
    
    print("Analysis saved to trigger_analysis.txt")

if __name__ == "__main__":
    analyze_humanized_detection()
