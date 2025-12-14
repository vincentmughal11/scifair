import pickle
import numpy as np
import re

def load_models():
    """Load hybrid detector models"""
    print("Loading hybrid models...")
    
    with open('../models/hybrid_tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('../models/hybrid_struct_scaler.pkl', 'rb') as f:
        struct_scaler = pickle.load(f)
    with open('../models/hybrid_baseline_model.pkl', 'rb') as f:
        baseline_model = pickle.load(f)
    with open('../models/hybrid_improved_model.pkl', 'rb') as f:
        improved_model = pickle.load(f)
    
    print("Models loaded!\n")
    return tfidf, struct_scaler, baseline_model, improved_model

def extract_structure_features(text):
    """Extract structural features"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return [0, 0, 0, 0]
    
    sent_lengths = [len(s.split()) for s in sentences]
    sent_len_sd = np.std(sent_lengths) if sent_lengths else 0
    
    words = re.findall(r'\w+', text.lower())
    if not words: return [0, 0, 0, 0]
    
    type_token_ratio = len(set(words)) / len(words)
    punct_count = len(re.findall(r'[.,!?;:]', text))
    punct_density = punct_count / len(words) if words else 0
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    
    return [sent_len_sd, type_token_ratio, punct_density, avg_word_len]

def analyze_text(text, tfidf, struct_scaler, baseline_model, improved_model):
    """Analyze text with hybrid models"""
    print("\n" + "=" * 80)
    print("HYBRID DETECTOR RESULTS (TF-IDF + Structure)")
    print("=" * 80)
    
    # Extract TF-IDF features
    tfidf_features = tfidf.transform([text]).toarray()
    
    # Extract Structure features
    struct_features = extract_structure_features(text)
    struct_features_scaled = struct_scaler.transform([struct_features])
    
    # Combine
    combined_features = np.hstack([tfidf_features, struct_features_scaled])
    
    # Display features
    print("\n FEATURES:")
    print(f"   TF-IDF:                   {tfidf_features.shape[1]} word features")
    print(f"   Sentence Length Var:      {struct_features[0]:.2f}")
    print(f"   Vocabulary Richness:      {struct_features[1]:.3f}")
    print(f"   Punctuation Density:      {struct_features[2]:.3f}")
    print(f"   Avg Word Length:          {struct_features[3]:.2f}")
    
    # Baseline prediction
    baseline_pred = baseline_model.predict(combined_features)[0]
    baseline_proba = baseline_model.predict_proba(combined_features)[0]
    
    print("\n BASELINE HYBRID (trained on Human + AI_RAW)")
    print(f"   Prediction:  {'🤖 AI' if baseline_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {baseline_proba[baseline_pred]:.1%}")
    print(f"   Probability: Human: {baseline_proba[0]:.1%} | AI: {baseline_proba[1]:.1%}")
    
    # Improved prediction
    improved_pred = improved_model.predict(combined_features)[0]
    improved_proba = improved_model.predict_proba(combined_features)[0]
    
    print("\n IMPROVED HYBRID (trained on all categories)")
    print(f"   Prediction:  {'🤖 AI' if improved_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {improved_proba[improved_pred]:.1%}")
    print(f"   Probability: Human: {improved_proba[0]:.1%} | AI: {improved_proba[1]:.1%}")
    
    # Agreement
    if baseline_pred == improved_pred:
        print(f"\n Both models AGREE: {'AI' if baseline_pred == 1 else 'HUMAN'}")
    else:
        print(f"\n  Models DISAGREE!")
    
    print("\n" + "=" * 80)

def main():
    tfidf, struct_scaler, baseline_model, improved_model = load_models()
    
    print("=" * 80)
    print("HYBRID DETECTOR - Interactive Testing")
    print("=" * 80)
    print("\nCombines Vocabulary (TF-IDF) + Style (Structure) analysis")
    print("Enter text to analyze, or type 'quit' to exit.\n")
    
    while True:
        print("-" * 80)
        user_input = input("\n Enter text to analyze (or 'quit' to exit):\n> ")
        
        if user_input.lower().strip() in ['quit', 'exit', 'q']:
            print("\nGoodbye! ")
            break
        
        if not user_input.strip():
            print("  Please enter some text.")
            continue
        
        if len(user_input.split()) < 10:
            print("  Warning: Very short text may give unreliable results.")
        
        analyze_text(user_input, tfidf, struct_scaler, baseline_model, improved_model)
        print("\n")

if __name__ == "__main__":
    main()
