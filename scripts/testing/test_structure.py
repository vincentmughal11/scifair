import pickle
import numpy as np
import re

def load_models():
    """Load structure detector models"""
    print("Loading models...")
    
    with open('../models/structure_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('../models/structure_baseline_model.pkl', 'rb') as f:
        baseline_model = pickle.load(f)
    with open('../models/structure_improved_model.pkl', 'rb') as f:
        improved_model = pickle.load(f)
    
    print("Models loaded successfully!\n")
    return scaler, baseline_model, improved_model

def extract_features(text):
    """Extract structural features from text"""
    # 1. Sentence Length SD
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [0, 0, 0, 0]
    
    sent_lengths = [len(s.split()) for s in sentences]
    sent_len_sd = np.std(sent_lengths) if sent_lengths else 0
    
    # 2. Vocabulary Richness (Type-Token Ratio)
    words = re.findall(r'\w+', text.lower())
    if not words:
        return [0, 0, 0, 0]
    type_token_ratio = len(set(words)) / len(words)
    
    # 3. Punctuation Density
    punct_count = len(re.findall(r'[.,!?;:]', text))
    punct_density = punct_count / len(words) if words else 0
    
    # 4. Avg Word Length
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    
    return [sent_len_sd, type_token_ratio, punct_density, avg_word_len]

def analyze_text(text, scaler, baseline_model, improved_model):
    """Analyze text with both models"""
    print("\n" + "=" * 80)
    print("STRUCTURE ANALYSIS RESULTS")
    print("=" * 80)
    
    # Extract features
    features = extract_features(text)
    features_scaled = scaler.transform([features])
    
    # Display features
    print("\n EXTRACTED FEATURES:")
    print(f"   Sentence Length Variation: {features[0]:.2f}")
    print(f"   Vocabulary Richness:       {features[1]:.3f}")
    print(f"   Punctuation Density:       {features[2]:.3f}")
    print(f"   Avg Word Length:           {features[3]:.2f}")
    
    # Baseline prediction
    baseline_pred = baseline_model.predict(features_scaled)[0]
    baseline_proba = baseline_model.predict_proba(features_scaled)[0]
    
    print("\n BASELINE STRUCTURE DETECTOR (trained on Human + AI_RAW only)")
    print(f"   Prediction:  {'🤖 AI' if baseline_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {baseline_proba[baseline_pred]:.1%}")
    print(f"   Probability: Human: {baseline_proba[0]:.1%} | AI: {baseline_proba[1]:.1%}")
    
    # Improved prediction
    improved_pred = improved_model.predict(features_scaled)[0]
    improved_proba = improved_model.predict_proba(features_scaled)[0]
    
    print("\n IMPROVED STRUCTURE DETECTOR (trained on all categories)")
    print(f"   Prediction:  {'🤖 AI' if improved_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {improved_proba[improved_pred]:.1%}")
    print(f"   Probability: Human: {improved_proba[0]:.1%} | AI: {improved_proba[1]:.1%}")
    
    # Agreement/disagreement
    if baseline_pred == improved_pred:
        print(f"\n Both models AGREE: {'AI' if baseline_pred == 1 else 'HUMAN'}")
    else:
        print(f"\n  Models DISAGREE!")
    
    print("\n" + "=" * 80)

def main():
    # Load models
    scaler, baseline_model, improved_model = load_models()
    
    print("=" * 80)
    print("STRUCTURE DETECTOR - Interactive Testing Tool")
    print("=" * 80)
    print("\nThis detector analyzes writing STYLE, not vocabulary.")
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
        
        analyze_text(user_input, scaler, baseline_model, improved_model)
        print("\n")

if __name__ == "__main__":
    main()
