import pickle

def load_models():
    """Load both baseline and improved models"""
    print("Loading models...")
    
    # Load baseline
    with open('../models/baseline_vectorizer.pkl', 'rb') as f:
        baseline_vectorizer = pickle.load(f)
    with open('../models/baseline_model.pkl', 'rb') as f:
        baseline_model = pickle.load(f)
    
    # Load improved
    with open('../models/improved_vectorizer.pkl', 'rb') as f:
        improved_vectorizer = pickle.load(f)
    with open('../models/improved_model.pkl', 'rb') as f:
        improved_model = pickle.load(f)
    
    print("Models loaded successfully!\n")
    return baseline_vectorizer, baseline_model, improved_vectorizer, improved_model

def get_ai_words(text, vectorizer, model):
    """Get AI-indicator words from the text"""
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    word_scores = dict(zip(feature_names, coefficients))
    
    # Get AI-like words (positive coefficients > 0.5)
    ai_words = {word: score for word, score in word_scores.items() if score > 0.5}
    
    # Find AI words in the text
    found_words = []
    for word in text.lower().split():
        word_clean = word.strip('.,!?;:()"\'')
        if word_clean in ai_words:
            found_words.append((word_clean, ai_words[word_clean]))
    
    return sorted(found_words, key=lambda x: x[1], reverse=True)

def analyze_text(text, baseline_vec, baseline_model, improved_vec, improved_model):
    """Analyze text with both models"""
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    # Baseline prediction
    X_baseline = baseline_vec.transform([text])
    baseline_pred = baseline_model.predict(X_baseline)[0]
    baseline_proba = baseline_model.predict_proba(X_baseline)[0]
    
    print("\n BASELINE DETECTOR (trained on Human + AI_RAW only)")
    print(f"   Prediction:  {'🤖 AI' if baseline_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {baseline_proba[baseline_pred]:.1%}")
    print(f"   Probability: Human: {baseline_proba[0]:.1%} | AI: {baseline_proba[1]:.1%}")
    
    # Improved prediction
    X_improved = improved_vec.transform([text])
    improved_pred = improved_model.predict(X_improved)[0]
    improved_proba = improved_model.predict_proba(X_improved)[0]
    
    print("\n IMPROVED DETECTOR (trained on all categories including humanized)")
    print(f"   Prediction:  {'🤖 AI' if improved_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {improved_proba[improved_pred]:.1%}")
    print(f"   Probability: Human: {improved_proba[0]:.1%} | AI: {improved_proba[1]:.1%}")
    
    # Agreement/disagreement
    if baseline_pred == improved_pred:
        print(f"\n Both models AGREE: {'AI' if baseline_pred == 1 else 'HUMAN'}")
    else:
        print(f"\n  Models DISAGREE!")
        print(f"   Baseline says: {'AI' if baseline_pred == 1 else 'HUMAN'}")
        print(f"   Improved says: {'AI' if improved_pred == 1 else 'HUMAN'}")
    
    # Show top AI-indicator words found
    ai_words = get_ai_words(text, baseline_vec, baseline_model)
    if ai_words:
        print("\n Top AI-Indicator Words Found:")
        for word, score in ai_words[:10]:  # Show top 10
            print(f"   • '{word}' (score: {score:.2f})")
    else:
        print("\n No strong AI-indicator words found.")
    
    print("\n" + "=" * 80)

def main():
    # Load models
    baseline_vec, baseline_model, improved_vec, improved_model = load_models()
    
    print("=" * 80)
    print("AI TEXT DETECTOR - Interactive Testing Tool")
    print("=" * 80)
    print("\nEnter text to analyze, or type 'quit' to exit.")
    print("Tip: Paste longer text for better accuracy.\n")
    
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
        
        analyze_text(user_input, baseline_vec, baseline_model, improved_vec, improved_model)
        
        # Ask if they want to continue
        print("\n")

if __name__ == "__main__":
    main()
