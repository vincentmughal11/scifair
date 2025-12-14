import pickle
import numpy as np
import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_models():
    """Load enhanced perplexity detector models"""
    print("Loading enhanced perplexity models...")
    print("(Loading GPT-2... this may take a moment)")
    
    # Load GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model.eval()
    
    # Load classifier and scaler
    with open('../models/perplexity_enhanced_full_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('../models/perplexity_enhanced_full_baseline_model.pkl', 'rb') as f:
        baseline_model = pickle.load(f)
    with open('../models/perplexity_enhanced_full_improved_model.pkl', 'rb') as f:
        improved_model = pickle.load(f)
    
    print("Models loaded!\n")
    return tokenizer, gpt2_model, scaler, baseline_model, improved_model

def calculate_perplexity(text, model, tokenizer, max_length=100):
    """Calculate perplexity for text"""
    try:
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        return perplexity
    except:
        return 1000.0

def extract_enhanced_features(text, model, tokenizer):
    """Extract 4 perplexity-based features"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
    
    if not sentences:
        return [1000.0, 0.0, 1000.0, 1000.0]
    
    sentence_ppls = []
    for sent in sentences[:10]:
        ppl = calculate_perplexity(sent, model, tokenizer, max_length=512)
        sentence_ppls.append(ppl)
    
    if not sentence_ppls:
        return [1000.0, 0.0, 1000.0, 1000.0]
    
    avg_ppl = np.mean(sentence_ppls)
    burstiness = np.std(sentence_ppls)
    max_ppl = np.max(sentence_ppls)
    min_ppl = np.min(sentence_ppls)
    
    return [avg_ppl, burstiness, max_ppl, min_ppl]

def analyze_text(text, tokenizer, gpt2_model, scaler, baseline_model, improved_model):
    """Analyze text with enhanced perplexity detector"""
    print("\n" + "=" * 80)
    print("ENHANCED PERPLEXITY DETECTOR RESULTS")
    print("=" * 80)
    
    # Extract features
    features = extract_enhanced_features(text, gpt2_model, tokenizer)
    
    print(f"\n PERPLEXITY FEATURES:")
    print(f"   Avg Perplexity:       {features[0]:.2f}")
    print(f"   Burstiness (Var):     {features[1]:.2f}")
    print(f"   Max Perplexity:       {features[2]:.2f}")
    print(f"   Min Perplexity:       {features[3]:.2f}")
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    
    # Baseline
    baseline_pred = baseline_model.predict(features_scaled)[0]
    baseline_proba = baseline_model.predict_proba(features_scaled)[0]
    
    print("\n BASELINE ENHANCED (trained on Human + AI_RAW)")
    print(f"   Prediction:  {'🤖 AI' if baseline_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {baseline_proba[baseline_pred]:.1%}")
    print(f"   Probability: Human: {baseline_proba[0]:.1%} | AI: {baseline_proba[1]:.1%}")
    
    # Improved
    improved_pred = improved_model.predict(features_scaled)[0]
    improved_proba = improved_model.predict_proba(features_scaled)[0]
    
    print("\n IMPROVED ENHANCED (trained on all categories)")
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
    tokenizer, gpt2_model, scaler, baseline_model, improved_model = load_models()
    
    print("=" * 80)
    print("ENHANCED PERPLEXITY DETECTOR - Interactive Testing")
    print("=" * 80)
    print("\nUses 4 features: Avg PPL, Burstiness, Max PPL, Min PPL")
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
        
        analyze_text(user_input, tokenizer, gpt2_model, scaler, baseline_model, improved_model)
        print("\n")

if __name__ == "__main__":
    main()
