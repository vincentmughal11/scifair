import pickle
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_models():
    """Load perplexity detector models"""
    print("Loading perplexity models...")
    print("(Note: This may take a moment to load GPT-2)")
    
    # Load GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model.eval()
    
    # Load classifier and scaler
    with open('../models/perplexity_pilot_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('../models/perplexity_pilot_baseline_model.pkl', 'rb') as f:
        baseline_model = pickle.load(f)
    with open('../models/perplexity_pilot_improved_model.pkl', 'rb') as f:
        improved_model = pickle.load(f)
    
    print("Models loaded!\n")
    return tokenizer, gpt2_model, scaler, baseline_model, improved_model

def calculate_perplexity(text, model, tokenizer, max_length=512):
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

def analyze_text(text, tokenizer, gpt2_model, scaler, baseline_model, improved_model):
    """Analyze text with perplexity detector"""
    print("\n" + "=" * 80)
    print("PERPLEXITY DETECTOR RESULTS")
    print("=" * 80)
    
    # Calculate perplexity
    ppl = calculate_perplexity(text, gpt2_model, tokenizer)
    
    print(f"\n PERPLEXITY SCORE: {ppl:.2f}")
    print("   (Lower = more predictable to AI, Higher = more surprising)")
    
    # Scale and predict
    features_scaled = scaler.transform([[ppl]])
    
    # Baseline
    baseline_pred = baseline_model.predict(features_scaled)[0]
    baseline_proba = baseline_model.predict_proba(features_scaled)[0]
    
    print("\n BASELINE PERPLEXITY DETECTOR (trained on Human + AI_RAW)")
    print(f"   Prediction:  {'🤖 AI' if baseline_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {baseline_proba[baseline_pred]:.1%}")
    print(f"   Probability: Human: {baseline_proba[0]:.1%} | AI: {baseline_proba[1]:.1%}")
    
    # Improved
    improved_pred = improved_model.predict(features_scaled)[0]
    improved_proba = improved_model.predict_proba(features_scaled)[0]
    
    print("\n IMPROVED PERPLEXITY DETECTOR (trained on all categories)")
    print(f"   Prediction:  {'🤖 AI' if improved_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {improved_proba[improved_pred]:.1%}")
    print(f"   Probability: Human: {improved_proba[0]:.1%} | AI: {improved_proba[1]:.1%}")
    
    # Warning
    print("\n  WARNING: This detector performed poorly in testing (50% accuracy)")
    print("   Consider using the TF-IDF or Hybrid detector instead.")
    
    print("\n" + "=" * 80)

def main():
    tokenizer, gpt2_model, scaler, baseline_model, improved_model = load_models()
    
    print("=" * 80)
    print("PERPLEXITY DETECTOR - Interactive Testing")
    print("=" * 80)
    print("\n  NOTE: This detector had poor accuracy in testing.")
    print("It measures text 'predictability' but struggles to distinguish AI from human.")
    print("\nEnter text to analyze, or type 'quit' to exit.\n")
    
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
