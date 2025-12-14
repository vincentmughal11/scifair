"""
Test script for Athena with ADJUSTABLE THRESHOLD
Allows setting custom AI detection threshold (default 5% instead of 50%)
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ADJUSTABLE THRESHOLD
AI_THRESHOLD = 0.05  # If AI probability > 5%, classify as AI (instead of default 50%)

def load_model(model_type='baseline'):
    """Load the appropriate Athena model"""
    print(f"Loading Athena {model_type} model...")
    
    if model_type == 'baseline':
        model_dir = '../models/athena_athena_data'
        accuracy = "98.82%"
    elif model_type == 'improved':
        model_dir = '../models/athena_improved'
        accuracy = "98.92%"
    elif model_type == 'user':
        model_dir = '../models/athena_user_humanized'
        accuracy = "98.90%"
    else:
        raise ValueError("model_type must be 'baseline', 'improved', or 'user'")
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    print(f"Model loaded! (Trained accuracy: {accuracy})")
    print(f"AI Detection Threshold: {AI_THRESHOLD*100:.0f}% (custom threshold)\n")
    return tokenizer, model

def predict(text, tokenizer, model):
    """Predict if text is AI or Human using custom threshold"""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        
        # Custom threshold: if AI probability > threshold, classify as AI
        ai_prob = probabilities[1].item()
        prediction = 1 if ai_prob > AI_THRESHOLD else 0
    
    return prediction, probabilities.tolist()

def analyze_text(text, tokenizer, model, model_name):
    """Analyze text with Athena detector"""
    print("\n" + "=" * 80)
    print(f"ATHENA ({model_name.upper()}) - THRESHOLD ADJUSTED")
    print("=" * 80)
    
    pred, probs = predict(text, tokenizer, model)
    
    # Show why the decision was made
    threshold_info = ""
    if pred == 1:
        if probs[1] > 0.5:
            threshold_info = " (exceeded 50% - clear AI)"
        else:
            threshold_info = f" (exceeded {AI_THRESHOLD*100:.0f}% threshold - suspected humanized AI)"
    else:
        threshold_info = f" (below {AI_THRESHOLD*100:.0f}% threshold)"
    
    print(f"\n PREDICTION: {'🤖 AI-GENERATED' if pred == 1 else ' HUMAN-WRITTEN'}{threshold_info}")
    print(f"   \n   Probabilities:")
    print(f"   - Human: {probs[0]:.1%}")
    print(f"   - AI:    {probs[1]:.1%}")
    
    if probs[0] > 0.95:
        print(f"\n    High confidence - likely genuine human text")
    elif 0.9 <= probs[0] <= 0.95:
        print(f"\n     Moderate confidence - possibly humanized AI")
    
    print("\n" + "=" * 80)

def main():
    import sys
    
    # Allow command line selection of model
    model_type = 'baseline'
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
    
    model_names = {
        'baseline': 'Baseline',
        'improved': 'Improved (Mixed Humanizers)',
        'user': 'User (Undetectable.ai)'
    }
    
    tokenizer, model = load_model(model_type)
    
    print("=" * 80)
    print(f"ATHENA {model_names[model_type].upper()} - Threshold Adjusted Testing")
    print("=" * 80)
    print(f"\n AI Threshold: {AI_THRESHOLD*100:.0f}%")
    print("   (Flags as AI if AI probability > 5% instead of default 50%)")
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
        
        analyze_text(user_input, tokenizer, model, model_names[model_type])
        print("\n")

if __name__ == "__main__":
    print("\n TIP: Run with 'python test_athena_threshold.py baseline|improved|user'")
    print("   to choose which model to use\n")
    main()
