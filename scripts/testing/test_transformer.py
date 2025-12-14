import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def load_model(model_type='baseline'):
    """Load the trained transformer model"""
    print(f"Loading {model_type} transformer model...")
    
    model_dir = f'../models/transformer_{model_type}_quick'
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    print("Model loaded!\n")
    return tokenizer, model

def predict(text, tokenizer, model):
    """Predict if text is AI or Human"""
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
    
    return prediction, probabilities.tolist()

def analyze_text(text, baseline_tokenizer, baseline_model, improved_tokenizer, improved_model):
    """Analyze text with both models"""
    print("\n" + "=" * 80)
    print("TRANSFORMER DETECTOR RESULTS")
    print("=" * 80)
    
    # Baseline prediction
    base_pred, base_probs = predict(text, baseline_tokenizer, baseline_model)
    
    print("\n BASELINE TRANSFORMER (trained on Human + AI_RAW)")
    print(f"   Prediction:  {'🤖 AI' if base_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {base_probs[base_pred]:.1%}")
    print(f"   Probability: Human: {base_probs[0]:.1%} | AI: {base_probs[1]:.1%}")
    
    # Improved prediction
    imp_pred, imp_probs = predict(text, improved_tokenizer, improved_model)
    
    print("\n IMPROVED TRANSFORMER (trained on all categories)")
    print(f"   Prediction:  {'🤖 AI' if imp_pred == 1 else ' HUMAN'}")
    print(f"   Confidence:  {imp_probs[imp_pred]:.1%}")
    print(f"   Probability: Human: {imp_probs[0]:.1%} | AI: {imp_probs[1]:.1%}")
    
    # Agreement
    if base_pred == imp_pred:
        print(f"\n Both models AGREE: {'AI' if base_pred == 1 else 'HUMAN'}")
    else:
        print(f"\n  Models DISAGREE!")
    
    print("\n" + "=" * 80)

def main():
    # Load both models
    baseline_tokenizer, baseline_model = load_model('baseline')
    improved_tokenizer, improved_model = load_model('improved')
    
    print("=" * 80)
    print("TRANSFORMER DETECTOR - Interactive Testing")
    print("=" * 80)
    print("\nPilot test results: Baseline 99.67%, Improved 99.84%")
    print("Now testing on YOUR samples to verify it's not overfitted!")
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
        
        analyze_text(user_input, baseline_tokenizer, baseline_model, improved_tokenizer, improved_model)
        print("\n")

if __name__ == "__main__":
    main()
