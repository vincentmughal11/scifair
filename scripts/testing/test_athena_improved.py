"""
Test script for Athena Improved AI Detector
Tests the improved model (trained with humanized samples) on user-provided text
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def load_model():
    """Load the improved Athena model"""
    print("Loading Athena Improved model...")
    
    model_dir = '../models/athena_improved'
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    print("Model loaded! (Trained accuracy: 98.92%)\n")
    print("This model was trained with 1,045 humanized AI samples for improved robustness.\n")
    return tokenizer, model

def predict(text, tokenizer, model):
    """Predict if text is AI or Human"""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
    
    return prediction, probabilities.tolist()

def analyze_text(text, tokenizer, model):
    """Analyze text with Athena Improved detector"""
    print("\n" + "=" * 80)
    print("ATHENA IMPROVED AI DETECTOR RESULTS")
    print("=" * 80)
    
    pred, probs = predict(text, tokenizer, model)
    
    print(f"\n PREDICTION: {'🤖 AI-GENERATED' if pred == 1 else ' HUMAN-WRITTEN'}")
    print(f"   Confidence:  {probs[pred]:.1%}")
    print(f"   \n   Probabilities:")
    print(f"   - Human: {probs[0]:.1%}")
    print(f"   - AI:    {probs[1]:.1%}")
    
    print("\n" + "=" * 80)

def main():
    tokenizer, model = load_model()
    
    print("=" * 80)
    print("ATHENA IMPROVED AI DETECTOR - Interactive Testing")
    print("=" * 80)
    print("\nModel achieved 98.92% accuracy on test set")
    print("Trained with humanized AI samples for improved robustness")
    print("Based on Athena research: https://birefringent-ai.com/athena")
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
            print("   Athena was trained on 100-400 word texts.")
        
        analyze_text(user_input, tokenizer, model)
        print("\n")

if __name__ == "__main__":
    main()
