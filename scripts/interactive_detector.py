"""
Athena AI Detector - Interactive Mode
Improved Model with Humanized Samples
"""

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import sys
import time
import shutil

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'athena_user_humanized')
AI_THRESHOLD = 0.05  # If AI probability > 5%, classify as AI

# --- UI Helpers ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_centered(text, width=None):
    if width is None:
        width = shutil.get_terminal_size().columns
    print(text.center(width))

def print_header():
    clear_screen()
    width = shutil.get_terminal_size().columns
    print("\n")
    print("=" * width)
    print_centered("ATHENA AI DETECTOR", width)
    print_centered("Advanced Detection using User-Humanized Model", width)
    print("=" * width)
    print("\n")

def print_separator():
    width = shutil.get_terminal_size().columns
    print("-" * width)

# --- Model Loading ---
def load_model():
    print(" >>> Initializing Athena System...")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")
            
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print(f" >>> Model Loaded Successfully! Device: {device}")
        time.sleep(1)
        return tokenizer, model, device
    except Exception as e:
        print(f"\n [ERROR] Failed to load model: {e}")
        sys.exit(1)

# --- Prediction Logic ---
def predict(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Softmax to get probabilities: [Human_Prob, AI_Prob]
        probs = F.softmax(logits, dim=1)[0]
        
    return probs

# --- Main Evaluation Loop ---
def main():
    tokenizer, model, device = load_model()
    print_header()
    
    print_centered("Type your text below to analyze.")
    print_centered("Type 'exit' or 'quit' to close.")
    print("\n")
    
    while True:
        try:
            print_separator()
            print("\n > Paste your text here:")
            user_input = input("   ")
            
            if user_input.lower().strip() in ['exit', 'quit']:
                print("\n >>> Exiting Athena. Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            if len(user_input.split()) < 5:
                print("\n [!] Text too short for reliable detection. Please enter more text.")
                continue

            # Simulate processing time for "feel"
            print("\n ... Analyzing ...")
            time.sleep(0.5)
            
            # Get Probabilities
            probs = predict(user_input, tokenizer, model, device)
            # human_prob = probs[0].item()
            ai_prob = probs[1].item()
            
            # Apply Custom Threshold Logic
            # "if it shows that AI confidence is more than 5%, then it's AI. else, it's human"
            if ai_prob > AI_THRESHOLD:
                result = "AI-GENERATED"
                color_code = "\033[91m" # Red
            else:
                result = "HUMAN"
                color_code = "\033[92m" # Green
                
            reset_code = "\033[0m"
            
            # Display Results
            print("\n" + "="*40)
            print(f" RESULT: {color_code}{result}{reset_code}")
            print("="*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n >>> Exiting Athena. Goodbye!")
            break
        except Exception as e:
            print(f"\n [ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()
