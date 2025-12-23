import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import tqdm

def get_humanized_data():
    file_path = '../data/Balanced_AI_Human_Humanized_UPDATED.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} NOT FOUND")
        return None
    
    df = pd.read_csv(file_path)
    # Humanized samples are 'generated' == 2 (AI_Humanized)
    humanized_samples = df[df['generated'] == 2].copy()
    texts = humanized_samples['text'].tolist()
    return texts

def evaluate_model(model_path, texts, batch_size=16):
    print(f"Evaluating {model_path}...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    correct = 0
    total = len(texts)
    
    # Process in batches
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        # Truncation is crucial here
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            # Label 1 is AI. All humanized samples are AI.
            correct += (preds == 1).sum().item()
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {min(i+batch_size, total)}/{total} samples...", end='\r')
            
    print(f"Processed {total}/{total} samples.        ")
    accuracy = correct / total
    return accuracy

def main():
    print("Loading Humanized Data...")
    texts = get_humanized_data()
    if not texts:
        return
        
    print(f"Found {len(texts)} humanized samples.")
    
    models = {
        'Transformer': '../models/transformer_baseline_quick',
        # 'Transformer Improved': '../models/transformer_improved_quick', # User only asked for Baseline vs Improved generally, but I should check what is available. 
        # Actually user asked for "baseline humanized accuracy vs improved humanized accuracy".
        # So I need both baseline and improved for both types if possible.
        'Transformer Improved': '../models/transformer_improved_quick',
        'Athena': '../models/athena_athena_data',
        'Athena Improved': '../models/athena_improved'
    }
    
    results = {}
    
    for name, path in models.items():
        if os.path.exists(path):
            acc = evaluate_model(path, texts)
            if acc is not None:
                results[name] = acc
                print(f"{name}: {acc:.2%}")
        else:
            print(f"Model path not found: {path} (Skipping)")
            
    # Print Summary for easy copying
    print("\n" + "="*40)
    print("SUMMARY RESULTS")
    print("="*40)
    print("Model,Humanized Accuracy")
    for name, acc in results.items():
        print(f"{name},{acc:.2%}")

if __name__ == "__main__":
    main()
