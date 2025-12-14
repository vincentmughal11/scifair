import pandas as pd
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os
import json
from tqdm import tqdm

# Create models directory
os.makedirs('../models', exist_ok=True)

def load_and_prep_data(pilot_mode=True):
    print("Loading and combining datasets...")
    
    # Load Humanized Data
    hum_df = pd.read_csv('../data/Balanced_AI_Human_Humanized_UPDATED.csv')
    humanized_samples = hum_df[hum_df['generated'] == 2].copy()
    humanized_samples = humanized_samples[['text']].copy()
    humanized_samples['label_str'] = 'AI_HUMANIZED'
    
    # Load Human and AI_RAW Data
    print("Reading Human and AI samples...")
    if pilot_mode:
        target_count = 250  # 250 each for pilot = 500 total + humanized
        print("PILOT MODE: Using 250 samples per category")
    else:
        target_count = 5000
        print("FULL MODE: Using 5000 samples per category")
    
    human_samples = []
    ai_samples = []
    
    chunk_size = 50000
    for chunk in pd.read_csv('../data/dataset.csv', chunksize=chunk_size):
        if len(human_samples) < target_count:
            humans = chunk[chunk['label'] == 0.0]
            human_samples.append(humans[['text']])
        
        if len(ai_samples) < target_count:
            ais = chunk[chunk['label'] == 1.0]
            ai_samples.append(ais[['text']])
            
        h_count = sum([len(x) for x in human_samples])
        a_count = sum([len(x) for x in ai_samples])
        if h_count >= target_count and a_count >= target_count:
            break
    
    human_df = pd.concat(human_samples).head(target_count).copy()
    human_df['label_str'] = 'HUMAN'
    ai_df = pd.concat(ai_samples).head(target_count).copy()
    ai_df['label_str'] = 'AI_RAW'
    
    # Use all humanized for pilot too (it's only ~1045)
    full_df = pd.concat([human_df, ai_df, humanized_samples], ignore_index=True)
    print(f"Total samples: {len(full_df)}")
    return full_df

def calculate_perplexity(text, model, tokenizer, max_length=512):
    """Calculate perplexity for a given text"""
    try:
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    except Exception as e:
        # If calculation fails, return a high perplexity
        return 1000.0

def extract_perplexity_features(df, model, tokenizer):
    """Extract perplexity-based features for all samples"""
    print("\nCalculating perplexity features (this will take a while)...")
    
    features = []
    for text in tqdm(df['text'], desc="Processing"):
        # Calculate perplexity
        ppl = calculate_perplexity(text, model, tokenizer)
        
        # For now, just use perplexity as the single feature
        # Later could add: burstiness (variance across sentences), etc.
        features.append([ppl])
    
    return np.array(features)

def train_and_evaluate(full_df, pilot_mode=True):
    print("\n=== BUILDING PERPLEXITY DETECTOR ===")
    
    # Load GPT-2 model
    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()  # Set to evaluation mode
    
    # Global Split
    train_df, test_df = train_test_split(
        full_df, 
        test_size=0.3, 
        random_state=42, 
        stratify=full_df['label_str']
    )
    
    print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
    
    # Extract features
    X_train = extract_perplexity_features(train_df, model, tokenizer)
    X_test = extract_perplexity_features(test_df, model, tokenizer)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    mode_str = 'pilot' if pilot_mode else 'full'
    with open(f'../models/perplexity_{mode_str}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    results = {}
    
    # --- BASELINE MODEL ---
    print("\nTraining Baseline Perplexity Model (Human + AI_RAW)...")
    mask = train_df['label_str'].isin(['HUMAN', 'AI_RAW'])
    X_base = X_train_scaled[mask]
    y_base = train_df[mask]['label_str'].apply(lambda x: 1 if x == 'AI_RAW' else 0)
    
    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_base, y_base)
    
    with open(f'../models/perplexity_{mode_str}_baseline_model.pkl', 'wb') as f:
        pickle.dump(base_model, f)
    
    # --- IMPROVED MODEL ---
    print("Training Improved Perplexity Model (All Categories)...")
    y_imp = train_df['label_str'].apply(lambda x: 0 if x == 'HUMAN' else 1)
    
    imp_model = LogisticRegression(random_state=42)
    imp_model.fit(X_train_scaled, y_imp)
    
    with open(f'../models/perplexity_{mode_str}_improved_model.pkl', 'wb') as f:
        pickle.dump(imp_model, f)
    
    # --- EVALUATION ---
    print("\nEvaluating...")
    
    def eval_model(model, X, y_labels, name):
        y_true = y_labels.apply(lambda x: 0 if x == 'HUMAN' else 1)
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        print(f"   {name}: {acc:.4f}")
        return acc
    
    # Standard Test
    std_mask = test_df['label_str'].isin(['HUMAN', 'AI_RAW'])
    results['baseline_std'] = eval_model(base_model, X_test_scaled[std_mask], test_df[std_mask]['label_str'], "Baseline (Standard)")
    results['improved_std'] = eval_model(imp_model, X_test_scaled[std_mask], test_df[std_mask]['label_str'], "Improved (Standard)")
    
    # Humanized Test
    hum_mask = test_df['label_str'] == 'AI_HUMANIZED'
    results['baseline_hum'] = eval_model(base_model, X_test_scaled[hum_mask], test_df[hum_mask]['label_str'], "Baseline (Humanized)")
    results['improved_hum'] = eval_model(imp_model, X_test_scaled[hum_mask], test_df[hum_mask]['label_str'], "Improved (Humanized)")
    
    with open(f'../results/perplexity_{mode_str}_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to ../results/perplexity_{mode_str}_results.json")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Check if user wants full mode
    pilot_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        pilot_mode = False
    
    df = load_and_prep_data(pilot_mode=pilot_mode)
    results = train_and_evaluate(df, pilot_mode=pilot_mode)
    
    if pilot_mode:
        print("\n" + "="*60)
        print("PILOT TEST COMPLETE")
        print("="*60)
        print("\nTo run full training (3 hours), use:")
        print("  python perplexity_detector.py --full")
