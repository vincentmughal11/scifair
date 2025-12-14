import pandas as pd
import numpy as np
import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os
import json
from tqdm import tqdm

os.makedirs('../models', exist_ok=True)

def load_and_prep_data(pilot_mode=True):
    print("Loading and combining datasets...")
    
    hum_df = pd.read_csv('../data/Balanced_AI_Human_Humanized_UPDATED.csv')
    humanized_samples = hum_df[hum_df['generated'] == 2].copy()
    humanized_samples = humanized_samples[['text']].copy()
    humanized_samples['label_str'] = 'AI_HUMANIZED'
    
    if pilot_mode:
        target_count = 250
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
    
    full_df = pd.concat([human_df, ai_df, humanized_samples], ignore_index=True)
    print(f"Total samples: {len(full_df)}")
    return full_df

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
    """
    Extract multiple perplexity-based features like commercial detectors:
    1. Average Perplexity (whole document)
    2. Burstiness (variance of sentence perplexities)
    3. Max Perplexity (most surprising sentence)
    4. Min Perplexity (most predictable sentence)
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
    
    if not sentences:
        return [1000.0, 0.0, 1000.0, 1000.0]
    
    # Calculate perplexity for each sentence
    sentence_ppls = []
    for sent in sentences[:10]:  # Limit to first 10 sentences for speed
        ppl = calculate_perplexity(sent, model, tokenizer, max_length=512)
        sentence_ppls.append(ppl)
    
    if not sentence_ppls:
        return [1000.0, 0.0, 1000.0, 1000.0]
    
    # Feature 1: Average Perplexity
    avg_ppl = np.mean(sentence_ppls)
    
    # Feature 2: Burstiness (Standard Deviation)
    # Humans have higher variance (more unpredictable from sentence to sentence)
    burstiness = np.std(sentence_ppls)
    
    # Feature 3: Max Perplexity (most surprising sentence)
    max_ppl = np.max(sentence_ppls)
    
    # Feature 4: Min Perplexity (most predictable sentence)
    min_ppl = np.min(sentence_ppls)
    
    return [avg_ppl, burstiness, max_ppl, min_ppl]

def extract_all_features(df, model, tokenizer):
    """Extract enhanced features for all samples"""
    print("\nCalculating enhanced perplexity features...")
    print("Features: Avg PPL, Burstiness, Max PPL, Min PPL")
    
    features = []
    for text in tqdm(df['text'], desc="Processing"):
        feat = extract_enhanced_features(text, model, tokenizer)
        features.append(feat)
    
    return np.array(features)

def train_and_evaluate(full_df, pilot_mode=True):
    print("\n=== BUILDING ENHANCED PERPLEXITY DETECTOR ===")
    print("Features designed to match GPTZero/commercial detectors")
    
    # Load GPT-2
    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    # Split
    train_df, test_df = train_test_split(
        full_df, 
        test_size=0.3, 
        random_state=42, 
        stratify=full_df['label_str']
    )
    
    print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
    
    # Extract features
    X_train = extract_all_features(train_df, model, tokenizer)
    X_test = extract_all_features(test_df, model, tokenizer)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save
    mode_str = 'enhanced_pilot' if pilot_mode else 'enhanced_full'
    with open(f'../models/perplexity_{mode_str}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    results = {}
    
    # BASELINE
    print("\nTraining Baseline Enhanced Model (Human + AI_RAW)...")
    mask = train_df['label_str'].isin(['HUMAN', 'AI_RAW'])
    X_base = X_train_scaled[mask]
    y_base = train_df[mask]['label_str'].apply(lambda x: 1 if x == 'AI_RAW' else 0)
    
    base_model = LogisticRegression(random_state=42, max_iter=1000)
    base_model.fit(X_base, y_base)
    
    with open(f'../models/perplexity_{mode_str}_baseline_model.pkl', 'wb') as f:
        pickle.dump(base_model, f)
    
    # IMPROVED
    print("Training Improved Enhanced Model (All Categories)...")
    y_imp = train_df['label_str'].apply(lambda x: 0 if x == 'HUMAN' else 1)
    
    imp_model = LogisticRegression(random_state=42, max_iter=1000)
    imp_model.fit(X_train_scaled, y_imp)
    
    with open(f'../models/perplexity_{mode_str}_improved_model.pkl', 'wb') as f:
        pickle.dump(imp_model, f)
    
    # EVALUATE
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
    if hum_mask.sum() > 0:
        results['baseline_hum'] = eval_model(base_model, X_test_scaled[hum_mask], test_df[hum_mask]['label_str'], "Baseline (Humanized)")
        results['improved_hum'] = eval_model(imp_model, X_test_scaled[hum_mask], test_df[hum_mask]['label_str'], "Improved (Humanized)")
    
    with open(f'../results/perplexity_{mode_str}_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to ../results/perplexity_{mode_str}_results.json")
    
    return results

if __name__ == "__main__":
    import sys
    
    pilot_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        pilot_mode = False
    
    df = load_and_prep_data(pilot_mode=pilot_mode)
    results = train_and_evaluate(df, pilot_mode=pilot_mode)
    
    if pilot_mode:
        print("\n" + "="*60)
        print("ENHANCED PILOT TEST COMPLETE")
        print("="*60)
        print("\nIf accuracy is >60%, run full training with:")
        print("  python perplexity_enhanced.py --full")
