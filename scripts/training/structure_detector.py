import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os
import json

# Create models directory
os.makedirs('models', exist_ok=True)

def load_and_prep_data():
    print("Loading and combining datasets...")
    
    # 1. Load Humanized Data (from Balanced CSV)
    print("Reading Humanized samples...")
    hum_df = pd.read_csv('data/Balanced_AI_Human_Humanized_UPDATED.csv')
    humanized_samples = hum_df[hum_df['generated'] == 2].copy()
    humanized_samples = humanized_samples[['text']].copy()
    humanized_samples['label_str'] = 'AI_HUMANIZED'
    
    # 2. Load Human and AI_RAW Data (from Large Dataset CSV)
    print("Reading Human and AI samples from large dataset...")
    human_samples = []
    ai_samples = []
    target_count = 5000
    
    chunk_size = 50000
    for chunk in pd.read_csv('data/dataset.csv', chunksize=chunk_size):
        if len(human_samples) < target_count:
            humans = chunk[chunk['label'] == 0.0]
            human_samples.append(humans[['text']])
            current_count = sum([len(x) for x in human_samples])
            if current_count >= target_count: pass
        
        if len(ai_samples) < target_count:
            ais = chunk[chunk['label'] == 1.0]
            ai_samples.append(ais[['text']])
            
        h_count = sum([len(x) for x in human_samples])
        a_count = sum([len(x) for x in ai_samples])
        if h_count >= target_count and a_count >= target_count: break
    
    human_df = pd.concat(human_samples).head(target_count).copy()
    human_df['label_str'] = 'HUMAN'
    ai_df = pd.concat(ai_samples).head(target_count).copy()
    ai_df['label_str'] = 'AI_RAW'
    
    # 3. Combine
    full_df = pd.concat([human_df, ai_df, humanized_samples], ignore_index=True)
    print(f"Total samples: {len(full_df)}")
    return full_df

def extract_features(text):
    """Extract structural features from text"""
    # 1. Sentence Length SD
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return [0, 0, 0, 0]
    
    sent_lengths = [len(s.split()) for s in sentences]
    sent_len_sd = np.std(sent_lengths) if sent_lengths else 0
    
    # 2. Vocabulary Richness (Type-Token Ratio)
    words = re.findall(r'\w+', text.lower())
    if not words: return [0, 0, 0, 0]
    type_token_ratio = len(set(words)) / len(words)
    
    # 3. Punctuation Density
    punct_count = len(re.findall(r'[.,!?;:]', text))
    punct_density = punct_count / len(words) if words else 0
    
    # 4. Avg Word Length
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    
    return [sent_len_sd, type_token_ratio, punct_density, avg_word_len]

def train_and_evaluate(full_df):
    print("\nExtracting features (this may take a minute)...")
    # Apply feature extraction
    features = full_df['text'].apply(extract_features)
    X = np.array(features.tolist())
    
    # Global Split
    train_indices, test_indices = train_test_split(
        np.arange(len(full_df)), 
        test_size=0.3, 
        random_state=42, 
        stratify=full_df['label_str']
    )
    
    X_train_full = X[train_indices]
    y_train_full = full_df.iloc[train_indices]['label_str']
    
    X_test_full = X[test_indices]
    y_test_full = full_df.iloc[test_indices]['label_str']
    
    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train_full) # Fit on full training set
    
    # Save scaler
    with open('models/structure_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    X_train_scaled = scaler.transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)
    
    results = {}
    
    # --- BASELINE MODEL (Human + AI_RAW only) ---
    print("\nTraining Baseline Structure Model...")
    mask = y_train_full.isin(['HUMAN', 'AI_RAW'])
    X_base = X_train_scaled[mask]
    y_base = y_train_full[mask].apply(lambda x: 1 if x == 'AI_RAW' else 0)
    
    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_base, y_base)
    
    with open('models/structure_baseline_model.pkl', 'wb') as f:
        pickle.dump(base_model, f)
        
    # --- IMPROVED MODEL (All Categories) ---
    print("Training Improved Structure Model...")
    y_imp = y_train_full.apply(lambda x: 0 if x == 'HUMAN' else 1)
    
    imp_model = LogisticRegression(random_state=42)
    imp_model.fit(X_train_scaled, y_imp)
    
    with open('models/structure_improved_model.pkl', 'wb') as f:
        pickle.dump(imp_model, f)
        
    # --- EVALUATION ---
    def eval_model(model, X, y_labels, name):
        y_true = y_labels.apply(lambda x: 0 if x == 'HUMAN' else 1)
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        return acc

    # Standard Test Set
    std_mask = y_test_full.isin(['HUMAN', 'AI_RAW'])
    results['baseline_std'] = eval_model(base_model, X_test_scaled[std_mask], y_test_full[std_mask], "Baseline (Standard)")
    results['improved_std'] = eval_model(imp_model, X_test_scaled[std_mask], y_test_full[std_mask], "Improved (Standard)")
    
    # Humanized Test Set
    hum_mask = y_test_full == 'AI_HUMANIZED'
    results['baseline_hum'] = eval_model(base_model, X_test_scaled[hum_mask], y_test_full[hum_mask], "Baseline (Humanized)")
    results['improved_hum'] = eval_model(imp_model, X_test_scaled[hum_mask], y_test_full[hum_mask], "Improved (Humanized)")
    
    with open('structure_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to structure_results.json")

if __name__ == "__main__":
    df = load_and_prep_data()
    train_and_evaluate(df)
