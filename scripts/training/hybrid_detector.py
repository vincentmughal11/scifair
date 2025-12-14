import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os
import json

# Create models directory
os.makedirs('models', exist_ok=True)

def load_and_prep_data():
    print("Loading and combining datasets...")
    
    # 1. Load Humanized Data
    hum_df = pd.read_csv('data/Balanced_AI_Human_Humanized_UPDATED.csv')
    humanized_samples = hum_df[hum_df['generated'] == 2].copy()
    humanized_samples = humanized_samples[['text']].copy()
    humanized_samples['label_str'] = 'AI_HUMANIZED'
    
    # 2. Load Human and AI_RAW Data
    print("Reading Human and AI samples...")
    human_samples = []
    ai_samples = []
    target_count = 5000
    
    chunk_size = 50000
    for chunk in pd.read_csv('data/dataset.csv', chunksize=chunk_size):
        if len(human_samples) < target_count:
            humans = chunk[chunk['label'] == 0.0]
            human_samples.append(humans[['text']])
        
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

def extract_structure_features(text):
    """Extract structural features"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return [0, 0, 0, 0]
    
    sent_lengths = [len(s.split()) for s in sentences]
    sent_len_sd = np.std(sent_lengths) if sent_lengths else 0
    
    words = re.findall(r'\w+', text.lower())
    if not words: return [0, 0, 0, 0]
    
    type_token_ratio = len(set(words)) / len(words)
    punct_count = len(re.findall(r'[.,!?;:]', text))
    punct_density = punct_count / len(words) if words else 0
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    
    return [sent_len_sd, type_token_ratio, punct_density, avg_word_len]

def train_and_evaluate(full_df):
    print("\n=== BUILDING HYBRID DETECTOR ===")
    print("Combining TF-IDF (5000 features) + Structure (4 features) = 5004 total features")
    
    # Global Split
    train_df, test_df = train_test_split(
        full_df, 
        test_size=0.3, 
        random_state=42, 
        stratify=full_df['label_str']
    )
    
    print("\n1. Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(train_df['text'])
    X_test_tfidf = tfidf.transform(test_df['text'])
    
    print("2. Extracting Structure features...")
    train_struct = train_df['text'].apply(extract_structure_features)
    test_struct = test_df['text'].apply(extract_structure_features)
    X_train_struct = np.array(train_struct.tolist())
    X_test_struct = np.array(test_struct.tolist())
    
    # Scale structure features
    struct_scaler = StandardScaler()
    X_train_struct_scaled = struct_scaler.fit_transform(X_train_struct)
    X_test_struct_scaled = struct_scaler.transform(X_test_struct)
    
    print("3. Combining features...")
    # Convert TF-IDF sparse matrix to dense and concatenate
    X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_struct_scaled])
    X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_struct_scaled])
    
    print(f"   Combined feature shape: {X_train_combined.shape}")
    
    # Save preprocessing objects
    with open('models/hybrid_tfidf.pkl', 'wb') as f: pickle.dump(tfidf, f)
    with open('models/hybrid_struct_scaler.pkl', 'wb') as f: pickle.dump(struct_scaler, f)
    
    results = {}
    
    # --- BASELINE MODEL ---
    print("\n4. Training Baseline Hybrid Model (Human + AI_RAW)...")
    mask = train_df['label_str'].isin(['HUMAN', 'AI_RAW'])
    X_base = X_train_combined[mask]
    y_base = train_df[mask]['label_str'].apply(lambda x: 1 if x == 'AI_RAW' else 0)
    
    base_model = LogisticRegression(random_state=42, max_iter=1000)
    base_model.fit(X_base, y_base)
    
    with open('models/hybrid_baseline_model.pkl', 'wb') as f: pickle.dump(base_model, f)
    
    # --- IMPROVED MODEL ---
    print("5. Training Improved Hybrid Model (All Categories)...")
    y_imp = train_df['label_str'].apply(lambda x: 0 if x == 'HUMAN' else 1)
    
    imp_model = LogisticRegression(random_state=42, max_iter=1000)
    imp_model.fit(X_train_combined, y_imp)
    
    with open('models/hybrid_improved_model.pkl', 'wb') as f: pickle.dump(imp_model, f)
    
    # --- EVALUATION ---
    print("\n6. Evaluating...")
    
    def eval_model(model, X, y_labels, name):
        y_true = y_labels.apply(lambda x: 0 if x == 'HUMAN' else 1)
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        print(f"   {name}: {acc:.4f}")
        return acc
    
    # Standard Test
    std_mask = test_df['label_str'].isin(['HUMAN', 'AI_RAW'])
    results['baseline_std'] = eval_model(base_model, X_test_combined[std_mask], test_df[std_mask]['label_str'], "Baseline (Standard)")
    results['improved_std'] = eval_model(imp_model, X_test_combined[std_mask], test_df[std_mask]['label_str'], "Improved (Standard)")
    
    # Humanized Test
    hum_mask = test_df['label_str'] == 'AI_HUMANIZED'
    results['baseline_hum'] = eval_model(base_model, X_test_combined[hum_mask], test_df[hum_mask]['label_str'], "Baseline (Humanized)")
    results['improved_hum'] = eval_model(imp_model, X_test_combined[hum_mask], test_df[hum_mask]['label_str'], "Improved (Humanized)")
    
    with open('results/hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to results/hybrid_results.json")

if __name__ == "__main__":
    df = load_and_prep_data()
    train_and_evaluate(df)
