import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    # Filter for label 2 (Humanized)
    humanized_samples = hum_df[hum_df['generated'] == 2].copy()
    humanized_samples = humanized_samples[['text']].copy()
    humanized_samples['label_str'] = 'AI_HUMANIZED'
    print(f"Loaded {len(humanized_samples)} Humanized samples")
    
    # 2. Load Human and AI_RAW Data (from Large Dataset CSV)
    print("Reading Human and AI samples from large dataset...")
    # We read in chunks to avoid memory issues and find enough samples
    human_samples = []
    ai_samples = []
    target_count = 5000
    
    chunk_size = 50000
    for chunk in pd.read_csv('data/dataset.csv', chunksize=chunk_size):
        # Human (Label 0)
        if len(human_samples) < target_count:
            humans = chunk[chunk['label'] == 0.0]
            human_samples.append(humans[['text']])
            # Flatten list of dfs temporarily to check count
            current_count = sum([len(x) for x in human_samples])
            if current_count >= target_count:
                # We have enough, stop collecting humans
                pass
        
        # AI (Label 1)
        if len(ai_samples) < target_count:
            ais = chunk[chunk['label'] == 1.0]
            ai_samples.append(ais[['text']])
            
        # Check if we have enough of both
        h_count = sum([len(x) for x in human_samples])
        a_count = sum([len(x) for x in ai_samples])
        
        if h_count >= target_count and a_count >= target_count:
            break
    
    # Concatenate and trim to exact target count
    human_df = pd.concat(human_samples).head(target_count).copy()
    human_df['label_str'] = 'HUMAN'
    
    ai_df = pd.concat(ai_samples).head(target_count).copy()
    ai_df['label_str'] = 'AI_RAW'
    
    print(f"Loaded {len(human_df)} Human samples")
    print(f"Loaded {len(ai_df)} AI samples")
    
    # 3. Combine Everything
    full_df = pd.concat([human_df, ai_df, humanized_samples], ignore_index=True)
    
    print("\nFinal Class Distribution:")
    print(full_df['label_str'].value_counts())
    
    # Global Split: 70% Train, 30% Test
    train_df, test_df = train_test_split(
        full_df, 
        test_size=0.3, 
        random_state=42, 
        stratify=full_df['label_str']
    )
    
    print(f"\nGlobal Train Set: {len(train_df)} rows")
    print(f"Global Test Set: {len(test_df)} rows")
    
    return train_df, test_df

def train_baseline(train_df):
    print("\n--- Training Baseline (Human + AI_RAW only) ---")
    
    # Filter for ONLY Human and AI_RAW
    mask = train_df['label_str'].isin(['HUMAN', 'AI_RAW'])
    baseline_train = train_df[mask].copy()
    print(f"Training Data: {len(baseline_train)} rows")
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(baseline_train['text'])
    
    # Labels: 1 for AI_RAW, 0 for HUMAN
    y_train = baseline_train['label_str'].apply(lambda x: 1 if x == 'AI_RAW' else 0)
    
    # Train
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    return vectorizer, model

def train_improved(train_df):
    print("\n--- Training Improved (All Categories) ---")
    print(f"Training Data: {len(train_df)} rows (Human + AI_RAW + AI_Humanized)")
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['text'])
    
    # Labels: 1 for ANY AI (Raw or Humanized), 0 for HUMAN
    y_train = train_df['label_str'].apply(lambda x: 0 if x == 'HUMAN' else 1)
    
    # Train
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    return vectorizer, model

def evaluate(vectorizer, model, test_df, name):
    print(f"\nEvaluating {name}...")
    
    X_test = vectorizer.transform(test_df['text'])
    y_true = test_df['label_str'].apply(lambda x: 0 if x == 'HUMAN' else 1)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        # Handle edge cases (e.g. only one class present)
        tn, fp, fn, tp = 0, 0, 0, 0
        fpr, fnr = 0, 0
        if len(np.unique(y_true)) == 1:
            if y_true.iloc[0] == 0: # Only Human
                tn = cm[0,0]
                # fp, fn, tp remain 0
            else: # Only AI
                tp = cm[0,0]
                # tn, fp, fn remain 0
    
    print(f"Accuracy: {acc:.4f}")
    print(f"FPR: {fpr:.4f}")
    print(f"FNR: {fnr:.4f}")
    
    return {
        "accuracy": acc,
        "fpr": fpr,
        "fnr": fnr
    }

def main():
    # 1. Load Data
    train_df, test_df = load_and_prep_data()
    
    # 2. Train Baseline
    base_vec, base_model = train_baseline(train_df)
    
    # 3. Train Improved
    imp_vec, imp_model = train_improved(train_df)
    
    # 4. Save Models
    print("\nSaving models...")
    with open('models/baseline_vectorizer.pkl', 'wb') as f: pickle.dump(base_vec, f)
    with open('models/baseline_model.pkl', 'wb') as f: pickle.dump(base_model, f)
    with open('models/improved_vectorizer.pkl', 'wb') as f: pickle.dump(imp_vec, f)
    with open('models/improved_model.pkl', 'wb') as f: pickle.dump(imp_model, f)
    
    # 5. Evaluate
    results = {}
    
    # Standard Test (Human + AI_RAW)
    std_mask = test_df['label_str'].isin(['HUMAN', 'AI_RAW'])
    std_test = test_df[std_mask]
    
    results['baseline_std'] = evaluate(base_vec, base_model, std_test, "Baseline on Standard Test")
    results['improved_std'] = evaluate(imp_vec, imp_model, std_test, "Improved on Standard Test")
    
    # Humanized Test
    hum_mask = test_df['label_str'] == 'AI_HUMANIZED'
    hum_test = test_df[hum_mask]
    
    results['baseline_hum'] = evaluate(base_vec, base_model, hum_test, "Baseline on Humanized Test")
    results['improved_hum'] = evaluate(imp_vec, imp_model, hum_test, "Improved on Humanized Test")
    
    # Save results
    with open('retrained_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to retrained_results.json")

if __name__ == "__main__":
    main()
