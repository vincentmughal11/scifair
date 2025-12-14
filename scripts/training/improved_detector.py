import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# Create models directory
os.makedirs('models', exist_ok=True)

def load_and_split_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Global Split: 70% Train, 30% Test (SAME as baseline)
    train_df, test_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42,  # SAME seed as baseline for fair comparison
        stratify=df['label']
    )
    
    print(f"Global Train Set: {len(train_df)} rows")
    print(f"Global Test Set: {len(test_df)} rows")
    
    return train_df, test_df

def train_model(train_df):
    print("Training on ALL categories (Human + AI_RAW + AI_HUMANIZED)...")
    print(f"Training Data: {len(train_df)} rows")
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['text'])
    
    # Create binary labels: 1 for ANY AI (AI_RAW or AI_HUMANIZED), 0 for HUMAN
    y_train = train_df['label'].apply(lambda x: 0 if x == 'HUMAN' else 1)
    
    print("Training Logistic Regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    return vectorizer, model

def evaluate_model(vectorizer, model, test_df, description):
    print(f"\n--- Evaluation: {description} ---")
    X_test = vectorizer.transform(test_df['text'])
    
    # Ground Truth: ANY AI (Raw or Humanized) = 1, Human = 0
    y_true = test_df['label'].apply(lambda x: 0 if x == 'HUMAN' else 1)
    
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")
    print("-" * 30)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'fnr': fnr
    }

def main():
    # 1. Load and Global Split (SAME as baseline)
    train_df, test_df = load_and_split_data('data/clean/final_dataset.csv')
    
    # 2. Train Model on ALL categories (NO filtering)
    vectorizer, model = train_model(train_df)
    
    # 3. Save Model
    with open('models/improved_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/improved_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to models/")
    
    # 4. Evaluate on different subsets
    
    # A. Standard Test (Human + AI_RAW)
    standard_test_mask = test_df['label'].isin(['HUMAN', 'AI_RAW'])
    standard_results = evaluate_model(vectorizer, model, test_df[standard_test_mask], "Standard Test Set (Human + AI_RAW)")
    
    # B. Humanized Test (AI_HUMANIZED)
    humanized_test_mask = test_df['label'] == 'AI_HUMANIZED'
    humanized_results = evaluate_model(vectorizer, model, test_df[humanized_test_mask], "Humanized Test Set (AI_HUMANIZED only)")
    
    # C. Full Test Set
    full_results = evaluate_model(vectorizer, model, test_df, "Full Global Test Set (All Categories)")

    # 5. Save results to JSON
    import json
    all_results = {
        "standard_test": standard_results,
        "humanized_test": humanized_results,
        "full_test": full_results
    }
    with open('improved_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("Results saved to improved_results.json")

if __name__ == "__main__":
    main()
