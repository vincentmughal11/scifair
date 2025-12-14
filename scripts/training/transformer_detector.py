import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

os.makedirs('../models', exist_ok=True)

def load_and_prep_data(quick_test=False):
    """Load and prepare dataset"""
    print("Loading datasets...")
    
    # Load humanized data
    hum_df = pd.read_csv('../data/Balanced_AI_Human_Humanized_UPDATED.csv')
    humanized_samples = hum_df[hum_df['generated'] == 2].copy()
    humanized_samples = humanized_samples[['text']].copy()
    humanized_samples['label_str'] = 'AI_HUMANIZED'
    
    # Load human and AI data
    if quick_test:
        target_count = 500  # Quick test with small data
        print("QUICK TEST MODE: Using 500 samples per category")
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
    
    # Convert labels to binary
    full_df['label'] = full_df['label_str'].apply(lambda x: 0 if x == 'HUMAN' else 1)
    
    return full_df

class AIDetectionDataset(torch.utils.data.Dataset):
    """Custom dataset for transformer training"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    """Compute accuracy for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def train_model(train_df, test_df, model_name='distilbert', quick_test=False):
    """Fine-tune DistilBERT for AI detection"""
    print(f"\n=== TRAINING {model_name.upper()} MODEL ===")
    
    # Initialize tokenizer and model
    print("Loading DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    # Prepare datasets
    print("Tokenizing data...")
    train_dataset = AIDetectionDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )
    
    test_dataset = AIDetectionDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer
    )
    
    # Training arguments
    epochs = 1 if quick_test else 3
    output_dir = f'../models/transformer_{model_name}_{"quick" if quick_test else "full"}'
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print(f"Training for {epochs} epoch(s)...")
    trainer.train()
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Evaluate
    print("\nEvaluating...")
    results = trainer.evaluate()
    print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
    
    return results

def main():
    import sys
    
    # Check if quick test mode
    quick_test = '--quick' in sys.argv
    
    # Load data
    full_df = load_and_prep_data(quick_test=quick_test)
    
    # Split
    train_df, test_df = train_test_split(
        full_df,
        test_size=0.3,
        random_state=42,
        stratify=full_df['label_str']
    )
    
    print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
    
    # Train baseline (Human + AI_RAW only)
    print("\n" + "="*60)
    print("BASELINE MODEL (Human + AI_RAW only)")
    print("="*60)
    baseline_train = train_df[train_df['label_str'].isin(['HUMAN', 'AI_RAW'])]
    baseline_test = test_df[test_df['label_str'].isin(['HUMAN', 'AI_RAW'])]
    
    baseline_results = train_model(baseline_train, baseline_test, 'baseline', quick_test)
    
    # Train improved (all categories)
    print("\n" + "="*60)
    print("IMPROVED MODEL (All categories)")
    print("="*60)
    
    improved_results = train_model(train_df, test_df, 'improved', quick_test)
    
    # Save results
    import json
    mode = 'quick' if quick_test else 'full'
    results_dict = {
        'baseline_accuracy': baseline_results['eval_accuracy'],
        'improved_accuracy': improved_results['eval_accuracy']
    }
    
    with open(f'../results/transformer_{mode}_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Baseline Accuracy: {baseline_results['eval_accuracy']:.2%}")
    print(f"Improved Accuracy: {improved_results['eval_accuracy']:.2%}")
    
    if quick_test:
        print("\nQuick test complete! To run full training:")
        print("  python transformer_detector.py")

if __name__ == "__main__":
    main()
