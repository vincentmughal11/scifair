"""
Train Improved Athena AI Detector with Humanized Samples
Combines Athena's original data + user's humanized samples
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import json

# Configuration
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """Load Athena's data + user's humanized samples"""
    print("Loading datasets...")
    
    # 1. Load Athena's human and AI data
    print("  Loading Athena's human samples...")
    human_df = pd.read_csv('../athena-source-main/data/human_train.csv')
    
    print("  Loading Athena's AI samples...")
    ai_df = pd.read_csv('../athena-source-main/data/AI_train_gpt_claude_gemini.csv')
    
    print(f"    Athena Human: {len(human_df):,}")
    print(f"    Athena AI: {len(ai_df):,}")
    
    # 2. Load user's humanized samples
    print("  Loading user's humanized samples...")
    hum_df = pd.read_csv('../data/Balanced_AI_Human_Humanized_UPDATED.csv')
    humanized_df = hum_df[hum_df['generated'] == 2].copy()
    
    print(f"    User Humanized: {len(humanized_df):,}")
    
    # 3. Create combined dataset
    human_texts = human_df['text'].tolist()
    ai_texts = ai_df['text'].tolist()
    humanized_texts = humanized_df['text'].tolist()
    
    # Labels: 0 = Human, 1 = AI (including humanized)
    texts = human_texts + ai_texts + humanized_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts) + [1] * len(humanized_texts)
    
    # Shuffle
    combined = list(zip(texts, labels))
    np.random.seed(42)
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    
    total = len(texts)
    human_count = len(human_texts)
    ai_count = len(ai_texts) + len(humanized_texts)
    
    print(f"\nCombined dataset:")
    print(f"  Total: {total:,} samples")
    print(f"  Human: {human_count:,} ({human_count/total*100:.1f}%)")
    print(f"  AI (raw + humanized): {ai_count:,} ({ai_count/total*100:.1f}%)")
    
    return list(texts), list(labels)

def compute_metrics(eval_pred):
    """Compute accuracy for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def main():
    print("\n" + "="*60)
    print("IMPROVED ATHENA - Training with Humanized Samples")
    print("="*60)
    
    # Load data
    texts, labels = load_data()
    
    # Train/test split
    print("\nSplitting data (80/20)...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    print(f"  Training samples: {len(train_texts):,}")
    print(f"  Test samples: {len(test_texts):,}")
    
    # Load tokenizer and create datasets
    print("\nPreparing datasets...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    
    # Load model
    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    # Training arguments
    save_dir = '../models/athena_improved'
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{save_dir}/logs',
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
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
    print(f"\nTraining for {EPOCHS} epochs...")
    print("Estimated time: ~4 hours")
    trainer.train()
    
    # Evaluate
    print("\nFinal evaluation...")
    results = trainer.evaluate()
    test_accuracy = results['eval_accuracy']
    
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {test_accuracy:.2%}")
    print(f"{'='*60}")
    
    # Save final model
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModel saved to {save_dir}")
    
    # Save results
    results_dict = {
        'test_accuracy': float(test_accuracy),
        'data_source': 'athena_original + user_humanized',
        'train_samples': len(train_texts),
        'test_samples': len(test_texts),
        'humanized_samples': 1045
    }
    
    with open('../results/athena_improved_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Baseline Athena:  98.82%")
    print(f"Improved Athena:  {test_accuracy:.2%}")
    print("\nTest the improved model with: python test_athena_improved.py")

if __name__ == "__main__":
    main()
