"""
Athena-style AI Detector Training Script (PyTorch Version)
Adapted from: https://medium.com/@tomrodolfolee/building-an-ai-detector-from-scratch-part-i-db72bc2bdadb

Trains DistilBERT on user's 373k sample database using PyTorch
Falls back to Athena's original data if accuracy is poor
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
USE_USER_DATA = False  # Set to False to use Athena's original data
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

def load_user_data():
    """Load user's 373k sample database"""
    print("Loading user's dataset...")
    
    # Load the large dataset
    print("  Reading dataset.csv...")
    df = pd.read_csv('../data/dataset.csv')
    
    # The dataset has label column: 0=Human, 1=AI
    human_df = df[df['label'] == 0].copy()
    ai_df = df[df['label'] == 1].copy()
    
    print(f"  Human samples: {len(human_df):,}")
    print(f"  AI samples: {len(ai_df):,}")
    
    # Balance the dataset (use min of both)
    min_count = min(len(human_df), len(ai_df))
    print(f"  Balancing to: {min_count:,} each")
    
    human_df = human_df.sample(n=min_count, random_state=42)
    ai_df = ai_df.sample(n=min_count, random_state=42)
    
    # Combine and create labels
    human_df['label_int'] = 0
    ai_df['label_int'] = 1
    
    combined = pd.concat([human_df[['text', 'label_int']], ai_df[['text', 'label_int']]], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"Total balanced samples: {len(combined):,}")
    return combined['text'].tolist(), combined['label_int'].tolist()

def load_athena_data():
    """Load Athena's original training data"""
    print("Loading Athena's original dataset...")
    
    # Load human and AI data
    human_df = pd.read_csv('../athena-source-main/data/human_train.csv')
    ai_df = pd.read_csv('../athena-source-main/data/AI_train_gpt_claude_gemini.csv')
    
    print(f"  Human samples: {len(human_df):,}")
    print(f"  AI samples: {len(ai_df):,}")
    
    # Create labels
    human_texts = human_df['text'].tolist()
    ai_texts = ai_df['text'].tolist()
    
    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)
    
    # Shuffle
    combined = list(zip(texts, labels))
    np.random.seed(42)
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    
    print(f"Total samples: {len(texts):,}")
    return list(texts), list(labels)

def compute_metrics(eval_pred):
    """Compute accuracy for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def main():
    print("\n" + "="*60)
    print("ATHENA AI DETECTOR - Training Script (PyTorch)")
    print("="*60)
    
    # Load data
    if USE_USER_DATA:
        texts, labels = load_user_data()
        save_name = 'user_data'
    else:
        texts, labels = load_athena_data()
        save_name = 'athena_data'
    
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
    save_dir = f'../models/athena_{save_name}'
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
    print("This will take 1-2 hours for 373k samples...")
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
        'data_source': 'user_373k' if USE_USER_DATA else 'athena_original',
        'train_samples': len(train_texts),
        'test_samples': len(test_texts)
    }
    
    with open(f'../results/athena_{save_name}_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print("\nTraining complete!")
    if test_accuracy < 0.90:
        print(f"\n  Accuracy is below 90% ({test_accuracy:.1%})")
        print(f"Consider setting USE_USER_DATA=False to train on Athena's data")
    else:
        print(f"\n Great accuracy! ({test_accuracy:.1%})")
        print(f"Test the model with: python test_athena.py")

if __name__ == "__main__":
    main()
