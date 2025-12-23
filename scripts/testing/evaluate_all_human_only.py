import pandas as pd
import numpy as np
import pickle
import torch
import os
import re
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import tqdm

def load_data():
    """Load and split data to get the exact test set human samples"""
    print("Loading and splitting dataset...")
    full_df = pd.read_csv('../data/dataset.csv')
    
    # Reproduce logic to get same Human samples as before
    human_samples = []
    chunk_size = 50000
    target_count = 5000 
    
    for chunk in pd.read_csv('../data/dataset.csv', chunksize=chunk_size):
        chunk_humans = chunk[chunk['label'] == 0.0]
        human_samples.append(chunk_humans[['text', 'label']])
        if sum(len(x) for x in human_samples) >= target_count:
            break
            
    human_df = pd.concat(human_samples).head(target_count).copy()
    human_df['label_str'] = 'HUMAN'
    
    ai_samples = []
    for chunk in pd.read_csv('../data/dataset.csv', chunksize=chunk_size):
        chunk_ai = chunk[chunk['label'] == 1.0]
        ai_samples.append(chunk_ai[['text', 'label']])
        if sum(len(x) for x in ai_samples) >= target_count:
            break
    
    ai_df = pd.concat(ai_samples).head(target_count).copy()
    ai_df['label_str'] = 'AI_RAW'
    
    full_df = pd.concat([human_df, ai_df], ignore_index=True)
    
    train_df, test_df = train_test_split(
        full_df,
        test_size=0.3,
        random_state=42,
        stratify=full_df['label_str']
    )
    
    human_test_df = test_df[test_df['label_str'] == 'HUMAN'].copy()
    print(f"Human Test Samples: {len(human_test_df)}")
    return human_test_df['text'].tolist()

def eval_sklearn(name, texts, vec_path, model_path):
    print(f"Evaluating {name}...")
    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        print(f"  Missing files for {name}")
        return None
        
    with open(vec_path, 'rb') as f:
        vec = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    X = vec.transform(texts)
    preds = model.predict(X)
    
    correct = (preds == 0).sum()
    acc = correct / len(texts)
    print(f"  {name}: {acc:.2%}")
    return acc

def get_structure_features(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sent_len_sd = 0
    else:
        sent_lengths = [len(s.split()) for s in sentences]
        sent_len_sd = np.std(sent_lengths) if sent_lengths else 0
    
    words = re.findall(r'\w+', text.lower())
    if not words:
        type_token_ratio = 0
        punct_density = 0
        avg_word_len = 0
    else:
        type_token_ratio = len(set(words)) / len(words)
        punct_count = len(re.findall(r'[.,!?;:]', text))
        punct_density = punct_count / len(words)
        avg_word_len = sum(len(w) for w in words) / len(words)
        
    return [sent_len_sd, type_token_ratio, punct_density, avg_word_len]

def eval_structure(name, texts, scaler_path, model_path):
    print(f"Evaluating {name}...")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print(f"  Missing files for {name}")
        return None
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    features_list = []
    print("  Extracting features...")
    for text in tqdm.tqdm(texts):
        features_list.append(get_structure_features(text))
        
    X = scaler.transform(features_list)
    preds = model.predict(X)
    
    correct = (preds == 0).sum()
    acc = correct / len(texts)
    print(f"  {name}: {acc:.2%}")
    return acc

def eval_hybrid(name, texts, tfidf_path, scaler_path, model_path):
    print(f"Evaluating {name}...")
    if not os.path.exists(tfidf_path) or not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print(f"  Missing files for {name}")
        return None

    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print("  Extracting TF-IDF features...")
    # TFIDF might run out of memory if we do all at once, but 1500 samples should be fine.
    # Note: tfidf.transform returns sparse matrix.
    tfidf_features = tfidf.transform(texts)
    
    # Structure features
    print("  Extracting structure features...")
    struct_features_list = []
    for text in tqdm.tqdm(texts):
        struct_features_list.append(get_structure_features(text))
    
    struct_features_scaled = scaler.transform(struct_features_list)
    
    # Combine (vstack sparse + dense? sklearn supports it, but hstack needs standard array or sparse)
    # tfidf is sparse. struct is dense.
    # If we convert struct to sparse, we can hstack. Or convert tfidf to dense (might be big).
    # 5000 features x 1500 samples is small enough for dense.
    
    import scipy.sparse
    # Convert struct to sparse
    struct_sparse = scipy.sparse.csr_matrix(struct_features_scaled)
    # Hstack
    combined_features = scipy.sparse.hstack([tfidf_features, struct_sparse])
    
    # Model might expect dense if it was trained on dense numpy arrays.
    # Test script: "np.hstack([tfidf_features, struct_features_scaled])" -> implies Dense array.
    # "tfidf_features = tfidf.transform([text]).toarray()" -> dense.
    
    combined_dense = combined_features.toarray()
    
    preds = model.predict(combined_dense)
    correct = (preds == 0).sum()
    acc = correct / len(texts)
    print(f"  {name}: {acc:.2%}")
    return acc

def eval_perplexity_single(name, texts, scaler_path, model_path, gpt_model, tokenizer):
    print(f"Evaluating {name}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    features_list = []
    print("  Calculating perplexity...")
    
    # Only do first 100 to save time? User wants "data". 
    # GPT-2 inference is slow on CPU. 1500 samples might take a while.
    # Let's try to do it all but give progress.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt_model.to(device)
    
    for text in tqdm.tqdm(texts):
        try:
            encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in encodings.items()}
            with torch.no_grad():
                outputs = gpt_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                ppl = torch.exp(loss).item()
        except:
            ppl = 1000.0
        features_list.append([ppl])
        
    X = scaler.transform(features_list)
    preds = model.predict(X)
    correct = (preds == 0).sum()
    acc = correct / len(texts)
    print(f"  {name}: {acc:.2%}")
    return acc

def calculate_enhanced_ppl_features(text, model, tokenizer, device):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
    
    if not sentences:
        return [1000.0, 0.0, 1000.0, 1000.0]
    
    sentence_ppls = []
    # Limit sentences to speed up? Script uses [:10].
    for sent in sentences[:10]:
        try:
            encodings = tokenizer(sent, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in encodings.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                ppl = torch.exp(outputs.loss).item()
            sentence_ppls.append(ppl)
        except:
            pass
            
    if not sentence_ppls:
        return [1000.0, 0.0, 1000.0, 1000.0]
    
    return [np.mean(sentence_ppls), np.std(sentence_ppls), np.max(sentence_ppls), np.min(sentence_ppls)]

def eval_perplexity_enhanced(name, texts, scaler_path, model_path, gpt_model, tokenizer):
    print(f"Evaluating {name}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt_model.to(device)
    
    features_list = []
    print("  Calculating enhanced features...")
    for text in tqdm.tqdm(texts):
        feat = calculate_enhanced_ppl_features(text, gpt_model, tokenizer, device)
        features_list.append(feat)
        
    X = scaler.transform(features_list)
    preds = model.predict(X)
    correct = (preds == 0).sum()
    acc = correct / len(texts)
    print(f"  {name}: {acc:.2%}")
    return acc

def eval_transformer(name, texts, model_dir):
    print(f"Evaluating {name}...")
    if not os.path.exists(model_dir):
        print(f"  Missing model dir: {model_dir}")
        return None
        
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    except:
        return None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    batch_size = 16
    correct = 0
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == 0).sum().item() # 0 is Human
            
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i+batch_size, total)}/{total}...", end='\r')
            
    acc = correct / total
    print(f"  {name}: {acc:.2%}")
    return acc

def main():
    texts = load_data()
    results = {}
    
    # Models to eval
    
    # 1. Scikit-learn simple
    results['TF-IDF Baseline'] = eval_sklearn('TF-IDF Baseline', texts, '../models/baseline_vectorizer.pkl', '../models/baseline_model.pkl')
    results['TF-IDF Improved'] = eval_sklearn('TF-IDF Improved', texts, '../models/improved_vectorizer.pkl', '../models/improved_model.pkl')
    
    results['Structure Baseline'] = eval_structure('Structure Baseline', texts, '../models/structure_scaler.pkl', '../models/structure_baseline_model.pkl')
    results['Structure Improved'] = eval_structure('Structure Improved', texts, '../models/structure_scaler.pkl', '../models/structure_improved_model.pkl')
    
    # 2. Hybrid
    results['Hybrid Baseline'] = eval_hybrid('Hybrid Baseline', texts, '../models/hybrid_tfidf.pkl', '../models/hybrid_struct_scaler.pkl', '../models/hybrid_baseline_model.pkl')
    results['Hybrid Improved'] = eval_hybrid('Hybrid Improved', texts, '../models/hybrid_tfidf.pkl', '../models/hybrid_struct_scaler.pkl', '../models/hybrid_improved_model.pkl')
    
    # 3. Perplexity (Load GPT2 once)
    try:
        print("Loading GPT-2 for Perplexity models...")
        gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt_model.eval()
        
        results['Perplexity (Single) Baseline'] = eval_perplexity_single('Perplexity (Single) Baseline', texts, '../models/perplexity_pilot_scaler.pkl', '../models/perplexity_pilot_baseline_model.pkl', gpt_model, gpt_tokenizer)
        results['Perplexity (Single) Improved'] = eval_perplexity_single('Perplexity (Single) Improved', texts, '../models/perplexity_pilot_scaler.pkl', '../models/perplexity_pilot_improved_model.pkl', gpt_model, gpt_tokenizer)
        
        results['Perplexity (Enhanced) Baseline'] = eval_perplexity_enhanced('Perplexity (Enhanced) Baseline', texts, '../models/perplexity_enhanced_full_scaler.pkl', '../models/perplexity_enhanced_full_baseline_model.pkl', gpt_model, gpt_tokenizer)
        results['Perplexity (Enhanced) Improved'] = eval_perplexity_enhanced('Perplexity (Enhanced) Improved', texts, '../models/perplexity_enhanced_full_scaler.pkl', '../models/perplexity_enhanced_full_improved_model.pkl', gpt_model, gpt_tokenizer)
    except Exception as e:
        print(f"Skipping perplexity due to error: {e}")

    # 4. Transformers
    results['Transformer Baseline'] = eval_transformer('Transformer Baseline', texts, '../models/transformer_baseline_quick')
    results['Transformer Improved'] = eval_transformer('Transformer Improved', texts, '../models/transformer_improved_quick')
    
    results['Athena Baseline'] = eval_transformer('Athena Baseline', texts, '../models/athena_athena_data')
    results['Athena Improved'] = eval_transformer('Athena Improved', texts, '../models/athena_improved')
    
    # Save CSV
    print("\nSAVING RESULTS...")
    with open('../results/model_human_accuracies.csv', 'w') as f:
        f.write("Model,Baseline Human Accuracy,Improved Human Accuracy\n")
        
        model_names = [
            'TF-IDF', 'Structure', 'Hybrid', 
            'Perplexity (Single)', 'Perplexity (Enhanced)', 
            'Transformer', 'Athena'
        ]
        
        for m in model_names:
            base = results.get(f"{m} Baseline", "N/A")
            imp = results.get(f"{m} Improved", "N/A")
            
            base_str = f"{base:.2%}" if isinstance(base, float) else "N/A"
            imp_str = f"{imp:.2%}" if isinstance(imp, float) else "N/A"
            
            f.write(f"{m},{base_str},{imp_str}\n")
            
    print("Done. Saved to ../results/model_human_accuracies.csv")

if __name__ == "__main__":
    main()
