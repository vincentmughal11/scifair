"""
Analyze humanized dataset quality
Checks average text length and compares to Athena's training specs
"""

import pandas as pd
import numpy as np

def analyze_dataset(file_path, label_col='generated', label_value=2):
    """Analyze text length in dataset"""
    print(f"Loading dataset: {file_path}")
    df = pd.read_csv(file_path)
    
    # Filter for humanized samples
    if label_col in df.columns:
        humanized = df[df[label_col] == label_value]
    else:
        humanized = df
    
    print(f"\nTotal humanized samples: {len(humanized)}")
    
    # Calculate word counts
    word_counts = humanized['text'].apply(lambda x: len(str(x).split()))
    
    # Statistics
    avg_words = word_counts.mean()
    median_words = word_counts.median()
    min_words = word_counts.min()
    max_words = word_counts.max()
    std_words = word_counts.std()
    
    print("\n" + "="*60)
    print("TEXT LENGTH ANALYSIS")
    print("="*60)
    print(f"Average words:  {avg_words:.1f}")
    print(f"Median words:   {median_words:.1f}")
    print(f"Min words:      {min_words}")
    print(f"Max words:      {max_words}")
    print(f"Std deviation:  {std_words:.1f}")
    
    # Distribution
    print("\nDistribution:")
    ranges = [(0, 50), (50, 100), (100, 200), (200, 400), (400, float('inf'))]
    for low, high in ranges:
        count = ((word_counts >= low) & (word_counts < high)).sum()
        pct = count / len(word_counts) * 100
        range_str = f"{low}-{high if high != float('inf') else '∞'} words"
        print(f"  {range_str:15} {count:5} samples ({pct:5.1f}%)")
    
    # Quality assessment
    print("\n" + "="*60)
    print("QUALITY ASSESSMENT")
    print("="*60)
    
    athena_range = (100, 400)
    in_range = ((word_counts >= athena_range[0]) & (word_counts <= athena_range[1])).sum()
    in_range_pct = in_range / len(word_counts) * 100
    
    print(f"\nAthena's training data: {athena_range[0]}-{athena_range[1]} words")
    print(f"Your data in this range: {in_range}/{len(word_counts)} ({in_range_pct:.1f}%)")
    
    if in_range_pct >= 70:
        print("\n GOOD: Most samples match Athena's expected length")
    elif in_range_pct >= 40:
        print("\n  MODERATE: Some samples are too short/long")
        print("   Consider filtering to 100-400 word range")
    else:
        print("\n POOR: Most samples are outside Athena's training range")
        print("   Recommendation: Generate new humanized samples")
        print("   Target: 100-400 words per sample")
    
    if avg_words < 100:
        print("\n  Warning: Very short samples may give unreliable results")
    elif avg_words > 400:
        print("\n  Warning: Very long samples - consider chunking")
    
    return word_counts

def main():
    # Analyze the humanized dataset
    print("\n" + "="*60)
    print("HUMANIZED DATASET QUALITY CHECK")
    print("="*60)
    
    file_path = '../data/Balanced_AI_Human_Humanized_UPDATED.csv'
    
    try:
        word_counts = analyze_dataset(file_path, label_col='generated', label_value=2)
        
        # Save statistics
        with open('humanized_analysis.txt', 'w') as f:
            f.write("Humanized Dataset Analysis\n")
            f.write("="*60 + "\n")
            f.write(f"Average words: {word_counts.mean():.1f}\n")
            f.write(f"Median words: {word_counts.median():.1f}\n")
            f.write(f"Samples in 100-400 range: {((word_counts >= 100) & (word_counts <= 400)).sum()}\n")
        
        print("\nAnalysis saved to humanized_analysis.txt")
        
    except FileNotFoundError:
        print(f"\n Error: Could not find {file_path}")
        print("Make sure the file exists at that location")
    except Exception as e:
        print(f"\n Error: {e}")

if __name__ == "__main__":
    main()
