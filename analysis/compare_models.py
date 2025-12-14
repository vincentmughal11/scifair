import json

# Load results
with open('baseline_results.json', 'r') as f:
    baseline = json.load(f)

with open('improved_results.json', 'r') as f:
    improved = json.load(f)

# Create comparison table
print("=" * 80)
print("BASELINE vs IMPROVED DETECTOR COMPARISON")
print("=" * 80)

categories = [
    ("Standard Test (Human + AI_RAW)", "standard_test"),
    ("Humanized Test (AI_HUMANIZED)", "humanized_test"),
    ("Full Test Set", "full_test")
]

for title, key in categories:
    print(f"\n{title}")
    print("-" * 80)
    
    b = baseline[key]
    i = improved[key]
    
    print(f"{'Metric':<12} {'Baseline':<12} {'Improved':<12} {'Difference'}")
    print("-" * 80)
    
    metrics = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('FPR', 'fpr'),
        ('FNR', 'fnr')
    ]
    
    for name, metric in metrics:
        b_val = b[metric]
        i_val = i[metric]
        diff = i_val - b_val
        print(f"{name:<12} {b_val:<12.4f} {i_val:<12.4f} {diff:+.4f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

std_diff = improved['standard_test']['accuracy'] - baseline['standard_test']['accuracy']
hum_diff = improved['humanized_test']['accuracy'] - baseline['humanized_test']['accuracy']
fpr_diff = improved['standard_test']['fpr'] - baseline['standard_test']['fpr']

print(f"\nStandard Test Accuracy: {std_diff:+.2%}")
print(f"Humanized Test Accuracy: {hum_diff:+.2%}")
print(f"False Positive Rate Change: {fpr_diff:+.2%}")

# Save to file
with open('comparison_summary.txt', 'w') as f:
    f.write("COMPARISON SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Standard Test:\n")
    f.write(f"  Baseline:  {baseline['standard_test']['accuracy']:.4f}\n")
    f.write(f"  Improved:  {improved['standard_test']['accuracy']:.4f}\n")
    f.write(f"  Change:    {std_diff:+.4f}\n\n")
    f.write(f"Humanized Test:\n")
    f.write(f"  Baseline:  {baseline['humanized_test']['accuracy']:.4f}\n")
    f.write(f"  Improved:  {improved['humanized_test']['accuracy']:.4f}\n")
    f.write(f"  Change:    {hum_diff:+.4f}\n")

print("\nComparison saved to comparison_summary.txt")
