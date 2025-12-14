import csv
import os

clean_dataset_path = "./clean/clean_dataset.csv"
humanized_samples_path = "./clean/ai_humanized_samples.csv"
output_path = "./clean/final_dataset.csv"

def merge_datasets():
    rows = []
    
    # Read clean dataset
    if os.path.exists(clean_dataset_path):
        with open(clean_dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    'text': row['text'],
                    'label': row['label']
                })
        print(f"Loaded {len(rows)} rows from {clean_dataset_path}")
    else:
        print(f"Warning: {clean_dataset_path} not found.")

    # Read humanized samples
    humanized_count = 0
    if os.path.exists(humanized_samples_path):
        with open(humanized_samples_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('humanized_text'):
                    rows.append({
                        'text': row['humanized_text'],
                        'label': row['new_label']
                    })
                    humanized_count += 1
        print(f"Loaded {humanized_count} rows from {humanized_samples_path}")
    else:
        print(f"Warning: {humanized_samples_path} not found.")

    # Write combined dataset
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Successfully created {output_path} with {len(rows)} total rows.")

if __name__ == "__main__":
    merge_datasets()
