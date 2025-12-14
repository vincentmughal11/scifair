import re

import pandas as pd

# === 1. Load your dataset ===
# Replace this with the actual filename if different
df = pd.read_csv("./data/dataset.csv")

# === 2. Rename column if needed ===
if "generated" in df.columns:
    df = df.rename(columns={"generated": "label"})

# === 3. Filter to only AI-generated samples ===
df["label"] = df["label"].map({0.0: "HUMAN", 1.0: "AI_RAW"})
df = df[df["label"].str.upper() == "AI_RAW"].copy()


# === 4. Count words per entry ===
def word_count(text):
    return len(str(text).split())


df["word_count"] = df["text"].apply(word_count)

# === 5. Sort by shortest entries first ===
df = df.sort_values(by="word_count").reset_index(drop=True)

# === 6. Select as many entries as possible under 20,000 total words ===
selected = []
total_words = 0

for _, row in df.iterrows():
    if total_words + row["word_count"] > 20000:
        break
    selected.append(row)
    total_words += row["word_count"]

# === 7. Save result to new CSV ===
final_df = pd.DataFrame(selected)
final_df = final_df.drop(columns=["word_count"])  # optional

final_df.to_csv("ai_to_humanize_limited.csv", index=False)

print(f" Saved {len(final_df)} AI_RAW samples to 'ai_to_humanize_limited.csv'")
print(f"🧠 Total word count: {total_words}")
