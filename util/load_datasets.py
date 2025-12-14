import re

import pandas as pd


# Step 1 – Count sentences in text
def count_sentences(text):
    sentences = re.split(r"[.!?]+(?:\s|$)", text.strip())
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)


# Step 2 – Load the dataset
df = pd.read_csv("./data/dataset.csv")  # Replace with your path
df = df.rename(columns={"generated": "label"})  # If needed
df["label"] = df["label"].map({0.0: "HUMAN", 1.0: "AI_RAW"})

# Step 3 – Add sentence count and filter
df["sentence_count"] = df["text"].apply(count_sentences)
filtered_df = df[(df["sentence_count"] >= 3) & (df["sentence_count"] <= 7)]

# Step 4 – Sample 1,000 per class
human_df = filtered_df[filtered_df["label"] == "HUMAN"].sample(n=1000, random_state=42)
ai_df = filtered_df[filtered_df["label"] == "AI_RAW"].sample(n=1000, random_state=42)

# Step 5 – Combine and shuffle final dataset
combined_df = pd.concat([human_df, ai_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
combined_df = combined_df.drop(columns=["sentence_count"])

# Step 6 – Save clean full dataset
combined_df.to_csv("clean_dataset.csv", index=False)

# Step 7 – Create a 200-sample file for humanization
ai_humanize_df = ai_df.sample(n=200, random_state=1337).reset_index(drop=True)
ai_humanize_df = ai_humanize_df.drop(columns=["sentence_count"])
ai_humanize_df.to_csv("ai_to_humanize.csv", index=False)

print(" Datasets saved: 'clean_dataset.csv' and 'ai_to_humanize.csv'")
