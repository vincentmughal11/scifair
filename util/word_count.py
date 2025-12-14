import pandas as pd

# Load your dataset
df = pd.read_csv("./clean/ai_to_humanize.csv")  # Replace with your actual file path


# Function to count words in a single entry
def word_count(text):
    return len(str(text).split())


# Apply the function to each row in the 'text' column
df["word_count"] = df["text"].apply(word_count)

# Total word count
total_words = df["word_count"].sum()

# Preview result
print(f" Total word count across all entries: {total_words:,}")
print(f" Average words per entry: {df['word_count'].mean():.2f}")
print(f" Number of entries: {len(df)}")
