
import os
import random
import pandas as pd
from nltk.corpus import wordnet
from nltk import download as nltk_download

# Download necessary resources
nltk_download('wordnet')
nltk_download('omw-1.4')

# Configuration
DATASET_FOLDER = "text_dataset"
AUGMENTED_FOLDER = "augmented_dataset"
TEXT_COLUMN = "text"
AUGMENTATION_COUNT = 3

# Ensure output directory exists
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

# Get synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ").lower()
            if w != word:
                synonyms.add(w)
    return list(synonyms)

# Augmentation methods
def augment_text(text):
    words = text.split()
    aug_type = random.choice(['synonym', 'deletion', 'swap', 'insertion'])
    changed_words = ""

    if aug_type == 'synonym':
        new_words = words.copy()
        candidates = [w for w in words if w.isalpha()]
        random.shuffle(candidates)
        for word in candidates:
            synonyms = get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if w == word else w for w in new_words]
                changed_words = f"{word} -> {synonym}"
                break
        result = new_words

    elif aug_type == 'deletion':
        deleted = [w for w in words if random.uniform(0, 1) <= 0.1]
        result = [w for w in words if w not in deleted]
        changed_words = f"deleted: {', '.join(deleted)}" if deleted else "none"

    elif aug_type == 'swap':
        result = words.copy()
        if len(result) >= 2:
            idx1, idx2 = random.sample(range(len(result)), 2)
            result[idx1], result[idx2] = result[idx2], result[idx1]
            changed_words = f"{words[idx1]} <-> {words[idx2]}"
        else:
            changed_words = "none"

    else:  # insertion
        result = words.copy()
        inserted = ""
        for _ in range(10):
            candidate = random.choice(words)
            synonyms = get_synonyms(candidate)
            if synonyms:
                inserted = random.choice(synonyms)
                pos = random.randint(0, len(result))
                result.insert(pos, inserted)
                changed_words = f"inserted: {inserted}"
                break
        if not inserted:
            changed_words = "none"

    return " ".join(result), aug_type, changed_words

# Main CSV processing function
def process_csv(file_path):
    df = pd.read_csv(file_path)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in CSV.")

    all_rows = []
    log_rows = []

    for i, row in df.iterrows():
        original_text = str(row[TEXT_COLUMN])
        label = row.get('label', '')  # optional label

        for j in range(AUGMENTATION_COUNT):
            augmented_text, aug_type, changed_words = augment_text(original_text)
            all_rows.append({
                "original_text": original_text,
                "augmented_text": augmented_text,
                "augmentation_type": aug_type,
                "short_description_of_changed_words": changed_words,
                "label": label
            })

            log_rows.append({
                "row_index": i,
                "original_text": original_text,
                "augmented_text": augmented_text,
                "augmentation_type": aug_type,
                "short_description_of_changed_words": changed_words
            })

    # Save augmented dataset
    output_file = os.path.join(AUGMENTED_FOLDER, "augmented_dataset.csv")
    pd.DataFrame(all_rows).to_csv(output_file, index=False, encoding="utf-8")

    # Save structured log
    log_file = os.path.join(AUGMENTED_FOLDER, "augmentation_log.csv")
    pd.DataFrame(log_rows).to_csv(log_file, index=False, encoding="utf-8")

    print(f"✅ Augmented data saved to: {output_file}")
    print(f"📝 Log saved to: {log_file}")

# Entry point
def process_dataset():
    files = os.listdir(DATASET_FOLDER)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        print("⚠️ No CSV files found in 'text_dataset/'")
        return

    for csv_file in csv_files:
        print(f"🔄 Processing {csv_file}...")
        process_csv(os.path.join(DATASET_FOLDER, csv_file))

if __name__ == "__main__":
    process_dataset()
