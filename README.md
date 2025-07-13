## 🍽️ Text Data Augmentation Script for Restaurant Conversations (NLP)

This project provides a complete Python solution to **automatically augment textual datasets**, making them more robust for training NLP models. It supports CSV datasets and generates additional training samples using synonym replacement, word swapping, deletions, and insertions.

Ideal for applications like:
- Chatbots for restaurants
- Sentiment analysis
- Text classification
- Language modeling

---

## 📂 Project Structure

├── augment_text_dataset.py # 🧠 Main augmentation script
├── text_dataset/ # 📥 Folder for original datasets
│ └── text_dataset.csv # (e.g., customer conversations)
├── augmented_dataset/ # 📤 Outputs written here
│ ├── augmented_dataset.csv # Original + augmented side-by-side
│ └── augmentation_log.csv # Structured log of each change
├── README.md # 📘 Project documentation


---

## 🔧 Features

✅ Handles any `.csv` file in the `text_dataset/` folder  
✅ Augments each row **multiple times** using:
- **Synonym replacement**
- **Random word deletion**
- **Random word swap**
- **Word insertion via synonym**

✅ Generates:
- **`augmented_dataset.csv`** with:
  - `original_text`
  - `augmented_text`
  - `augmentation_type`
  - `short_description_of_changed_words`
  - `label` (if provided)

- **`augmentation_log.csv`** with all augmentation steps for auditing

✅ UTF-8 and Windows-compatible

---

## 📦 Installation

Install required Python libraries:

```bash
pip install pandas nltk

python augment_text_dataset.py
