from data_loader import load_data, load_emotions
from ng_model import TrigramLanguageModel
from processing import process_text

print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

emotions = load_emotions()
print(f"Loaded {len(emotions)} emotion categories.")

raw_train_data = load_data("train")
print(f"Train samples count: {len(raw_train_data)}")
print("\nFirst 3 training samples (raw):")
print(raw_train_data[["text", "labels"]].head(3))

print("\n" + "=" * 60)
print("2. PREPROCESSING DATA")
print("=" * 60)

cleaned_train_data = raw_train_data.copy()
cleaned_train_data["tokens"] = cleaned_train_data["text"].apply(process_text)

print("\nFirst 3 training samples (cleaned & tokenized):")
print(cleaned_train_data[["text", "tokens"]].head(3))

print("\n" + "=" * 60)
print("3. TRAINING TRIGRAM LANGUAGE MODEL")
print("=" * 60)

model = TrigramLanguageModel()
model.fit(cleaned_train_data["tokens"].tolist())
print(f"Model trained! Vocabulary size: {len(model.vocabulary)}")
print(f"Total bigrams: {len(model.bigram_counts)}")
print(f"Total trigrams: {len(model.trigram_counts)}")

print("\n" + "=" * 60)
print("4. EVALUATING ON TEST SAMPLES")
print("=" * 60)

test_data = load_data("test")
sample_text = test_data.iloc[0]["text"]
sample_label = test_data.iloc[0]["labels"]

# Tokenize test sentence with process_text
sample_tokens = process_text(sample_text)

print(f"Raw sentence:      {sample_text}")
print(f"Processed tokens:  {sample_tokens}")
if str(sample_label).isdigit() and int(sample_label) < len(emotions):
    print(f"Emotion label:     {sample_label} ({emotions[int(sample_label)]})")

log_probability, steps = model.calculate_sentence_probability(sample_tokens)
perplexity = model.calculate_perplexity(sample_tokens)

print(f"\nLog Probability:   {log_probability:.4f}")
print(f"Perplexity:        {perplexity:.4f}")

print("\nTrigram evaluation steps:")
for step in steps:
    ctx = " ".join(step["context"])
    print(f"  P({step['word']:<12} | [{ctx:<16}]) = {step['probability']:.6f}")

print("\n" + "=" * 60)
print("ALL CHECKS COMPLETED SUCCESSFULLY")
print("=" * 60)
