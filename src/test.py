from data_loader import load_data, load_emotions
from ng_model import ngram_model
from processing import process_text

print("=" * 60)
print("1. LOADING & PREPROCESSING DATA")
print("=" * 60)

raw_train_data = load_data("train")
print(f"Loaded {len(raw_train_data)} training sentences.")

# Preprocess training data
data_set = raw_train_data["text"].apply(process_text).tolist()

# Load test data and select 3 sentences
test_data = load_data("test")
emotions = load_emotions()

selected_raw_sentences = [test_data.iloc[i]["text"] for i in range(3)]
sentences = [process_text(text) for text in selected_raw_sentences]

print("\n" + "=" * 60)
print("2. THREE SELECTED SENTENCES FROM DATASET")
print("=" * 60)
for idx, (raw, tokens) in enumerate(zip(selected_raw_sentences, sentences), start=1):
    label = test_data.iloc[idx - 1]["labels"]
    label_name = emotions[int(label)] if str(label).isdigit() and int(label) < len(emotions) else label
    print(f"\nSentence {idx}:")
    print(f"  Raw:    \"{raw.strip()}\"")
    print(f"  Tokens: {tokens}")
    print(f"  Label:  {label_name}")

print("\n" + "=" * 60)
print("3. N-GRAM MARKOV CALCULATION STEPS (WITH LAPLACE SMOOTHING)")
print("=" * 60)

probabilities = ngram_model(sentences, data_set)

print("\n" + "=" * 60)
print("4. FINAL SENTENCE PROBABILITIES")
print("=" * 60)
for sentence_num, prob in probabilities.items():
    print(f"Sentence {sentence_num} Probability: {prob:.6e}")
