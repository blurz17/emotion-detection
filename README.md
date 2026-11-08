# emotion-detection

Trigram Language Model implementation and text preprocessing pipeline for emotion analysis on the GoEmotions dataset.

## Project Structure
- `src/ng_model.py`: N-Gram language model with `comparison` and `ngram_model` calculation using the Markov assumption
- `src/data_loader.py`: Dataset loader supporting TSV/CSV splits and emotion labels
- `src/processing.py`: Text cleaning, negation-aware stopword removal, and lemmatization
- `src/negation_words.py`: Curated negation vocabulary preserved during stopword filtering
- `src/test.py`: Evaluates 3 sentences from the dataset with step-by-step Markov calculations
- `tests/test_all.py`: Unit test suite
- `data/`: GoEmotions dataset (`train.tsv`, `test.tsv`, `emotions.txt`)

## Usage
Run the pipeline to calculate probabilities for 3 dataset sentences:
```bash
python src/test.py
```

Run tests:
```bash
python -m unittest tests/test_all.py
```
