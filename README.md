# emotion-detection

Trigram Language Model with Laplace (add-1) smoothing and text preprocessing pipeline for emotion analysis on the GoEmotions dataset.

## Project Structure
- `src/data_loader.py`: Dataset loader supporting TSV/CSV splits and emotion labels
- `src/processing.py`: Text cleaning, negation-aware stopword removal, and lemmatization
- `src/negation_words.py`: Curated negation vocabulary preserved during stopword filtering
- `src/ng_model.py`: Trigram Language Model with smoothing, sentence probability, and perplexity
- `src/test.py`: End-to-end evaluation script
- `tests/test_all.py`: Unit test suite
- `data/`: GoEmotions dataset (`train.tsv`, `test.tsv`, `emotions.txt`)

## Usage
Run the pipeline:
```bash
python src/test.py
```

Run tests:
```bash
python -m unittest tests/test_all.py
```
