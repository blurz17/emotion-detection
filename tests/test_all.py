import sys
from pathlib import Path
import unittest

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import load_data, load_emotions
from ng_model import comparison, ngram_model
from processing import NEGATIVE_WORDS, clean_noise, process_text


class TestDataLoader(unittest.TestCase):

    def test_load_train_and_test(self):
        train_df = load_data("train")
        test_df = load_data("test")
        self.assertFalse(train_df.empty)
        self.assertFalse(test_df.empty)
        self.assertIn("text", train_df.columns)
        self.assertIn("labels", train_df.columns)
        self.assertIn("comment_id", train_df.columns)

    def test_load_emotions(self):
        emotions = load_emotions()
        self.assertEqual(len(emotions), 28)
        self.assertEqual(emotions[0], "admiration")
        self.assertEqual(emotions[-1], "neutral")

    def test_invalid_split(self):
        with self.assertRaises(ValueError):
            load_data("invalid_split")


class TestProcessing(unittest.TestCase):

    def test_clean_noise(self):
        dirty = "Check this out: https://example.com/test! It's awesome 123"
        cleaned = clean_noise(dirty)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("!", cleaned)

    def test_process_text_negation_preservation(self):
        text = "I do not like this, I don't care"
        tokens = process_text(text)
        self.assertIn("not", tokens)
        self.assertTrue("not" in tokens or "dont" in tokens)

    def test_lemmatization_and_lowercasing(self):
        text = "Running studies and Dogs"
        tokens = process_text(text)
        self.assertTrue(all(w.islower() for w in tokens))


class TestNGramModel(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            ["i", "love", "nlp"],
            ["i", "love", "python"],
            ["nlp", "is", "fun"],
        ]

    def test_comparison_counts(self):
        tria, bia = comparison(self.corpus, ("<s>", "<s>"), "i")
        self.assertEqual(tria, 2)
        self.assertEqual(bia, 3)

    def test_comparison_zero_counts(self):
        tria, bia = comparison(self.corpus, ("random", "word"), "missing")
        self.assertEqual(tria, 0)
        self.assertEqual(bia, 0)

    def test_ngram_model_probability(self):
        test_sentences = [["i", "love", "nlp"]]
        dataset = [["i", "love", "nlp"]]
        results = ngram_model(test_sentences, dataset)
        self.assertIn("log_probability", results[1])
        self.assertIn("probability", results[1])
        self.assertGreater(results[1]["probability"], 0.0)
        self.assertLess(results[1]["log_probability"], 0.0)


if __name__ == "__main__":
    unittest.main()
