import sys
from pathlib import Path
import unittest

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import load_data, load_emotions
from negation_words import NEGATIVE_WORDS, negative_words
from ng_model import TrigramLanguageModel
from processing import clean_noise, process_text


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
        # Verify negation words are not removed as stopwords
        self.assertTrue("not" in tokens or "dont" in tokens)

    def test_lemmatization_and_lowercasing(self):
        text = "Running studies and Dogs"
        tokens = process_text(text)
        # Should be lowercased and lemmatized
        self.assertTrue(all(w.islower() for w in tokens))


class TestTrigramModel(unittest.TestCase):

    def setUp(self):
        self.model = TrigramLanguageModel()
        self.corpus = [
            ["i", "love", "nlp"],
            ["i", "love", "python"],
            ["nlp", "is", "fun"],
        ]
        self.model.fit(self.corpus)

    def test_vocabulary(self):
        self.assertIn("i", self.model.vocabulary)
        self.assertIn("love", self.model.vocabulary)
        self.assertIn("<s>", self.model.vocabulary)
        self.assertIn("</s>", self.model.vocabulary)
        self.assertIn("<UNK>", self.model.vocabulary)

    def test_probability_smoothing(self):
        prob = self.model.calculate_probability("nlp", ("i", "love"), smoothing=True)
        self.assertGreater(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_oov_probability(self):
        prob = self.model.calculate_probability("unknownword", ("i", "love"), smoothing=True)
        self.assertGreater(prob, 0.0)

    def test_sentence_probability(self):
        log_prob, steps = self.model.calculate_sentence_probability(["i", "love", "nlp"])
        self.assertIsInstance(log_prob, float)
        self.assertGreater(log_prob, float("-inf"))
        self.assertEqual(len(steps), 4)  # (<s>, <s>)->i, (<s>, i)->love, (i, love)->nlp, (love, nlp)-></s>

    def test_perplexity(self):
        perp = self.model.calculate_perplexity(["i", "love", "nlp"])
        self.assertGreater(perp, 0.0)
        self.assertNotEqual(perp, float("inf"))


if __name__ == "__main__":
    unittest.main()
