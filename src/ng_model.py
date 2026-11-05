from collections import Counter
import math


class TrigramLanguageModel:
    """
    Trigram Language Model with Laplace (add-1) smoothing and OOV (<UNK>) handling.
    """

    def __init__(self):
        self.trigram_counts = Counter()  # (w1, w2, w3) -> count
        self.bigram_counts = Counter()   # (w1, w2) -> count
        self.vocabulary = set()

    def fit(self, documents):
        """
        Train the model on a list of tokenized documents.
        e.g., [["i", "love", "nlp"], ["happy", "day"]]
        """
        for tokens in documents:
            self.vocabulary.update(tokens)

            padded_tokens = ["<s>", "<s>"] + list(tokens) + ["</s>"]

            for i in range(2, len(padded_tokens)):
                context = (padded_tokens[i - 2], padded_tokens[i - 1])
                word = padded_tokens[i]

                self.trigram_counts[context + (word,)] += 1
                self.bigram_counts[context] += 1

        # Include special tokens in vocabulary
        self.vocabulary.update({"<s>", "</s>", "<UNK>"})

    def calculate_probability(self, word, context, smoothing=True):
        """
        Calculate P(word | context), where context is a 2-word tuple.
        """
        word = word if word in self.vocabulary else "<UNK>"
        context = tuple(w if w in self.vocabulary else "<UNK>" for w in context)

        trigram_count = self.trigram_counts[context + (word,)]
        bigram_count = self.bigram_counts[context]

        if smoothing:
            return (trigram_count + 1) / (bigram_count + len(self.vocabulary))

        if bigram_count == 0:
            return 0.0

        return trigram_count / bigram_count

    def calculate_sentence_probability(self, tokens, smoothing=True):
        """
        Calculate log probability and return step-by-step probabilities for a sentence.
        """
        # If passed as a raw string, split into words
        if isinstance(tokens, str):
            tokens = tokens.split()

        padded_tokens = ["<s>", "<s>"] + list(tokens) + ["</s>"]

        log_probability = 0.0
        steps = []

        for i in range(2, len(padded_tokens)):
            context = (padded_tokens[i - 2], padded_tokens[i - 1])
            word = padded_tokens[i]

            probability = self.calculate_probability(word, context, smoothing=smoothing)

            steps.append({
                "context": context,
                "word": word,
                "probability": probability
            })

            if probability == 0:
                return float("-inf"), steps

            log_probability += math.log(probability)

        return log_probability, steps

    def calculate_perplexity(self, tokens, smoothing=True):
        """
        Calculate the perplexity of a sentence.
        Lower perplexity indicates higher model confidence.
        """
        log_prob, steps = self.calculate_sentence_probability(tokens, smoothing=smoothing)
        if not steps or log_prob == float("-inf"):
            return float("inf")
        return math.exp(-log_prob / len(steps))
