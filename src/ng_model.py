def comparison(data_set, context, word):
    tria = 0
    bia = 0

    for sentence in data_set:
        sentence = ["<s>", "<s>"] + sentence + ["</s>"]

        for i in range(len(sentence) - 2):
            if tuple(sentence[i:i+2]) == context:
                bia += 1

                if sentence[i+2] == word:
                    tria += 1

    return tria, bia


def ngram_model(sentences, data_set):
    sentence_probabilities = {}

    # Calculate vocabulary size (|V|) for Laplace smoothing
    vocab = set()
    for sentence in data_set:
        vocab.update(sentence)
    vocab.update(["<s>", "</s>"])
    V = len(vocab)

    for number, sentence in enumerate(sentences, start=1):
        sentence = ["<s>", "<s>"] + sentence + ["</s>"]
        probability = 1.0

        for i in range(2, len(sentence)):
            context = tuple(sentence[i-2:i])
            word = sentence[i]

            tria, bia = comparison(data_set, context, word)

            # Laplace (Add-1) Smoothing
            word_probability = (tria + 1) / (bia + V)

            print(f"P({word} | {context}) = ({tria} + 1)/({bia} + {V}) = {word_probability:.6e}")

            probability *= word_probability

        sentence_probabilities[number] = probability

    return sentence_probabilities
