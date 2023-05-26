from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

import numpy as np

# Create a stemmer object
stemmer = PorterStemmer()


# Preprocessing and cleaning functions
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words or token == "tool"]

    # Remove non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]

    # Stem the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return stemmed_tokens


def create_word2vec_model(tokens):
    # Create Word2Vec model for the stemmed tokens
    model = Word2Vec([tokens], min_count=1)
    return model


def calculate_cosine_similarity(model, input_string, tool_description):
    # Preprocess the input string and tool description
    tokens1 = preprocess_text(input_string)
    tokens2 = preprocess_text(tool_description)

    # Get the intersection of tokens1 and the model's vocabulary
    tokens1 = [token for token in tokens1 if token in model.wv.key_to_index]

    # Get the intersection of tokens2 and the model's vocabulary
    tokens2 = [token for token in tokens2 if token in model.wv.key_to_index]

    # Check if either tokens1 or tokens2 is empty
    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0.0, set()

    # Get vectors for the tokens
    vectors1 = model.wv[tokens1]
    vectors2 = model.wv[tokens2]

    # Check if vectors2 is empty or contains only zeros
    if vectors2.size == 0 or np.all(vectors2 == 0):
        return 0.0, set()

    # Pad the vectors to ensure the same dimensionality
    max_length = max(len(vectors1), len(vectors2))
    vectors1 = np.pad(vectors1, [(0, max_length - len(vectors1)), (0, 0)])
    vectors2 = np.pad(vectors2, [(0, max_length - len(vectors2)), (0, 0)])

    # Calculate the cosine similarity between the vectors
    similarity = cosine_similarity(vectors1, vectors2)

    # Find the common words in both stemmed texts
    common_words = set(tokens1).intersection(tokens2)

    return similarity[0][0], common_words
