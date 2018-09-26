# region Imports
import random
import math
from collections import Counter

from sklearn.feature_extraction.text \
    import CountVectorizer as Vectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.feature_selection import mutual_info_classif

# endregion

# region Constants

# Data split proportions
P_TRAIN, P_VALIDATION, P_TEST = 0.7, 0.15, 0.15
assert sum([P_TRAIN, P_VALIDATION, P_TEST]) == 1

# Paths to data
FAKE_PATH = "clean_fake.txt"
REAL_PATH = "clean_real.txt"

# Labels
FAKE_LABEL = "fake"
REAL_LABEL = "real"


# endregion

def print_logged(func):
    def ret(*args, **kwargs):
        val = func(*args, **kwargs)
        print(val)
        return val

    return ret


def read_labelled_headlines():
    """
    Makes a list of randomly shuffled (headline, label) tuples
    """
    ret = [
        (headline.strip(), label)
        for (path, label) in [(FAKE_PATH, FAKE_LABEL), (REAL_PATH, REAL_LABEL)]
        for headline in open(path, "r").readlines()
    ]

    # Seeding RNG for consistent shuffling
    random.seed(411)
    random.shuffle(ret)

    return ret


def make_vectorizer(top_words):
    return Vectorizer(

        # Split based on words
        analyzer="word",

        # Cap features
        max_features=top_words,

        # Only take words that appear in 2 or more headlines
        # (deciding based on headline-unique
        # words is probably over-fitting)
        min_df=2,

        # Make the math simple for computing
        # probabilities / info-gain (word inclusion is either-or)
        # Shouldn't have a significant impact because very few
        # headlines contain have duplicate words
        binary=True
    )


def load_data(vectorizer):
    # Read in the headlines
    labelled_headlines = read_labelled_headlines()

    # Train the vectorizer on all the headlines,
    # and create a document-term matrix (documents are headlines)
    doc_matrix = vectorizer.fit_transform(
        headline for (headline, label) in labelled_headlines
    )

    # Split up the matrix into data
    num_rows = len(labelled_headlines)
    train_end = int(P_TRAIN * num_rows)
    valid_end = (train_end + 1) + int(P_VALIDATION * num_rows)
    train_data, valid_data, test_data = doc_matrix[:train_end], \
                                        doc_matrix[train_end:valid_end], \
                                        doc_matrix[valid_end:]

    # Split up the labels using same indices
    labels = [label for (headline, label) in labelled_headlines]
    train_labels, valid_labels, test_labels = labels[:train_end], \
                                              labels[train_end: valid_end], \
                                              labels[valid_end:]

    return (train_data, train_labels), \
           (valid_data, valid_labels), \
           (test_data, test_labels)


def evaluate_model(model, data, labels):
    predictions = model.predict(data)

    # Want to know (correct / total) predictions
    return sum(predictions[i] == labels[i]
               for i in range(len(labels))) / len(labels)


def select_model(train_data, train_labels,
                 valid_data, valid_labels):
    # Look for the best model over all combinations of depth and criterion
    best_model, best_score = None, 0
    for criterion in ["entropy", "gini"]:
        for max_depth in range(2, 7):

            model = DecisionTreeClassifier(
                random_state=411,
                criterion=criterion,
                max_depth=max_depth
            )

            # Train the model on the data
            model.fit(train_data, train_labels)

            # Evaluate performance on validation data
            score = evaluate_model(model, valid_data, valid_labels)

            # Print out the performance
            print(f"Model(depth={max_depth}, criteria={criterion}): score = {score}")

            # Update the best scoring model
            if score > best_score:
                best_score = score
                best_model = model

    return best_model


def output_model_diagram(model, feature_names):
    export_graphviz(model,
                    out_file="best-decision-tree.gv",
                    max_depth=2,
                    class_names=model.classes_,
                    feature_names=feature_names)


def n_log_n(n):
    """Returns n * log(n)"""
    if n == 0:
        return 0

    return n * math.log(n)


def label_entropy(labels):
    """
    Computes the entropy of the list of labels
    """
    # Count each label
    label_counts = Counter(labels)

    # Sum up the probabilities
    return -1 * sum(
        n_log_n(label_counts[label] / len(labels))
        for label in label_counts
    )


def label_entropy_given_word(doc_matrix, labels, vocabulary, word):
    # Column of the word in the document matrix
    word_col = doc_matrix.getcol(vocabulary[word]).toarray().flatten()

    presence_indices = {
        0: (1 - word_col).nonzero()[0],  # Indices where word is NOT present
        1: word_col.nonzero()[0]  # Indices where word is present
    }

    # Maps (label, word_presence) : number of headlines
    label_and_presence_count = Counter(
        (presence, labels[ind])
        for presence in presence_indices
        for ind in presence_indices[presence]
    )

    # Turn the above count into probabilities
    p_label_and_presence = {
        x: (label_and_presence_count[x] / len(labels))
        for x in label_and_presence_count
    }

    # Probabilities of presence
    p_presence = {
        presence: len(presence_indices[presence]) / len(labels)
        for presence in presence_indices
    }

    # Turn the above probabilities into conditional probabilities
    p_label_given_presence = {
        (presence, label): p_label_and_presence[(presence, label)] /
                           p_presence[presence]
        for (presence, label) in p_label_and_presence
    }

    # Sum up the above probabilities (want P(Y) * nlogn(P(Y|X))
    return -1 * sum(
        p_presence[presence] * n_log_n(p_label_given_presence[(presence, label)])
        for (presence, label) in p_label_given_presence
    )


def compute_information_gain(doc_matrix, labels, vocabulary, word):
    return label_entropy(labels) - \
           label_entropy_given_word(doc_matrix, labels, vocabulary, word)


if __name__ == '__main__':

    # Create a vectorizer and data based on it
    vectorizer = make_vectorizer(1000)
    (train_data, train_labels), \
    (valid_data, valid_labels), \
    (test_data, test_labels) = load_data(vectorizer)

    # Choose the best model and output a diagram of it
    best_model = select_model(train_data, train_labels,
                              valid_data, valid_labels)
    output_model_diagram(best_model, vectorizer.get_feature_names())

    info_gains = dict(zip(
        vectorizer.get_feature_names(),
        mutual_info_classif(train_data, train_labels, discrete_features=True)
    ))

    # Compute information gain on several keywords
    for word in ["trump", "donald", "hillary", "clinton", "korea", "america"]:
        info_gain = compute_information_gain(
            train_data,
            train_labels,
            vectorizer.vocabulary_,
            word
        )

        print(f"Information Gain in Label by splitting on {word}: {info_gain}")
