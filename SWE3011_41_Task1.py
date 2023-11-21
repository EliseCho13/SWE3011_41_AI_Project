from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

train_ds = load_dataset("glue", "sst2", split="train")

# Evaluation should be done using test_ds
test_ds = load_dataset("csv", data_files="./test_dataset.csv")['train']


def transform_data(X_train, X_test):
    """
    Input:
    - X_train, X_test: Series containing the text data for training and testing respectively.

    Output:
    - X_train_tfidf, X_test_tfidf: Transformed text data in TF-IDF format for training and testing respectively.
    - vectorizer: Fitted TfidfVectorizer object.
    """
    #########################################
    # TODO: Convert the text data to TF-IDF format and return the transformed data and the vectorizer
    vectorizer = None
    X_test_tfidf = None
    X_train_tfidf = None
    #########################################
    return X_train_tfidf, X_test_tfidf, vectorizer


X_train, y_train = train_ds['sentence'], train_ds['label']
X_test, y_test = test_ds['sentence'], test_ds['label']
X_train_tfidf, X_test_tfidf, vectorizer = transform_data(X_train, X_test)
