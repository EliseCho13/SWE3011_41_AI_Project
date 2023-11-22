from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
# import sys
# print("interpreter path is ", sys.executable)
# moduleNotFound에러 발생, pip install -U numpy scikit-learn scipy로 해결

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
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    #########################################
    return X_train_tfidf, X_test_tfidf, vectorizer


def logistic_regression(X_train_tfidf, y_train):
    """
    Input:
    - X_train_tfidf: Transformed text data in TF-IDF format for training.
    - y_train: Series containing the labels for training.

    Output:
    - clf: Trained Logistic Regression model.
    """
    #########################################
    # TODO: Train a logistic regression model and return the trained model
    clf = LogisticRegression()
    clf.fit(X_train_tfidf, y_train)
    #########################################
    return clf


def random_forest(X_train_tfidf, y_train):
    """
    Input:
    - X_train_tfidf: Transformed text data in TF-IDF format for training.
    - y_train: Series containing the labels for training.

    Output:
    - clf: Trained Random Forest classifier.
    """
    #########################################
    # TODO: Train a Random Forest classifier and return the trained model
    clf = RandomForestClassifier()
    clf.fit(X_train_tfidf, y_train)
    #########################################
    return clf


def naive_bayes_classifier(X_train_tfidf, y_train):
    """
    Input:
    - X_train_tfidf: Transformed text data in TF-IDF format for training.
    - y_train: Series containing the labels for training.

    Output:
    - clf: Trained Multinomial Naive Bayes classifier.
    """
    #########################################
    # TODO: Train a Multinomial Naive Bayes classifier and return the trained model
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)
    #########################################
    return clf


def evaluate_model(clf, X_test_tfidf, y_test):
    """
    Input:
    - clf: Trained Logistic Regression model.
    - X_test_tfidf: Transformed text data in TF-IDF format for testing.
    - y_test: Series containing the labels for testing.

    Output:
    - None (This function will print the evaluation results.)
    """
    #########################################
    # TODO: Evaluate the model and print the results
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    #########################################
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


X_train, y_train = train_ds['sentence'], train_ds['label']
X_test, y_test = test_ds['sentence'], test_ds['label']
X_train_tfidf, X_test_tfidf, vectorizer = transform_data(X_train, X_test)
clf = logistic_regression(X_train_tfidf, y_train)
# clf_rf = random_forest(X_train_tfidf, y_train)
# clf_nb = naive_bayes_classifier(X_train_tfidf, y_train)

evaluate_model(clf, X_test_tfidf, y_test)
# evaluate_model(clf_rf, X_test_tfidf, y_test)
# evaluate_model(clf_nb, X_test_tfidf, y_test)
