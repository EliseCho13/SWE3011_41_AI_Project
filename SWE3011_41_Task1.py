from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset

# import sys
# print("interpreter path is ", sys.executable) #python 설치 위치 확인
# moduleNotFound에러 발생, pip install -U numpy scikit-learn scipy로 해결

# Load datasets
train_ds = load_dataset("glue", "sst2", split="train")
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
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # print("\nX_train_tfidf is \n", X_train_tfidf.toarray())
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
    clf = LogisticRegression(C=1, solver='lbfgs', penalty='l2', max_iter=1000)
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
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=4, min_samples_split=2, n_jobs=-1)
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
    clf = MultinomialNB(alpha=1.0)  # 0.1, 0.5, 1.0
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
