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

# print("train sentences are ", train_ds['sentence'])
# print("train labels are ", train_ds['label'])
# labels are positive and negative


def transform_data(X_train, X_test):
    """
    Input:
    - X_train, X_test: Series containing the text data for training and testing respectively.

    Output:
    - X_train_tfidf, X_test_tfidf: Transformed text data in TF-IDF format for training and testing respectively.
    - vectorizer: Fitted TfidfVectorizer object.
    """
    #########################################
    # Convert the text data to TF-IDF format and return the transformed data and the vectorizer
    # TF-IDF(Term Frequency-Inverse Document Frequency)는 단어의 빈도와 역 문서 빈도(문서의 빈도에 특정 식을 취함)를 사용하여 DTM 내의 각 단어들마다 중요한 정도를 가중치로 주는 방법
    # TF와 IDF를 곱한 값.
    vectorizer = TfidfVectorizer()
    # fit_transform()은 train dataset의 mean. variance를 학습시키는 데에 사용
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # print("\nX_train_tfidf is \n", X_train_tfidf.toarray())
    # test data까지 학습하면 성능에 영향을 주므로 구분해야 해서 fit_말고 그냥 transform 사용
    # train data로부터 학습한 mean, variance를 test data에 적용하기 위해  transform사용
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
    # Train a logistic regression model and return the trained model
    clf = LogisticRegression()
    clf.fit(X_train_tfidf, y_train)
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    # https://velog.io/@gayeon/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%84%EC%84%9D-%EC%B4%88%EB%B3%B4%EC%9E%90%EB%A5%BC-%EC%9C%84%ED%95%9C-Logistic-Regression-with-Scikit-Learn
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
    # Train a Random Forest classifier and return the trained model
    clf = RandomForestClassifier()
    clf.fit(X_train_tfidf, y_train)
    # https://zephyrus1111.tistory.com/249
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
    # Train a Multinomial Naive Bayes classifier and return the trained model
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)
    # https://todayisbetterthanyesterday.tistory.com/17
    # https://todayisbetterthanyesterday.tistory.com/18
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
    # Evaluate the model and print the results
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    # https://datainsider.tistory.com/53
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
