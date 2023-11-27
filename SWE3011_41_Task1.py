from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset

# Load datasets
train_ds = load_dataset("glue", "sst2", split="train")
test_ds = load_dataset("csv", data_files="./test_dataset.csv")['train']

# Extract features and labels
X_train, y_train = train_ds['sentence'], train_ds['label']
X_test, y_test = test_ds['sentence'], test_ds['label']

# Function to transform data using TF-IDF


def transform_data(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Function to evaluate a model


def evaluate_model(clf, X_test_tfidf, y_test):
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# Logistic Regression with GridSearchCV
# logistic_regression_params = {
#     'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'clf__penalty': ['l1', 'l2']
# }

# logistic_regression_pipeline = Pipeline([
#     ('vectorizer', TfidfVectorizer()),
#     ('clf', LogisticRegression())
# ])

# logistic_regression_grid = GridSearchCV(
#     logistic_regression_pipeline, logistic_regression_params, cv=5, scoring='accuracy', n_jobs=-1)
# logistic_regression_grid.fit(X_train, y_train)

# best_logistic_regression_params = logistic_regression_grid.best_params_
# best_logistic_regression_model = logistic_regression_grid.best_estimator_

# print("Best Logistic Regression Parameters:", best_logistic_regression_params)
# evaluate_model(best_logistic_regression_model, X_test, y_test)

# Random Forest with GridSearchCV
random_forest_params = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [None, 10],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2]
}


random_forest_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

random_forest_grid = GridSearchCV(
    random_forest_pipeline, random_forest_params, cv=5, scoring='accuracy', n_jobs=1)
random_forest_grid.fit(X_train, y_train)

best_random_forest_params = random_forest_grid.best_params_
best_random_forest_model = random_forest_grid.best_estimator_

print("before\n")
print("Best Random Forest Parameters:", best_random_forest_params)
print("after\n")
evaluate_model(best_random_forest_model, X_test, y_test)

# Naive Bayes with GridSearchCV
naive_bayes_params = {
    'clf__alpha': [0.1, 0.5, 1.0]
}

naive_bayes_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

naive_bayes_grid = GridSearchCV(
    naive_bayes_pipeline, naive_bayes_params, cv=5, scoring='accuracy', n_jobs=1)
naive_bayes_grid.fit(X_train, y_train)

best_naive_bayes_params = naive_bayes_grid.best_params_
best_naive_bayes_model = naive_bayes_grid.best_estimator_

# print("Best Naive Bayes Parameters:", best_naive_bayes_params)
# evaluate_model(best_naive_bayes_model, X_test, y_test)
