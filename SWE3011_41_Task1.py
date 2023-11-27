from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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

# Create a pipeline with TfidfVectorizer and RandomForestClassifier
random_forest_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

# Define the parameters for GridSearchCV
random_forest_params = {
    # 'vectorizer__analyzer': ['word', 'char', 'char_wb'],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'clf__n_estimators': [10, 100],
    'clf__max_depth': [6, 8, 10, 12],
    'clf__min_samples_leaf': [8, 12, 18],
    # 'clf__min_samples_split': [8, 16, 20]
}

# GridSearchCV
grid_cv = GridSearchCV(
    random_forest_pipeline, param_grid=random_forest_params, cv=3, n_jobs=-1)
grid_cv.fit(X_train, y_train)

# Print the results
print('Best Hyperparameters: ', grid_cv.best_params_)
print('Best Accuracy: {:.4f}'.format(grid_cv.best_score_))

# Get the best model from the grid search
best_random_forest_model = grid_cv.best_estimator_

# Evaluate the best model on the test set
X_test_tfidf = best_random_forest_model.named_steps['vectorizer'].transform(
    X_test)
y_pred = best_random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
