from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report

# Fetch the SMS Spam dataset
data = fetch_openml(data_id=1119, as_frame=True)

# Automatically select the first column as text
X = data.data.iloc[:, 0].astype(str)  # Convert all entries to string
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Hyperparameters for GridSearch
params = {
    'clf__C': [0.1, 1, 10],
    'tfidf__max_df': [0.8, 1.0],
    'tfidf__ngram_range': [(1,1), (1,2)]
}

# Grid search
grid = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Predictions and evaluation
y_pred = grid.best_estimator_.predict(X_test)
print("Best Parameters:", grid.best_params_)
print(classification_report(y_test, y_pred))

