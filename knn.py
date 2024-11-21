from dataProcessing import getTrainingSetandLabels
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X_skaters, y_skaters = getTrainingSetandLabels()
X_train, X_test, y_train, y_test = train_test_split(X_skaters, y_skaters, test_size=0.2)

# Scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train without hyperparameter tuning
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)

# Hyperparameter tuning with 5-fold cross-validation
param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "p": [1, 2]
}
gridSearch = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=0
)

gridSearch.fit(X_train, y_train)
bestKnnModel = gridSearch.best_estimator_    # Model with tuned hyperparameters
print(f"Best Hyperparameters: {gridSearch.best_params_}")

# Results without k-fold
# y_pred = knn.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(f"confusion matrix: {confusion_matrix(y_test, y_pred)}")
# print(f"Recall: {recall_score(y_test, y_pred, zero_division=0)}")
# print(f"Precision: {precision_score(y_test, y_pred, zero_division=0)}")
# print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0)}")

# k-fold cross-validation results
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
crossValidatationResults = cross_validate(bestKnnModel, X_skaters, y_skaters, cv=cv, scoring=scoring)

print(f"Accuracy scores: {crossValidatationResults["test_accuracy"]}")
print(f"Mean accuracy score: {np.mean(crossValidatationResults["test_accuracy"])} +- {np.std(crossValidatationResults["test_accuracy"])}")

print(f"Precision scores: {crossValidatationResults["test_precision"]}")
print(f"Mean precision score: {np.mean(crossValidatationResults["test_precision"])} +- {np.std(crossValidatationResults["test_precision"])}")

print(f"Recall scores: {crossValidatationResults["test_recall"]}")
print(f"Mean recall score: {np.mean(crossValidatationResults["test_recall"])} +- {np.std(crossValidatationResults["test_recall"])}")

print(f"F1 scores: {crossValidatationResults["test_f1"]}")
print(f"Mean F1 score: {np.mean(crossValidatationResults["test_f1"])} +- {np.std(crossValidatationResults["test_f1"])}")