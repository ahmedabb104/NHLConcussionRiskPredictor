from dataProcessing import getTrainingSetandLabels
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate, RepeatedStratifiedKFold
import numpy as np

X_skaters, y_skaters = getTrainingSetandLabels()
X_train, X_test, y_train, y_test = train_test_split(X_skaters, y_skaters, test_size=0.25, shuffle=True)

# Train without hyperparameter tuning
# rf = RandomForestClassifier(class_weight="balanced")
# rf.fit(X_train, y_train)

# Hyperparameter tuning with 5-fold cross-validation
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}
gridSearch = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    scoring="precision",
    cv=5,
    n_jobs=-1,
    verbose=0
)

gridSearch.fit(X_train, y_train)
bestRandomForestModel = gridSearch.best_estimator_
print(f"Best Hyperparameters: {gridSearch.best_params_}")

# Results without k-fold
# y_pred = rf.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(f"confusion matrix: {confusion_matrix(y_test, y_pred)}")
# print(f"Recall: {recall_score(y_test, y_pred, zero_division=1)}")
# print(f"Precision: {precision_score(y_test, y_pred, zero_division=1)}")
# print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=1)}")
# print(f"Classification Report:\n")
# print(classification_report(y_test, y_pred))


# k-fold cross-validation results
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=1),
    "recall": make_scorer(recall_score, zero_division=1),
    "f1": make_scorer(f1_score, zero_division=1),
}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
crossValidatationResults = cross_validate(bestRandomForestModel, X_test, y_test, cv=cv, scoring=scoring)

print(f"Accuracy scores: {crossValidatationResults["test_accuracy"]}")
print(f"Mean accuracy score: {np.mean(crossValidatationResults["test_accuracy"])}")

print(f"Precision scores: {crossValidatationResults["test_precision"]}")
print(f"Mean precision score: {np.mean(crossValidatationResults["test_precision"])}")

print(f"Recall scores: {crossValidatationResults["test_recall"]}")
print(f"Mean recall score: {np.mean(crossValidatationResults["test_recall"])}")

print(f"F1 scores: {crossValidatationResults["test_f1"]}")
print(f"Mean F1 score: {np.mean(crossValidatationResults["test_f1"])}")
