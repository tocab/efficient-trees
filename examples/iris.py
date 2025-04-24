"""Iris Classification."""

import polars as pl
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from efficient_trees.tree import DecisionTreeClassifier
from examples.utils.utils import plot_tree

iris = load_iris()
X, y = iris.data, iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# data
X_train_pl = pl.DataFrame(X_train, schema=iris.feature_names)
y_train_pl = pl.Series(y_train)
X_test_pl = pl.DataFrame(X_test, schema=iris.feature_names)
y_test_pl = pl.Series(y_test)
train_pl = X_train_pl.with_columns(target=y_train_pl)

# Train the decision tree
decision_tree_classifier = DecisionTreeClassifier(max_depth=4)
decision_tree_classifier.fit(train_pl, "target")
plot_tree(decision_tree_classifier.tree, "decision_tree_iris.pdf")

# Predictions and evaluation
y_train_pred_pl = decision_tree_classifier.predict(X_train_pl.iter_rows(named=True))
y_test_pred_pl = decision_tree_classifier.predict(X_test_pl.iter_rows(named=True))

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train_pl, y_train_pred_pl)
test_accuracy = accuracy_score(y_test_pl, y_test_pred_pl)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
