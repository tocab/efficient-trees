"""Iris Classification."""

import polars as pl
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from efficient_trees.random_forest import RandomForestClassifier
from efficient_trees.tree import DecisionTreeClassifier
from examples.utils.utils import plot_tree

iris = load_iris()
X, y = iris.data, iris.target  # type: ignore

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# data
X_train_pl = pl.DataFrame(X_train, schema=iris.feature_names)  # type: ignore
y_train_pl = pl.Series(y_train)
X_test_pl = pl.DataFrame(X_test, schema=iris.feature_names)  # type: ignore
y_test_pl = pl.Series(y_test)
train_pl = X_train_pl.with_columns(target=y_train_pl)

models = [
    DecisionTreeClassifier(max_depth=4),
    RandomForestClassifier(n_estimators=3, max_depth=4, max_samples=0.2, sample_with_replacement=True),
]
# Train the decision tree
for model in models:
    model.fit(train_pl, "target")
    if isinstance(model, DecisionTreeClassifier):
        plot_tree(model.tree, "decision_tree_iris.pdf")  # type: ignore

    # Predictions and evaluation
    y_train_pred_pl = model.predict(X_train_pl.iter_rows(named=True))
    y_test_pred_pl = model.predict(X_test_pl.iter_rows(named=True))

    # Calculate accuracy scores
    train_accuracy = accuracy_score(y_train_pl, y_train_pred_pl)
    test_accuracy = accuracy_score(y_test_pl, y_test_pred_pl)

    print(model)
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
