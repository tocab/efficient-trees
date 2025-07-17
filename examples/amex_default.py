"""Amex Default Prediction example."""

import kagglehub
import polars as pl
from sklearn.metrics import accuracy_score

from efficient_trees.tree import DecisionTreeClassifier
from efficient_trees.random_forest import RandomForestClassifier
from examples.utils.utils import plot_tree

# Download latest version
path = kagglehub.dataset_download("odins0n/amex-parquet")
data = pl.scan_parquet(path + "/train_data.parquet")

columns_to_exclude = [
    "customer_ID",
    "__index_level_0__",
    "S_2",  # Date column, would be needed to transformed
    "D_63",  # Another string column
    "D_64",  # Another string column
]
target_name = "target"

data = data.drop(columns_to_exclude).fill_null(0.0)

models = [
    DecisionTreeClassifier(max_depth=4, streaming=True),
    RandomForestClassifier(seed=42, n_estimators=100, max_depth=4, streaming=True)
]

for model in models:
    model.fit(data, target_name)
    if isinstance(model, DecisionTreeClassifier):
        model.save_model("decision_tree.pkl")
        plot_tree(model.tree, "decision_tree_iris.pdf")  # type: ignore

    model_predictions = model.predict_many(data.drop(target_name))
    model_train_accuracy = accuracy_score(data.select(target_name).collect()[target_name], model_predictions)
    print(model)
    print(f"Training Accuracy: {model_train_accuracy:.2f}")