"""Heart Disease Classification."""

import kagglehub
import polars as pl
from sklearn.metrics import accuracy_score

from efficient_trees.random_forest import RandomForestClassifier
from efficient_trees.tree import DecisionTreeClassifier
from examples.utils.utils import plot_tree

# Download latest version
path = kagglehub.dataset_download("colewelkins/cardiovascular-disease")
data = pl.scan_csv(path + "/cardio_data_processed.csv")

# drop columns that should not be used
data = data.drop(["id", "age", "bp_category", "bp_category_encoded"])

target_name = "cardio"

# Define categorical columns
categorical_columns = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

data = data.collect().sample(fraction=1.0, shuffle=True)
count_training_data = int(len(data) * 0.8)

training_data = data.slice(0, count_training_data)
test_data = data.slice(count_training_data)

models = [
    DecisionTreeClassifier(max_depth=8, streaming=True, categorical_columns=categorical_columns),
    RandomForestClassifier(
        seed=42, n_estimators=3, max_depth=8, streaming=True, categorical_columns=categorical_columns
    ),
]

for model in models:
    model.fit(training_data, target_name)
    if isinstance(model, DecisionTreeClassifier):
        model.save_model("decision_tree.pkl")
        plot_tree(model.tree, "decision_tree_heart_disease.pdf")  # type: ignore

    for data_type, dataset in zip(["Training", "Test"], [training_data, test_data], strict=True):
        predictions = model.predict_many(dataset.drop(target_name).fill_null(0.0))
        accuracy = accuracy_score(dataset.select(target_name)[target_name], predictions)
        print(model)
        print(f"{data_type} Accuracy: {accuracy:.2f}")
