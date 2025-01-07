import kagglehub
import polars as pl
from sklearn.metrics import accuracy_score

from efficient_trees.tree import DecisionTreeClassifier
from examples.utils.utils import plot_tree

# Download latest version
path = kagglehub.dataset_download("oktayrdeki/heart-disease")
data = pl.scan_csv(path + "/heart_disease.csv")

# map values in binary columns
binary_columns = [
    "Smoking",
    "Family Heart Disease",
    "Diabetes",
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol",
    "Heart Disease Status",
]
target_name = "Heart Disease Status"
data = data.with_columns([pl.col(col).replace({"Yes": 1, "No": 0}).cast(pl.Int8) for col in binary_columns])

# Define categorical columns
categorical_columns = ["Gender", "Exercise Habits", "Alcohol Consumption", "Stress Level", "Sugar Consumption"]

columns = (
    [
        "Age",
        "Blood Pressure",
        "Cholesterol Level",
        "BMI",
        "Sleep Hours",
        "Triglyceride Level",
        "Fasting Blood Sugar",
        "CRP Level",
        "Homocysteine Level",
    ]
    + binary_columns
    + categorical_columns
)

data = data.select(columns)

data = data.collect().sample(fraction=1.0, shuffle=True)
count_training_data = int(len(data) * 0.8)

training_data = data.slice(0, count_training_data)
test_data = data.slice(count_training_data)

tree = DecisionTreeClassifier(max_depth=8, streaming=True, categorical_columns=categorical_columns)
tree.fit(training_data, target_name)
tree.save_model("decision_tree.pkl")
plot_tree(tree.tree, "decision_tree_iris.pdf")

for data_type, dataset in zip(["Training", "Test"], [training_data, test_data]):
    predictions = tree.predict_many(dataset.drop(target_name).fill_null(0.0))
    accuracy = accuracy_score(dataset.select(target_name)[target_name], predictions)
    print(f"{data_type} Accuracy: {accuracy:.2f}")
