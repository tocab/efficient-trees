import multiprocessing

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from memory_profiler import memory_usage
from sklearn.tree import DecisionTreeClassifier as SkLearnDecisionTreeClassifier

from efficient_trees.tree import DecisionTreeClassifier as ETDecisionTreeClassifier


# Wrapper to measure memory usage over time
def measure_memory_usage(func, *args, **kwargs):
    mem_usage = memory_usage((func, args, kwargs))
    return mem_usage


# Example functions to benchmark
def sklearn_tree(data_path):
    data = pd.read_parquet(data_path)
    columns_to_exclude = ["customer_ID", "__index_level_0__", "S_2", "D_63", "D_64", "target"]
    tree = SkLearnDecisionTreeClassifier(max_depth=4, criterion="entropy")
    tree.fit(data[[col for col in data.columns if col not in columns_to_exclude]], data["target"])


def efficient_tree(data_path):
    data = pl.scan_parquet(data_path)
    columns_to_exclude = [
        "customer_ID",
        "__index_level_0__",
        "S_2",
        "D_63",
        "D_64",
    ]
    target_name = "target"
    data = data.drop(columns_to_exclude).fill_null(0.0)
    tree = ETDecisionTreeClassifier(max_depth=4)
    tree.fit(data, target_name)


# Main benchmarking script
def main():
    # Set this, otherwise it doesn't match well with polars.
    multiprocessing.set_start_method("spawn")
    # Download latest version
    path = kagglehub.dataset_download("odins0n/amex-parquet")
    path += "/train_data.parquet"

    # Measure memory usage for efficient-trees
    mem_efficient = measure_memory_usage(efficient_tree, path)
    mem_efficient.append(0.0)

    # Measure memory usage for sklearn
    mem_sklearn = measure_memory_usage(sklearn_tree, path)
    mem_sklearn.append(0.0)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot([val / 10.0 for val in range(1, len(mem_sklearn) + 1)], mem_sklearn, label="Scikit-Learn")
    plt.plot([val / 10.0 for val in range(1, len(mem_efficient) + 1)], mem_efficient, label="Efficient-Trees")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("sklearn_vs_et.pdf")
    plt.show()


if __name__ == "__main__":
    main()
