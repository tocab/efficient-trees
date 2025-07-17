"""Memory profiling of efficient-trees Random Forest vs. sklearn and lightgbm."""

import multiprocessing as mp

import kagglehub
import lightgbm as lgbm
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from memory_profiler import memory_usage
from sklearn.ensemble import RandomForestClassifier as SkLearnRandomForestClassifier

from efficient_trees.random_forest import RandomForestClassifier as ETRandomForestClassifier


# Wrapper to measure memory usage over time
def measure_memory_usage(func, queue, *args, **kwargs):
    """Measure memory usage of a function over time."""
    mem_usage = memory_usage((func, args, kwargs))  # type: ignore
    queue.put(mem_usage)


# Example functions to benchmark
def train_sklearn_forest(data_path):
    """Train a sklearn Random Forest."""
    data = pd.read_parquet(data_path)
    columns_to_exclude = ["customer_ID", "__index_level_0__", "S_2", "D_63", "D_64", "target"]
    forest = SkLearnRandomForestClassifier(n_estimators=10, max_depth=4, criterion="entropy")
    forest.fit(data[[col for col in data.columns if col not in columns_to_exclude]], data["target"])


def train_efficient_forest_lazy(data_path):
    """Train an efficient-trees Random Forest with lazy evaluation."""
    data = pl.scan_parquet(data_path)
    columns_to_exclude = ["customer_ID", "__index_level_0__", "S_2", "D_63", "D_64"]
    target_name = "target"
    data = data.drop(columns_to_exclude).fill_null(0.0)
    forest = ETRandomForestClassifier(n_estimators=10, max_depth=4, streaming=True)
    forest.fit(data, target_name)


def train_efficient_forest(data_path):
    """Train an efficient-trees Random Forest with eager evaluation."""
    data = pl.scan_parquet(data_path)
    columns_to_exclude = ["customer_ID", "__index_level_0__", "S_2", "D_63", "D_64"]
    target_name = "target"
    data = data.drop(columns_to_exclude).fill_null(0.0).collect(streaming=True)  # type: ignore
    forest = ETRandomForestClassifier(n_estimators=10, max_depth=4)
    forest.fit(data, target_name)


def train_lightgbm(data_path):
    """Train a lightgbm."""
    data = pl.scan_parquet(data_path)
    columns_to_exclude = ["customer_ID", "__index_level_0__", "S_2", "D_63", "D_64"]

    # It is in general a big pain to make lightgbm compatible with arrow. If anything goes wrong during data
    # preparation, lightgbm defaults to transforming the input data into a csr_matrix which is expensive.
    # To make it run with pyarrow, these requirements need to be given:
    # - pyarrow needs to be installed
    # - cffi needs to be installed
    # Internally, lightgbm will try to import these packages and fail silently if they are not available.
    lgbm_dataset = lgbm.Dataset(
        data=data.drop(columns_to_exclude + ["target"]).collect(streaming=True).to_arrow(),  # type: ignore
        label=data.select("target").collect(streaming=True)["target"].to_arrow(),  # type: ignore
        free_raw_data=True,
    )
    params = {"objective": "binary", "max_depth": 4}
    lgbm.train(
        params=params,
        train_set=lgbm_dataset,
        # To have a fair comparison, num_boost_round could be set to 1. But for boosting, we won't end up with
        # good results after only 1 iteration, so leave num_boost_round at 100 (default).
        # num_boost_round=1
    )


# Main benchmarking script
def main():
    """Main function to benchmark memory usage of different tree implementations."""
    # Set this, otherwise it doesn't match well with polars.
    mp.set_start_method("spawn")
    # Download latest version
    path = kagglehub.dataset_download("odins0n/amex-parquet")
    path += "/train_data.parquet"

    methods = {
        "lightgbm": train_lightgbm,
        "efficient-forest": train_efficient_forest,
        "efficient-forest-lazy": train_efficient_forest_lazy,
        "sklearn": train_sklearn_forest,
    }

    results = {}
    for method_name, method in methods.items():
        print(f"Train {method_name}")
        # Use multiprocessing here to ensure that the memory is cleaned up in the right way
        queue = mp.Queue()
        p = mp.Process(target=measure_memory_usage, args=(method, queue, path))
        p.start()
        p.join()
        p.terminate()
        mem_timestamps = queue.get()
        # mem_timestamps = measure_memory_usage(method, queue, path)
        mem_timestamps.append(0.0)
        results[method_name] = mem_timestamps

    # Plot the results
    plt.figure(figsize=(10, 6))
    for method_name, method_result in results.items():
        plt.plot([val / 10.0 for val in range(1, len(method_result) + 1)], method_result, label=method_name)
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("sklearn_vs_efficient_forest.pdf")
    plt.show()


if __name__ == "__main__":
    main()
