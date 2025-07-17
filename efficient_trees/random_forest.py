"""
This module defines a `RandomForestClassifier` that implements a random forest classifier using the Polars library.

The class is designed to handle both numerical and categorical features, and can optionally
use lazy evaluation and streaming capabilities of Polars.
"""

import pickle
from collections.abc import Iterable

import polars as pl

from efficient_trees.enums import Criterion
from efficient_trees.tree import DecisionTreeClassifier

class RandomForestClassifier:
    """
    A random forest classifier using Polars as backend.
    """

    def __init__(
        self,
        seed: int | None = None,        
        n_estimators: int = 100,
        max_samples: int | float | None = 0.2,
        sample_with_replacement: bool = True,
        streaming: bool = False,        
        criterion: Criterion = Criterion.ENTROPY,
        categorical_columns: list[str] | None = None,
        max_depth: int | None = None,
    ):
        """
        Initialize the RandomForestClassifier.

        :param seed: Random seed for reproducibility. If None, no seed is set.
        :param n_estimators: Number of trees in the forest.
        :param max_samples: The number of samples to draw from X to train each decision tree.
            - If None (default), then draw `X.shape[0]` samples.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max(round(n_samples * max_samples), 1)` samples. Thus,
            `max_samples` should be in the interval `(0.0, 1.0]`.

        :param sample_with_replacement: If True, samples are drawn with replacement.
        :param streaming: If True, uses Polars' lazy evaluation.        
        :param criterion: The function to measure the quality of a split.
        :param categorical_columns: List of categorical columns to be used in the model.
            If None, all columns are treated as numerical.
        :param max_depth: Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.        
        """
        self.seed = seed
        self.streaming = streaming
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.categorical_columns = categorical_columns
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.sample_with_replacement = sample_with_replacement
        self.trees: list[DecisionTreeClassifier] = []
        if self.seed is not None:
            pl.set_random_seed(self.seed)
        self.fitted_ = False

    def __repr__(self) -> str:
        return (
            f"RandomForestClassifier(n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, "
            f"sample_with_replacement={self.sample_with_replacement}, "
            f"streaming={self.streaming}, "
            f"criterion={self.criterion}, "
            f"categorical_columns={self.categorical_columns}, "
            f"max_depth={self.max_depth})"
        )

    def fit(self,  data: pl.DataFrame | pl.LazyFrame, target_name: str) -> None:
        """
        Fit the random forest model to the training data.

        :param data: Training data as a Polars DataFrame or LazyFrame.
        :param target_name: Name of the target column in the DataFrame.
        """
        breakpoint()
        for _ in range(self.n_estimators):
            # Sample data with replacement
            sampled_data = data.sample(
                n=self.max_samples if isinstance(self.max_samples, int) else None,
                fraction=self.max_samples if isinstance(self.max_samples, float) else None, 
                shuffle=True,
                with_replacement=self.sample_with_replacement,
            )

            tree = DecisionTreeClassifier(
                streaming=self.streaming,
                criterion=self.criterion,
                max_depth=self.max_depth,
                categorical_columns=self.categorical_columns,
            )
            tree.fit(sampled_data, target_name)
            self.trees.append(tree)
        self.fitted_ = True

    def predict(self, data: Iterable[dict]) -> list[int | float]:
        """
        Predict method.

        :param data: list of dicts
        :return: List of predicted target values.
        """
        if not self.fitted_:
            raise ValueError("The model has not been fitted yet.")

        # Collect predictions from each tree
        breakpoint()
        raw_predictions = [tree.predict(data) for tree in self.trees]
        tree_predictions = pl.DataFrame(raw_predictions)
        aggregated_predictions = tree_predictions.select(pl.mean_horizontal()).to_list()
            
        return aggregated_predictions
    
    def predict_many(self, data: pl.DataFrame | pl.LazyFrame) -> pl.Series:
        """
        Predict the class labels for the input data.

        :param data: Input data as a Polars DataFrame or LazyFrame.
        :return: A Polars Series containing the predicted class labels.
        """
        if not self.fitted_:
            raise ValueError("The model has not been fitted yet.")

        # Collect predictions from each tree
        breakpoint()
        raw_predictions = [tree.predict_many(data) for tree in self.trees]
        tree_predictions = pl.DataFrame(raw_predictions)
        aggregated_predictions = tree_predictions.select(pl.mean_horizontal()).to_list()
            
        return aggregated_predictions