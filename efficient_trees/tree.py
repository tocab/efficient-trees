"""
This module defines a `DecisionTreeClassifier` that implements a decision tree classifier using the Polars library.

The class is designed to handle both numerical and categorical features, and can optionally
use lazy evaluation and streaming capabilities of Polars.
"""

import pickle
from collections.abc import Iterable

import polars as pl

from efficient_trees.enums import Criterion


class DecisionTreeClassifier:
    """
    A decision tree classifier using Polars as backend.
    """

    def __init__(
        self,
        streaming: bool = False,
        max_depth: int | None = None,
        categorical_columns: list[str] | None = None,
        criterion: Criterion = Criterion.ENTROPY,
    ):
        """
        Init method.

        :param streaming: Boolean flag to enable Polars streaming capabilities.
        :param max_depth: Maximum depth of the decision tree.
        """
        self.max_depth = max_depth
        self.streaming = streaming
        self.categorical_columns = categorical_columns
        self.categorical_mappings = None
        self.tree = None
        self.criterion = criterion
        self.fitted_ = False

    def __repr__(self) -> str:
        return (
            f"DecisionTreeClassifier(max_depth={self.max_depth}, streaming={self.streaming}, "
            f"categorical_columns={self.categorical_columns}, criterion={self.criterion})"
        )

    def save_model(self, path: str) -> None:
        """
        Save the model to a file.

        :param path: Path to save the model.
        """
        # Save as pickle
        with open(path, "wb") as f:
            pickle.dump(
                {"tree": self.tree, "categorical_mappings": self.categorical_mappings},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load_model(self, path: str) -> None:
        """
        Load a model from a file.

        :param path: Path to the saved model.
        """
        # Load as pickle
        with open(path, "rb") as f:
            loaded = pickle.load(f)
            self.tree = loaded["tree"]
            self.categorical_mappings = loaded["categorical_mappings"]

    def apply_categorical_mappings(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """
        Apply categorical mappings on input frame.

        :param data: Polars DataFrame or LazyFrame with categorical columns.

        :return: Polars DataFrame or LazyFrame with mapped categorical columns
        """
        return data.with_columns(
            [pl.col(col).replace(self.categorical_mappings[col]).cast(pl.UInt32) for col in self.categorical_columns]  # type: ignore
        )

    def fit(self, data: pl.DataFrame | pl.LazyFrame, target_name: str) -> None:
        """
        Fit method to train the decision tree.

        :param data: Polars DataFrame or LazyFrame containing the training data.
        :param target_name: Name of the target column
        """
        columns = data.collect_schema().names()
        feature_names = [col for col in columns if col != target_name]

        # Shrink dtypes
        data = data.select(pl.all().shrink_dtype())

        # Prepare categorical columns with target encoding
        if self.categorical_columns:
            categorical_mappings = {}
            for categorical_column in self.categorical_columns:
                categorical_mappings[categorical_column] = {
                    value: index
                    for index, value in enumerate(
                        data.lazy()
                        .group_by(categorical_column)
                        .agg(pl.col(target_name).mean().alias("avg"))
                        .sort("avg")
                        .collect(streaming=self.streaming)[categorical_column]  # type: ignore
                    )
                }

            self.categorical_mappings = categorical_mappings
            data = self.apply_categorical_mappings(data)

        unique_targets = data.select(target_name).unique()
        if isinstance(unique_targets, pl.LazyFrame):
            unique_targets = unique_targets.collect(streaming=self.streaming)  # type: ignore
        unique_targets = unique_targets[target_name].to_list()

        self.tree = self._build_tree(data, feature_names, target_name, unique_targets, depth=0)
        self.fitted_ = True

    def predict_many(self, data: pl.DataFrame | pl.LazyFrame) -> list[int | float]:
        """
        Predict method.

        :param data: Polars DataFrame or LazyFrame.
        :return: List of predicted target values.
        """
        if not self.fitted_:
            raise ValueError("The model has not been fitted yet.")
        if self.categorical_mappings:
            data = self.apply_categorical_mappings(data)

        def _predict_many(node: dict, temp_data: pl.DataFrame):
            if node["type"] == "node":
                left = _predict_many(node["left"], temp_data.filter(pl.col(node["feature"]) <= node["threshold"]))
                right = _predict_many(node["right"], temp_data.filter(pl.col(node["feature"]) > node["threshold"]))
                return pl.concat([left, right], how="diagonal_relaxed")
            else:
                return temp_data.select(pl.col("temp_prediction_index"), pl.lit(node["value"]).alias("prediction"))

        data = data.with_row_index("temp_prediction_index")
        predictions = _predict_many(self.tree, data).sort("temp_prediction_index").select(pl.col("prediction"))  # type: ignore

        # Convert predictions to a list
        if isinstance(predictions, pl.LazyFrame):
            # Despite the execution plans says there is no streaming, using streaming here significantly
            # increases the performance and decreases the memory food print.
            predictions = predictions.collect(streaming=True)  # type: ignore

        predictions = predictions["prediction"].to_list()
        return predictions

    def predict(self, data: Iterable[dict]) -> list[int | float]:
        """
        Predict method.

        :param data: list of dicts
        :return: List of predicted target values.
        """
        if not self.fitted_:
            raise ValueError("The model has not been fitted yet.")

        def _predict_sample(node, sample):
            if node["type"] == "leaf":
                return node["value"]
            if sample[node["feature"]] <= node["threshold"]:
                return _predict_sample(node["left"], sample)
            else:
                return _predict_sample(node["right"], sample)

        predictions = [_predict_sample(self.tree, sample) for sample in data]
        return predictions

    def get_majority_class(self, df: pl.DataFrame | pl.LazyFrame, target_name: str) -> str:
        """
        Returns the majority class of a dataframe.

        :param df: The dataframe to evaluate.
        :param target_name: Name of the target column.

        :return: majority class.
        """
        majority_class = df.group_by(target_name).len().filter(pl.col("len") == pl.col("len").max()).select(target_name)
        if isinstance(majority_class, pl.LazyFrame):
            majority_class = majority_class.collect(streaming=self.streaming)  # type: ignore
        return majority_class[target_name][0]

    def _build_tree(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        feature_names: list[str],
        target_name: str,
        unique_targets: list[int],
        depth: int,
    ) -> dict:
        """
        Builds the decision tree recursively.

        If max_depth is reached, returns a leaf node with the majority class.
        Otherwise, finds the best split and creates internal nodes for left and right children.

        :param data: The dataframe to evaluate.
        :param feature_names: Name of the feature columns.
        :param target_name: Name of the target column.
        :param unique_targets: unique target values.
        :param depth: The current depth of the tree.

        :return: A dictionary representing the node.
        """
        if self.max_depth is not None and depth >= self.max_depth:
            return {"type": "leaf", "value": self.get_majority_class(data, target_name)}

        # Pre-define expressions for criterion
        if self.criterion == Criterion.ENTROPY:
            criterion_expressions = {
                direction: (
                    -1
                    * pl.sum_horizontal(
                        [
                            (
                                pl.col(f"{direction}_proportion_class_{target_value}")
                                * pl.col(f"{direction}_proportion_class_{target_value}").log(base=2)
                            ).fill_nan(0.0)
                            for target_value in unique_targets
                        ]
                    )
                ).alias(f"{direction}_criterion")
                for direction in ["left", "right", "parent"]
            }
        elif self.criterion == Criterion.GINI:
            criterion_expressions = {
                direction: (
                    1.0
                    - pl.sum_horizontal(
                        [pl.col(f"{direction}_proportion_class_{target_value}") ** 2 for target_value in unique_targets]
                    )
                ).alias(f"{direction}_criterion")
                for direction in ["left", "right", "parent"]
            }

        # Make data lazy here to avoid that it is evaluated in each loop iteration.
        data = data.lazy()

        # Evaluate entropy per feature:
        information_gain_dfs = []
        for feature_name in feature_names:
            feature_data = data.select([feature_name, target_name]).filter(pl.col(feature_name).is_not_null())
            feature_data = feature_data.rename({feature_name: "feature_value"})

            # No streaming (yet)
            information_gain_df = (
                feature_data.group_by("feature_value")
                .agg(
                    [
                        pl.col(target_name)
                        .filter(pl.col(target_name) == target_value)
                        .len()
                        .alias(f"class_{target_value}_count")
                        for target_value in unique_targets
                    ]
                    + [pl.col(target_name).len().alias("count_examples")]
                )
                .sort("feature_value")
                .select(
                    [
                        pl.col(f"class_{target_value}_count").cum_sum().alias(f"cum_sum_class_{target_value}_count")
                        for target_value in unique_targets
                    ]
                    + [
                        pl.col(f"class_{target_value}_count").sum().alias(f"sum_class_{target_value}_count")
                        for target_value in unique_targets
                    ]
                    + [
                        pl.col("count_examples").cum_sum().alias("cum_sum_count_examples"),
                        pl.col("count_examples").sum().alias("sum_count_examples"),
                    ]
                    + [
                        # From previous select
                        pl.col("feature_value"),
                    ]
                )
                .filter(
                    # At least one example available
                    pl.col("sum_count_examples") > pl.col("cum_sum_count_examples")
                )
                .select(
                    [
                        (pl.col(f"cum_sum_class_{target_value}_count") / pl.col("cum_sum_count_examples")).alias(
                            f"left_proportion_class_{target_value}"
                        )
                        for target_value in unique_targets
                    ]
                    + [
                        (
                            (pl.col(f"sum_class_{target_value}_count") - pl.col(f"cum_sum_class_{target_value}_count"))
                            / (pl.col("sum_count_examples") - pl.col("cum_sum_count_examples"))
                        ).alias(f"right_proportion_class_{target_value}")
                        for target_value in unique_targets
                    ]
                    + [
                        (pl.col(f"sum_class_{target_value}_count") / pl.col("sum_count_examples")).alias(
                            f"parent_proportion_class_{target_value}"
                        )
                        for target_value in unique_targets
                    ]
                    + [
                        # From previous select
                        pl.col("cum_sum_count_examples"),
                        pl.col("sum_count_examples"),
                        pl.col("feature_value"),
                    ]
                )
                .select(
                    criterion_expressions["left"],  # type: ignore
                    criterion_expressions["right"],  # type: ignore
                    criterion_expressions["parent"],  # type: ignore
                    # From previous select
                    pl.col("cum_sum_count_examples"),
                    pl.col("sum_count_examples"),
                    pl.col("feature_value"),
                )
                .select(
                    (
                        pl.col("cum_sum_count_examples") / pl.col("sum_count_examples") * pl.col("left_criterion")
                        + (pl.col("sum_count_examples") - pl.col("cum_sum_count_examples"))
                        / pl.col("sum_count_examples")
                        * pl.col("right_criterion")
                    ).alias("child_criterion"),
                    # From previous select
                    pl.col("parent_criterion"),
                    pl.col("feature_value"),
                )
                .select(
                    (pl.col("parent_criterion") - pl.col("child_criterion")).alias("information_gain"),
                    # From previous select
                    pl.col("parent_criterion"),
                    pl.col("feature_value"),
                )
                .filter(pl.col("information_gain").is_not_nan())
                .sort("information_gain", descending=True)
                .head(1)
                .with_columns(feature=pl.lit(feature_name))
            )
            information_gain_dfs.append(information_gain_df)

        if isinstance(information_gain_dfs[0], pl.LazyFrame):
            information_gain_dfs = pl.collect_all(information_gain_dfs, streaming=self.streaming)  # type: ignore

        information_gain_dfs = pl.concat(information_gain_dfs, how="vertical_relaxed").sort(
            "information_gain", descending=True
        )

        information_gain = 0
        if len(information_gain_dfs) > 0:
            best_params = information_gain_dfs.row(0, named=True)
            information_gain = best_params["information_gain"]

        if information_gain > 0:
            left_mask = data.select(filter=pl.col(best_params["feature"]) <= best_params["feature_value"])  # type: ignore
            if isinstance(left_mask, pl.LazyFrame):
                left_mask = left_mask.collect(streaming=self.streaming)  # type: ignore
            left_mask = left_mask["filter"]

            # Split data
            left_df = data.filter(left_mask)
            right_df = data.filter(~left_mask)

            left_subtree = self._build_tree(left_df, feature_names, target_name, unique_targets, depth + 1)
            right_subtree = self._build_tree(right_df, feature_names, target_name, unique_targets, depth + 1)

            if isinstance(data, pl.LazyFrame):
                target_distribution = (
                    data.select(target_name)
                    .collect(streaming=self.streaming)[target_name]  # type: ignore
                    .value_counts()
                    .sort(target_name)["count"]
                    .to_list()
                )
            else:
                target_distribution = data[target_name].value_counts().sort(target_name)["count"].to_list()

            return {
                "type": "node",
                "feature": best_params["feature"],  # type: ignore
                "threshold": best_params["feature_value"],  # type: ignore
                "information_gain": best_params["information_gain"],  # type: ignore
                "criterion_value": best_params["parent_criterion"],  # type: ignore
                "target_distribution": target_distribution,
                "left": left_subtree,
                "right": right_subtree,
            }
        else:
            return {"type": "leaf", "value": self.get_majority_class(data, target_name)}
