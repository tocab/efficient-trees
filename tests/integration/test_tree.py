import itertools

import polars as pl
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from efficient_trees.enums import Criterion
from efficient_trees.tree import DecisionTreeClassifier


@pytest.fixture
def data(request):
    iris = load_iris()
    X, y = iris.data, iris.target # type: ignore

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # data
    df_train = pl.DataFrame(X_train, schema=iris.feature_names).with_columns(target=pl.Series(y_train)) # type: ignore
    df_test = pl.DataFrame(X_test, schema=iris.feature_names).with_columns(target=pl.Series(y_test)) # type: ignore
    if request.param == "str":
        df_train = df_train.with_columns(
            pl.col("target").cast(pl.String, strict=False).replace({"0": "setosa", "1": "versicolor", "2": "virginica"})
        )
        df_test = df_test.with_columns(
            pl.col("target").cast(pl.String, strict=False).replace({"0": "setosa", "1": "versicolor", "2": "virginica"})
        )

    # add dummy categorical feature
    cycle = itertools.cycle([1, 2, 3])
    df_train = df_train.with_columns(categorical_feature=pl.Series([cycle.__next__() for _ in range(len(df_train))]))
    df_test = df_test.with_columns(categorical_feature=pl.Series([cycle.__next__() for _ in range(len(df_test))]))

    # fmt: off
    predictions = {
        "train_predictions_entropy": [
            1, 2, 2, 1, 2, 1, 2, 1, 0, 2, 1, 0, 0, 1, 1, 2, 1, 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 2, 2, 1, 1, 2, 1, 0, 1, 2,
            0, 0, 1, 1, 0, 2, 0, 0, 2, 1, 2, 2, 2, 2, 1, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 2, 2, 0, 1, 1, 2, 1, 2, 0, 2, 1,
            2, 1, 1, 1, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 2, 0, 1, 2, 2, 1, 2, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2
        ],
        "test_predictions_entropy": [
            1, 0, 2, 1, 1, 0, 1, 2, 1, 1, 2, 0, 0, 0, 0, 1, 2, 1, 1, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 0, 0, 2,
            1, 0, 0, 0, 2, 2, 1, 0, 0
        ],
        "train_predictions_gini": [
            1, 2, 2, 1, 2, 1, 2, 1, 0, 2, 1, 0, 0, 1, 1, 2, 1, 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 2, 2, 1, 1, 2, 1, 0, 1, 2,
            0, 0, 1, 2, 0, 2, 0, 0, 2, 1, 2, 2, 2, 2, 1, 0, 0, 2, 2, 0, 0, 0, 1, 2, 0, 2, 2, 0, 1, 1, 2, 1, 2, 0, 2, 1,
            2, 1, 1, 1, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 2, 0, 1, 2, 2, 1, 2, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2
        ],
        "test_predictions_gini": [
            1, 0, 2, 1, 1, 0, 1, 2, 1, 1, 2, 0, 0, 0, 0, 1, 2, 1, 1, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 0, 0, 2,
            1, 0, 0, 0, 2, 2, 1, 0, 0
        ],
    }
    if request.param == "str":
        for prediction in predictions:
            predictions[prediction] = pl.Series(predictions[prediction]).cast(pl.String, strict=False).replace(
                {"0": "setosa", "1": "versicolor", "2": "virginica"}
            ).to_list()
    # fmt: on

    return df_train, df_test, predictions


@pytest.mark.parametrize("data", ["int", "str"], indirect=True)
@pytest.mark.parametrize("is_lazy", [True, False])
@pytest.mark.parametrize("use_categorical_feature", [True, False])
@pytest.mark.parametrize("criterion", [Criterion.ENTROPY, Criterion.GINI])
def test_tree(data, is_lazy, use_categorical_feature, criterion):
    df_train, df_test, predictions = data
    if is_lazy:
        df_train = df_train.lazy()
        df_test = df_test.lazy()

    if not use_categorical_feature:
        df_train = df_train.drop("categorical_feature")
        df_test = df_test.drop("categorical_feature")

    decision_tree_classifier = DecisionTreeClassifier(max_depth=4, criterion=criterion)
    decision_tree_classifier.fit(df_train, "target")

    train_predict_many = decision_tree_classifier.predict_many(df_train)
    test_predict_many = decision_tree_classifier.predict_many(df_test)

    if is_lazy:
        df_train = df_train.collect()
        df_test = df_test.collect()

    train_predict = decision_tree_classifier.predict(df_train.iter_rows(named=True))
    test_predict = decision_tree_classifier.predict(df_test.iter_rows(named=True))

    assert train_predict == train_predict_many
    assert test_predict == test_predict_many

    if criterion == Criterion.ENTROPY:
        assert train_predict == predictions["train_predictions_entropy"]
        assert test_predict == predictions["test_predictions_entropy"]
    elif criterion == Criterion.GINI:
        assert train_predict == predictions["train_predictions_gini"]
        assert test_predict == predictions["test_predictions_gini"]
