from typing import Dict, Tuple
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the rows with missing values.
    Drops the education column since education-num acts as the label encoded equivalent.

    Args:
        df: Raw data.

    Returns: Cleaned data.

    """
    df = df.replace("?", np.NaN)
    df = df.drop("education", axis=1)
    return df.dropna(axis=0)


def encode(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Encodes the categorical variables.

    Args:
        df: Cleaned data.
        parameters: Parameters defined in parameters.yml.

    Returns:
        Encoded data.
    """
    df = pd.get_dummies(df, columns=parameters["features"]["categorical"])
    df[parameters["target"]] = df[parameters["target"]].astype("category").cat.codes
    return df


def split_data(df: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        df: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """
    X = df.drop(parameters["target"], axis=1)
    y = df[parameters["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for salary.

    Returns:
        Trained model.
    """
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for salary.
    """
    y_pred = regressor.predict(X_test)
    score = f1_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a f1 score of %.3f on test data.", score)