import pandas as pd
import numpy as np
import sklearn.metrics.pairwise
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    Normalizer,
    MaxAbsScaler,
    PowerTransformer,
)


__all__ = ["calculate_scaled_cosine_similarity", "split_data_on_column"]


def calculate_scaled_cosine_similarity(
    data: pd.DataFrame, scale_method: str = "minmax"
) -> pd.DataFrame:
    """
    Calculate the scaled cosine similarity matrix for the given data.

    Parameters:
    - data: The input data to calculate the cosine similarity matrix for.
    - scale_method: The method to scale the data. Default is 'minmax'.

    Returns:
    - cosine_similarity_df: The cosine similarity matrix, with the maximum similarity index for each row.

    Raises:
    - ValueError: If the specified scale method is not recognized.
    - ValueError: If the input data is empty.

    """

    if data.empty:
        raise ValueError(
            "The input data is empty. Please provide a non-empty DataFrame."
        )

    # Scale the data according to the specified method
    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "l2": Normalizer(norm="l2"),
        "l1": Normalizer(norm="l1"),
        "maxabs": MaxAbsScaler(),
        "yeojohnson": PowerTransformer(method="yeo-johnson"),
    }

    scaler = scalers.get(scale_method.lower())

    if not scaler:
        raise ValueError(
            f"Scaling method '{scale_method}' is not recognized. "
            "Choose from 'minmax', 'standard', 'robust', 'l2', 'l1', 'maxabs', or 'yeojohnson'."
        )

    normalized_data = scaler.fit_transform(data)

    normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

    cosine_similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(normalized_df)

    # Set diagonal to 0 from 1
    np.fill_diagonal(cosine_similarity_matrix, 0)

    # Convert to DataFrame and add max_index column
    cosine_similarity_df = pd.DataFrame(
        cosine_similarity_matrix, index=normalized_df.index, columns=normalized_df.index
    )
    cosine_similarity_df["max_index"] = cosine_similarity_df.idxmax(axis=1)

    return cosine_similarity_df


def split_data_on_column(df, column_name, gap=0.9):
    """
    Split the data in a DataFrame based on a specified column.

    Args:
        df (pandas.DataFrame): The DataFrame to split.
        column_name (str): The name of the column to split on.
        gap (float, optional): The percentage of data to exclude from both ends. Defaults to 0.9.

    Returns:
        tuple: A tuple containing the lower subset and upper subset of the data.

    """

    if df.empty:
        raise ValueError(
            "The input data is empty. Please provide a non-empty DataFrame."
        )

    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")

    gap_length = (1 - gap) / 2
    lower_quantile = df[column_name].quantile(0 + gap_length)
    upper_quantile = df[column_name].quantile(1 - gap_length)

    print(lower_quantile)
    print(upper_quantile)

    lower_subset = df[df[column_name] <= lower_quantile]
    middle_subset = df[
        (df[column_name] > lower_quantile) & (df[column_name] <= upper_quantile)
    ]
    upper_subset = df[df[column_name] >= upper_quantile]

    return lower_subset, upper_subset
