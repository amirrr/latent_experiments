import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    Normalizer,
    MaxAbsScaler,
    PowerTransformer,
)


def calculate_scaled_cosine_similarity(data, scale_method="minmax"):
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

    cosine_similarity_matrix = cosine_similarity(normalized_df)

    # Set diagonal to 0 from 1
    np.fill_diagonal(cosine_similarity_matrix, 0)

    # Convert to DataFrame and add max_index column
    cosine_similarity_df = pd.DataFrame(
        cosine_similarity_matrix, index=normalized_df.index, columns=normalized_df.index
    )
    cosine_similarity_df["max_index"] = cosine_similarity_df.idxmax(axis=1)

    return cosine_similarity_df
