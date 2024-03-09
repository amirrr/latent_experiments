import pandas as pd
import numpy as np

from .latent_experiments import calculate_scaled_cosine_similarity
import pytest


def test_calculate_scaled_cosine_similarity():
    # Test case 1: Test with sample data
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    expected_result = pd.DataFrame(
        [[0.0, 0.0, 0.0, 0], [0.0, 0.0, 1.0, 2], [0.0, 1.0, 0.0, 1]],
        columns=[0, 1, 2, "max_index"],
    )

    pd.testing.assert_frame_equal(
        calculate_scaled_cosine_similarity(data), expected_result, check_names=False
    )

    # Test case 2: Test with empty data
    empty_data = pd.DataFrame()

    with pytest.raises(ValueError) as excinfo:
        calculate_scaled_cosine_similarity(empty_data)

    assert "The input data is empty. Please provide a non-empty DataFrame." in str(
        excinfo.value
    )

    # Test case 3: Test with single row data
    data = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    expected_result = pd.DataFrame([[0.0, 0]], columns=[0, "max_index"])

    print(calculate_scaled_cosine_similarity(data))
    pd.testing.assert_frame_equal(
        calculate_scaled_cosine_similarity(data), expected_result, check_names=False
    )

    # Test case 4: Test with invalid scale method
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    scale_method = "invalid"
    with pytest.raises(ValueError):
        calculate_scaled_cosine_similarity(data, scale_method)

    # Test case 5: Test with large data
    data = pd.DataFrame(np.random.rand(100, 100))
    assert calculate_scaled_cosine_similarity(data).shape == (100, 101)
