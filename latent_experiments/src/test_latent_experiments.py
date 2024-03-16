import pandas as pd
import numpy as np

from .latent_experiments import (
    calculate_scaled_cosine_similarity,
    split_data_on_column,
)
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


def test_split_data_on_column():
    # Test case 1: Test with sample data
    data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    column_name = "A"
    gap = 0.9

    lower_subset, upper_subset = split_data_on_column(data, column_name, gap)

    expected_lower_subset = pd.DataFrame({"A": [1], "B": [6]}, index=[0])
    expected_upper_subset = pd.DataFrame({"A": [5], "B": [10]}, index=[4])

    pd.testing.assert_frame_equal(lower_subset, expected_lower_subset)
    pd.testing.assert_frame_equal(upper_subset, expected_upper_subset)

    # Test case 2: Test with empty data
    empty_data = pd.DataFrame()
    column_name = "A"
    gap = 0.9

    with pytest.raises(ValueError) as excinfo:
        split_data_on_column(empty_data, column_name, gap)

    assert "The input data is empty. Please provide a non-empty DataFrame." in str(
        excinfo.value
    )

    # Test case 3: Test with non-existent column
    data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    column_name = "C"
    gap = 0.9

    with pytest.raises(KeyError) as excinfo:
        split_data_on_column(data, column_name, gap)

    assert f"Column '{column_name}' does not exist in the DataFrame." in str(
        excinfo.value
    )

    # Test case 4: Test with different gap value
    data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    column_name = "A"
    gap = 0.5

    lower_subset, upper_subset = split_data_on_column(data, column_name, gap)

    print(lower_subset)
    print(upper_subset)

    expected_lower_subset = pd.DataFrame({"A": [1, 2], "B": [6, 7]}, index=[0, 1])
    expected_upper_subset = pd.DataFrame({"A": [4, 5], "B": [9, 10]}, index=[3, 4])

    pd.testing.assert_frame_equal(lower_subset, expected_lower_subset)
    pd.testing.assert_frame_equal(upper_subset, expected_upper_subset)
