import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, MaxAbsScaler, PowerTransformer
from sklearn.metrics.pairwise import cosine_similarity

from latent_experiments import calculate_scaled_cosine_similarity

def test_calculate_scaled_cosine_similarity():
    # Test case 1: Test with sample data
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    expected_result = pd.DataFrame([[0, 0.97463185, 0.97463185],
                                    [0.97463185, 0, 0.97463185],
                                    [0.97463185, 0.97463185, 0]], 
                                   index=['A', 'B', 'C'], columns=['A', 'B', 'C'])
    assert calculate_scaled_cosine_similarity(data).equals(expected_result)

    # Test case 2: Test with empty data
    data = pd.DataFrame()
    expected_result = pd.DataFrame()
    assert calculate_scaled_cosine_similarity(data).equals(expected_result)

    # Test case 3: Test with single row data
    data = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
    expected_result = pd.DataFrame([[0]], index=['A'], columns=['A'])
    assert calculate_scaled_cosine_similarity(data).equals(expected_result)

    # Test case 4: Test with invalid scale method
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    scale_method = 'invalid'
    with pytest.raises(ValueError):
        calculate_scaled_cosine_similarity(data, scale_method)

    # Test case 5: Test with large data
    data = pd.DataFrame(np.random.rand(100, 100))
    assert calculate_scaled_cosine_similarity(data).shape == (100, 101)