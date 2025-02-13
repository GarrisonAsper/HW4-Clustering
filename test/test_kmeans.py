# Write your k-means unit tests here
import pytest
import numpy as np
from cluster import KMeans

@pytest.fixture
def data():
    """Generate sample 2D data."""
    np.random.seed(42)
    return np.random.rand(100, 2)

@pytest.fixture
def test_model():
    """Create a KMeans instance with k=3."""
    return KMeans(k=3)

def test_fit(test_model, data):
    """Test if fit updates centroids correctly."""
    test_model.fit(data)
    centroids = test_model.get_centroids() 

    assert centroids.shape == (3, 2)  # Check if shape matches (k, features)

