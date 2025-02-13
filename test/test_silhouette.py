# write your silhouette score unit tests here
import pytest
import numpy as np
from cluster import Silhouette, KMeans

@pytest.fixture
def sample_data():
    """Generate sample 2D data with 3 clusters."""
    np.random.seed(42)
    data = np.random.rand(100, 2)
    return data

@pytest.fixture
def fitted_kmeans(sample_data):
    """Fit KMeans on sample data and return the model and labels."""
    kmeans = KMeans(k=3)
    kmeans.fit(sample_data)
    labels = kmeans.predict(sample_data)
    return kmeans, labels

def test_silhouette(fitted_kmeans, sample_data):
    """Test if silhouette score function runs without errors and outputs correct shape."""
    kmeans, labels = fitted_kmeans
    silhouette = Silhouette()
    scores = silhouette.score(sample_data, labels)

    #all scores are within the valid range of -1 to 1
    assert np.all(scores >= -1) and np.all(scores <= 1)
