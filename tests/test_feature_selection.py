"""Unit tests for feature selection module."""

import pytest
import numpy as np
import pandas as pd

from src.models.feature_selection import FeatureSelector


@pytest.fixture
def sample_data():
    """Create sample feature data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 20

    # Create features with varying importance
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Make some features important
    y = (X['feature_0'] + X['feature_1'] - X['feature_2'] > 0).astype(int)

    return X, y


class TestFeatureSelector:
    """Test FeatureSelector class."""

    def test_init(self):
        """Test initialization."""
        selector = FeatureSelector()
        assert selector.selected_features is None
        assert selector.feature_scores is None

    def test_select_correlation(self, sample_data):
        """Test correlation-based selection."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.select_correlation(X, y, n_features=10, threshold=0.01)

        assert 'selected_features' in results
        assert 'correlations' in results
        assert results['method'] == 'correlation'
        assert len(results['selected_features']) <= 10
        assert selector.selected_features is not None

    def test_select_mutual_information(self, sample_data):
        """Test mutual information selection."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.select_mutual_information(X, y, n_features=10)

        assert 'selected_features' in results
        assert 'scores' in results
        assert results['method'] == 'mutual_information'
        assert len(results['selected_features']) == 10

    def test_select_model_based_xgboost(self, sample_data):
        """Test model-based selection with XGBoost."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.select_model_based(X, y, n_features=10, model_type='xgboost')

        assert 'selected_features' in results
        assert 'importances' in results
        assert results['method'] == 'model_based_xgboost'
        assert len(results['selected_features']) == 10
        assert selector.selected_features is not None

    def test_select_model_based_random_forest(self, sample_data):
        """Test model-based selection with Random Forest."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.select_model_based(X, y, n_features=10, model_type='random_forest')

        assert 'selected_features' in results
        assert results['method'] == 'model_based_random_forest'
        assert len(results['selected_features']) == 10

    def test_select_recursive_elimination(self, sample_data):
        """Test recursive feature elimination."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.select_recursive_elimination(X, y, n_features=10)

        assert 'selected_features' in results
        assert 'ranking' in results
        assert results['method'] == 'rfe'
        assert len(results['selected_features']) == 10

    def test_select_sequential_forward(self, sample_data):
        """Test sequential forward selection."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.select_sequential(X, y, n_features=10, direction='forward')

        assert 'selected_features' in results
        assert results['method'] == 'sequential_forward'
        assert len(results['selected_features']) == 10

    def test_select_sequential_backward(self, sample_data):
        """Test sequential backward selection."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.select_sequential(X, y, n_features=10, direction='backward')

        assert 'selected_features' in results
        assert results['method'] == 'sequential_backward'
        assert len(results['selected_features']) <= 10

    def test_compare_methods(self, sample_data):
        """Test comparing all methods."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.compare_methods(X, y, n_features=10)

        # Check that all methods are present
        expected_methods = [
            'correlation', 'mutual_information', 'xgboost',
            'random_forest', 'rfe', 'sequential_forward', 'consensus'
        ]

        for method in expected_methods:
            assert method in results
            if method != 'consensus':
                assert 'selected_features' in results[method]

    def test_consensus_selection(self, sample_data):
        """Test consensus feature selection."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.compare_methods(X, y, n_features=10)
        consensus = results['consensus']

        assert 'consensus_threshold' in consensus
        assert 'feature_agreement' in consensus
        assert 'selected_features' in consensus
        # Consensus features should be agreed by multiple methods
        assert consensus['consensus_threshold'] == 3

    def test_get_selected_features(self, sample_data):
        """Test getting selected features."""
        X, y = sample_data
        selector = FeatureSelector()

        # Before selection
        assert selector.get_selected_features() is None

        # After selection
        selector.select_correlation(X, y, n_features=10)
        selected = selector.get_selected_features()
        assert selected is not None
        assert isinstance(selected, list)

    def test_get_feature_scores(self, sample_data):
        """Test getting feature scores."""
        X, y = sample_data
        selector = FeatureSelector()

        # Before selection
        assert selector.get_feature_scores() is None

        # After selection
        selector.select_correlation(X, y, n_features=10)
        scores = selector.get_feature_scores()
        assert scores is not None
        assert isinstance(scores, dict)


class TestFeatureSelectorEdgeCases:
    """Test edge cases for feature selection."""

    def test_select_more_features_than_available(self, sample_data):
        """Test selecting more features than available."""
        X, y = sample_data
        selector = FeatureSelector()

        # Request 50 features when only 20 exist
        results = selector.select_mutual_information(X, y, n_features=50)

        assert len(results['selected_features']) <= 20

    def test_select_with_single_feature(self):
        """Test selection with single feature."""
        X = pd.DataFrame(np.random.randn(100, 1), columns=['feature_0'])
        y = pd.Series((X['feature_0'] > 0).astype(int), name='target')

        selector = FeatureSelector()
        results = selector.select_correlation(X, y, n_features=10)

        assert len(results['selected_features']) == 1

    def test_select_with_perfect_features(self):
        """Test selection when features perfectly predict target."""
        X = pd.DataFrame({
            'feature_0': [0, 0, 0, 0, 1, 1, 1, 1],
            'feature_1': [0, 0, 0, 0, 1, 1, 1, 1]
        })
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name='target')

        selector = FeatureSelector()
        results = selector.select_mutual_information(X, y, n_features=2)

        assert len(results['selected_features']) == 2

    def test_select_with_random_features(self):
        """Test selection with independent random features."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 20), columns=[f'feature_{i}' for i in range(20)])
        y = pd.Series(np.random.randint(0, 2, 100), name='target')

        selector = FeatureSelector()
        results = selector.select_correlation(X, y, n_features=10, threshold=0.0)

        assert len(results['selected_features']) <= 10

    def test_multiple_selections_same_selector(self, sample_data):
        """Test multiple selections with same selector instance."""
        X, y = sample_data
        selector = FeatureSelector()

        # First selection
        results1 = selector.select_correlation(X, y, n_features=10)
        selected1 = selector.get_selected_features()

        # Second selection (overwrites)
        results2 = selector.select_mutual_information(X, y, n_features=10)
        selected2 = selector.get_selected_features()

        # Second should be different from first
        assert selected1 != selected2

    def test_method_robustness_with_small_data(self):
        """Test that methods handle small datasets gracefully."""
        X = pd.DataFrame(np.random.randn(20, 10), columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(np.random.randint(0, 2, 20), name='target')

        selector = FeatureSelector()

        # All methods should handle small data
        methods = [
            ('correlation', lambda: selector.select_correlation(X, y, n_features=5)),
            ('mutual_information', lambda: selector.select_mutual_information(X, y, n_features=5)),
            ('xgboost', lambda: selector.select_model_based(X, y, n_features=5, model_type='xgboost')),
        ]

        for method_name, method_func in methods:
            results = method_func()
            assert 'selected_features' in results
            assert len(results['selected_features']) > 0


class TestFeatureSelectorConsensus:
    """Test consensus feature selection."""

    def test_consensus_agreement_count(self, sample_data):
        """Test that consensus correctly counts feature agreements."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.compare_methods(X, y, n_features=10)
        consensus = results['consensus']

        feature_agreement = consensus['feature_agreement']

        # All features in agreement should have count >= 3
        for feature, count in feature_agreement.items():
            if feature in consensus['selected_features']:
                assert count >= 3

    def test_consensus_features_are_subset(self, sample_data):
        """Test that consensus features are subset of all selected features."""
        X, y = sample_data
        selector = FeatureSelector()

        results = selector.compare_methods(X, y, n_features=10)
        consensus_features = set(results['consensus']['selected_features'])
        all_features = set()

        for method, result in results.items():
            if method != 'consensus' and 'selected_features' in result:
                all_features.update(result['selected_features'])

        # Consensus should be subset of all
        assert consensus_features.issubset(all_features)
