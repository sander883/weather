"""Unit tests for hyperparameter tuning module."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.models.hyperparameter_tuning import HyperparameterTuner


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def temp_model_dir():
    """Create temporary model directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestHyperparameterTuner:
    """Test HyperparameterTuner class."""

    def test_init(self, temp_model_dir):
        """Test initialization."""
        tuner = HyperparameterTuner(model_dir=temp_model_dir)
        assert tuner.model_dir == Path(temp_model_dir)
        assert tuner.seed == 42
        assert tuner.best_params is None

    def test_tune_xgboost_grid(self, sample_data, temp_model_dir):
        """Test grid search tuning."""
        X, y = sample_data
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100]
        }

        results = tuner.tune_xgboost_grid(X, y, param_grid=param_grid, cv=3)

        assert 'best_params' in results
        assert 'best_score' in results
        assert results['method'] == 'grid_search'
        assert tuner.best_params is not None
        assert tuner.best_score is not None

    def test_tune_xgboost_random(self, sample_data, temp_model_dir):
        """Test randomized search tuning."""
        X, y = sample_data
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        results = tuner.tune_xgboost_random(X, y, n_iter=10, cv=3)

        assert 'best_params' in results
        assert 'best_score' in results
        assert results['method'] == 'random_search'
        assert tuner.best_params is not None

    def test_evaluate_params(self, sample_data, temp_model_dir):
        """Test parameter evaluation."""
        X, y = sample_data
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        params = {
            'max_depth': 5,
            'n_estimators': 100,
            'learning_rate': 0.05
        }

        results = tuner.evaluate_params(X, y, params, cv=3)

        assert 'cv_mean' in results
        assert 'cv_std' in results
        assert 'cv_scores' in results
        assert len(results['cv_scores']) == 3
        assert 0 <= results['cv_mean'] <= 1

    def test_compare_params(self, sample_data, temp_model_dir):
        """Test comparing multiple parameter configurations."""
        X, y = sample_data
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        configs = [
            {'max_depth': 3, 'n_estimators': 50, 'learning_rate': 0.01},
            {'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.05},
            {'max_depth': 7, 'n_estimators': 150, 'learning_rate': 0.1}
        ]

        results = tuner.compare_params(X, y, configs, cv=2)

        assert len(results) == 3
        # Results should be sorted by score (descending)
        assert results[0]['cv_mean'] >= results[1]['cv_mean']
        assert results[1]['cv_mean'] >= results[2]['cv_mean']

    def test_get_best_params(self, sample_data, temp_model_dir):
        """Test getting best parameters."""
        X, y = sample_data
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        # Before tuning
        assert tuner.get_best_params() is None

        # After tuning
        tuner.tune_xgboost_random(X, y, n_iter=5, cv=2)
        best_params = tuner.get_best_params()
        assert best_params is not None
        assert isinstance(best_params, dict)

    def test_get_best_score(self, sample_data, temp_model_dir):
        """Test getting best score."""
        X, y = sample_data
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        # Before tuning
        assert tuner.get_best_score() is None

        # After tuning
        tuner.tune_xgboost_random(X, y, n_iter=5, cv=2)
        best_score = tuner.get_best_score()
        assert best_score is not None
        assert 0 <= best_score <= 1

    def test_save_tuning_results(self, sample_data, temp_model_dir):
        """Test saving tuning results."""
        X, y = sample_data
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        tuner.tune_xgboost_random(X, y, n_iter=3, cv=2)
        filepath = tuner.save_tuning_results('test_results.json')

        assert Path(filepath).exists()
        assert 'test_results.json' in filepath

    def test_load_tuning_results(self, sample_data, temp_model_dir):
        """Test loading tuning results."""
        X, y = sample_data
        tuner1 = HyperparameterTuner(model_dir=temp_model_dir)

        tuner1.tune_xgboost_random(X, y, n_iter=3, cv=2)
        filepath = tuner1.save_tuning_results('test_load.json')

        # Load with new tuner
        tuner2 = HyperparameterTuner(model_dir=temp_model_dir)
        success = tuner2.load_tuning_results(filepath)

        assert success
        assert len(tuner2.tuning_history) > 0

    def test_create_param_combinations(self, temp_model_dir):
        """Test creating parameter combinations."""
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        base_params = {'random_state': 42, 'verbosity': 0}
        variations = {
            'max_depth': [3, 5],
            'n_estimators': [100, 200]
        }

        combinations = tuner.create_param_combinations(base_params, variations)

        assert len(combinations) == 4  # 2 * 2
        # Each combination should have base params
        for combo in combinations:
            assert combo['random_state'] == 42
            assert combo['verbosity'] == 0


class TestHyperparameterTunerEdgeCases:
    """Test edge cases for hyperparameter tuning."""

    def test_tune_with_small_data(self, temp_model_dir):
        """Test tuning with minimal data."""
        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, 10)
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        results = tuner.tune_xgboost_random(X, y, n_iter=2, cv=2)

        assert 'best_params' in results

    def test_tune_with_single_feature(self, temp_model_dir):
        """Test tuning with single feature."""
        X = np.random.randn(50, 1)
        y = (X[:, 0] > 0).astype(int)
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        results = tuner.tune_xgboost_random(X, y, n_iter=3, cv=2)

        assert 'best_params' in results

    def test_tune_with_imbalanced_classes(self, temp_model_dir):
        """Test tuning with imbalanced class distribution."""
        X = np.random.randn(100, 5)
        y = np.array([0] * 90 + [1] * 10)
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        results = tuner.tune_xgboost_random(X, y, n_iter=3, cv=2)

        assert 'best_params' in results

    def test_multiple_tuning_runs(self, sample_data, temp_model_dir):
        """Test multiple tuning runs."""
        X, y = sample_data
        tuner = HyperparameterTuner(model_dir=temp_model_dir)

        # First tuning run
        tuner.tune_xgboost_random(X, y, n_iter=3, cv=2)
        history_len_1 = len(tuner.tuning_history)

        # Second tuning run
        tuner.tune_xgboost_random(X, y, n_iter=3, cv=2)
        history_len_2 = len(tuner.tuning_history)

        assert history_len_2 > history_len_1
        assert history_len_2 == 2
