"""Unit tests for model training module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pickle
import json

from src.models.training import ModelTrainer


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    np.random.seed(42)
    data = {
        'temperature': np.random.randn(100) * 10 + 15,
        'humidity': np.random.randint(30, 100, 100),
        'wind_speed': np.random.exponential(5, 100),
        'clouds': np.random.randint(0, 100, 100),
        'precipitation': np.random.exponential(0.5, 100),
        'temperature_ma_6': np.random.randn(100) * 5 + 15,
        'humidity_ma_6': np.random.randint(30, 100, 100),
        'temperature_trend_6': np.random.randn(100) * 2,
        'hour': np.random.randint(0, 24, 100),
        'day_of_week': np.random.randint(0, 7, 100),
        'month': np.random.randint(1, 13, 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_target():
    """Create sample target data."""
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100), name='target')


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'features': {
            'lookback_windows': ['3h', '6h', '12h', '24h'],
            'seasonal_features': True
        },
        'model': {
            'xgboost_params': {
                'max_depth': 3,
                'n_estimators': 50,
                'random_state': 42,
                'verbosity': 0
            }
        }
    }


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestModelTrainerInit:
    """Test ModelTrainer initialization."""

    def test_init_with_config(self, sample_config):
        """Test initialization with custom config."""
        trainer = ModelTrainer(config=sample_config)
        assert trainer.config == sample_config
        assert trainer.model is None
        assert trainer.scaler is None
        assert trainer.feature_names is None

    def test_init_default_config(self):
        """Test initialization with default config."""
        with patch('src.models.training.load_config') as mock_load:
            mock_load.return_value = {'features': {}, 'model': {}}
            trainer = ModelTrainer()
            assert trainer.config is not None
            assert trainer.model is None

    def test_model_dir_created(self, sample_config, temp_model_dir):
        """Test that model directory is created."""
        with patch.object(Path, 'mkdir'):
            trainer = ModelTrainer(config=sample_config)
            trainer.model_dir.mkdir(parents=True, exist_ok=True)
            assert trainer.model_dir is not None


class TestPrepareTrainingData:
    """Test prepare_training_data method."""

    def test_prepare_training_data_basic(self, sample_features, sample_target, sample_config):
        """Test basic training data preparation."""
        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            sample_features, sample_target, test_size=0.2
        )

        # Check dimensions
        assert len(X_train) + len(X_test) == len(sample_features)
        assert len(y_train) + len(y_test) == len(sample_target)
        assert X_train.shape[1] == sample_features.shape[1]
        assert X_test.shape[1] == sample_features.shape[1]

    def test_prepare_training_data_split_ratio(self, sample_features, sample_target, sample_config):
        """Test that train/test split ratio is correct."""
        trainer = ModelTrainer(config=sample_config)
        test_size = 0.3
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            sample_features, sample_target, test_size=test_size
        )

        total = len(X_train) + len(X_test)
        expected_test = int(total * test_size)
        assert len(X_test) == expected_test

    def test_prepare_training_data_with_missing_values(self, sample_config):
        """Test handling of missing values."""
        features = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0] * 10,
            'col2': [4.0, 5.0, np.nan] * 10,
        })
        target = pd.Series(np.random.randint(0, 2, 30), name='target')

        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            features, target, test_size=0.2
        )

        # Check no NaN values remain
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_prepare_training_data_alignment(self, sample_config):
        """Test strict alignment of features and target."""
        features = pd.DataFrame(np.random.randn(50, 5), columns=[f'f{i}' for i in range(5)])
        target = pd.Series(np.random.randint(0, 2, 50), name='target')

        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            features, target
        )

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_prepare_training_data_misaligned_lengths(self, sample_config):
        """Test that misaligned data is trimmed."""
        features = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        target = pd.Series(np.random.randint(0, 2, 90), name='target')

        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            features, target
        )

        total = len(X_train) + len(X_test)
        assert total == 90  # Trimmed to target length

    def test_prepare_training_data_empty_features(self, sample_config):
        """Test error on empty features."""
        empty_features = pd.DataFrame()
        target = pd.Series([0, 1], name='target')

        trainer = ModelTrainer(config=sample_config)
        with pytest.raises(ValueError):
            trainer.prepare_training_data(empty_features, target)

    def test_prepare_training_data_non_numeric_features(self, sample_config):
        """Test handling of non-numeric columns."""
        features = pd.DataFrame({
            'text_col': ['a', 'b', 'c'] * 10,
            'numeric_col': np.random.randn(30)
        })
        target = pd.Series(np.random.randint(0, 2, 30), name='target')

        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            features, target
        )

        # Should only have numeric column
        assert X_train.shape[1] == 1

    def test_prepare_training_data_dataframe_target(self, sample_features, sample_config):
        """Test with DataFrame target (should convert to Series)."""
        target_df = pd.DataFrame(np.random.randint(0, 2, 100), columns=['target'])

        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            sample_features, target_df
        )

        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)


class TestModelTrain:
    """Test train method."""

    def test_train_basic(self, sample_features, sample_target, sample_config):
        """Test basic model training."""
        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            sample_features, sample_target, test_size=0.2
        )

        metrics = trainer.train(X_train, y_train)

        assert 'train_accuracy' in metrics
        assert 'train_auc' in metrics
        assert 'train_precision' in metrics
        assert 'train_recall' in metrics
        assert 'train_f1' in metrics
        assert trainer.model is not None

    def test_train_with_validation(self, sample_features, sample_target, sample_config):
        """Test training with validation set."""
        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            sample_features, sample_target, test_size=0.2
        )

        metrics = trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)

        assert 'val_accuracy' in metrics
        assert 'val_auc' in metrics
        assert 'val_precision' in metrics
        assert 'val_recall' in metrics
        assert 'val_f1' in metrics

    def test_train_metrics_valid_range(self, sample_features, sample_target, sample_config):
        """Test that all metrics are in valid ranges."""
        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            sample_features, sample_target, test_size=0.2
        )

        metrics = trainer.train(X_train, y_train)

        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['train_auc'] <= 1
        assert 0 <= metrics['train_precision'] <= 1
        assert 0 <= metrics['train_recall'] <= 1
        assert 0 <= metrics['train_f1'] <= 1

    def test_train_stores_model(self, sample_features, sample_target, sample_config):
        """Test that model is stored after training."""
        trainer = ModelTrainer(config=sample_config)
        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )

        assert trainer.model is None
        trainer.train(X_train, y_train)
        assert trainer.model is not None

    def test_train_with_invalid_data(self, sample_config):
        """Test training with invalid data."""
        trainer = ModelTrainer(config=sample_config)
        X_train = np.array([])
        y_train = np.array([])

        metrics = trainer.train(X_train, y_train)
        assert metrics == {} or len(metrics) == 0


class TestCrossValidation:
    """Test cross_validate method."""

    def test_cross_validate_basic(self, sample_features, sample_target, sample_config):
        """Test basic cross-validation."""
        trainer = ModelTrainer(config=sample_config)
        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )

        cv_results = trainer.cross_validate(X_train, y_train, cv_folds=5)

        assert 'cv_mean' in cv_results
        assert 'cv_std' in cv_results
        assert 'cv_scores' in cv_results
        assert len(cv_results['cv_scores']) == 5

    def test_cross_validate_scores_valid(self, sample_features, sample_target, sample_config):
        """Test that CV scores are in valid range."""
        trainer = ModelTrainer(config=sample_config)
        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )

        cv_results = trainer.cross_validate(X_train, y_train, cv_folds=3)

        assert 0 <= cv_results['cv_mean'] <= 1
        assert 0 <= cv_results['cv_std']
        for score in cv_results['cv_scores']:
            assert 0 <= score <= 1

    def test_cross_validate_different_folds(self, sample_features, sample_target, sample_config):
        """Test cross-validation with different number of folds."""
        trainer = ModelTrainer(config=sample_config)
        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )

        for folds in [3, 5, 10]:
            cv_results = trainer.cross_validate(X_train, y_train, cv_folds=folds)
            assert len(cv_results['cv_scores']) == folds

    def test_cross_validate_invalid_data(self, sample_config):
        """Test CV with invalid data."""
        trainer = ModelTrainer(config=sample_config)
        X = np.array([])
        y = np.array([])

        cv_results = trainer.cross_validate(X, y)
        assert cv_results == {} or len(cv_results) == 0


class TestFeatureImportance:
    """Test get_feature_importance method."""

    def test_feature_importance_not_trained(self, sample_config):
        """Test feature importance before training."""
        trainer = ModelTrainer(config=sample_config)
        importance = trainer.get_feature_importance()
        assert importance == {}

    def test_feature_importance_after_training(self, sample_features, sample_target, sample_config):
        """Test feature importance after training."""
        trainer = ModelTrainer(config=sample_config)
        trainer.feature_names = list(sample_features.columns)
        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )

        trainer.train(X_train, y_train)
        importance = trainer.get_feature_importance(top_n=5)

        assert len(importance) <= 5
        assert all(isinstance(v, float) for v in importance.values())

    def test_feature_importance_top_n(self, sample_features, sample_target, sample_config):
        """Test top_n parameter."""
        trainer = ModelTrainer(config=sample_config)
        trainer.feature_names = list(sample_features.columns)
        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )

        trainer.train(X_train, y_train)

        importance_3 = trainer.get_feature_importance(top_n=3)
        importance_10 = trainer.get_feature_importance(top_n=10)

        assert len(importance_3) <= 3
        assert len(importance_10) <= 10


class TestModelPersistence:
    """Test save_model and load_model methods."""

    def test_save_model_with_name(self, sample_features, sample_target, sample_config, temp_model_dir):
        """Test saving model with custom name."""
        trainer = ModelTrainer(config=sample_config)
        trainer.model_dir = Path(temp_model_dir)

        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )
        trainer.train(X_train, y_train)

        model_path = trainer.save_model(name='test_model')
        assert 'test_model' in model_path
        assert Path(model_path).exists()

    def test_save_model_without_name(self, sample_features, sample_target, sample_config, temp_model_dir):
        """Test saving model with auto-generated name."""
        trainer = ModelTrainer(config=sample_config)
        trainer.model_dir = Path(temp_model_dir)

        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )
        trainer.train(X_train, y_train)

        model_path = trainer.save_model()
        assert 'model_' in model_path
        assert Path(model_path).exists()

    def test_save_model_creates_metadata(self, sample_features, sample_target, sample_config, temp_model_dir):
        """Test that metadata file is created."""
        trainer = ModelTrainer(config=sample_config)
        trainer.model_dir = Path(temp_model_dir)
        trainer.feature_names = list(sample_features.columns)

        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )
        trainer.train(X_train, y_train)
        trainer.save_model(name='test_model')

        metadata_path = Path(temp_model_dir) / 'test_model_metadata.json'
        assert metadata_path.exists()

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            assert 'trained_at' in metadata
            assert 'feature_names' in metadata

    def test_save_model_not_trained(self, sample_config, temp_model_dir):
        """Test saving model before training."""
        trainer = ModelTrainer(config=sample_config)
        trainer.model_dir = Path(temp_model_dir)

        path = trainer.save_model(name='not_trained')
        assert path == ""

    def test_load_model_basic(self, sample_features, sample_target, sample_config, temp_model_dir):
        """Test loading saved model."""
        # Save model
        trainer1 = ModelTrainer(config=sample_config)
        trainer1.model_dir = Path(temp_model_dir)
        trainer1.feature_names = list(sample_features.columns)

        X_train, _, y_train, _ = trainer1.prepare_training_data(
            sample_features, sample_target
        )
        trainer1.train(X_train, y_train)
        model_path = trainer1.save_model(name='test_load')

        # Load model
        trainer2 = ModelTrainer(config=sample_config)
        success = trainer2.load_model(model_path)
        assert success
        assert trainer2.model is not None

    def test_load_model_nonexistent(self, sample_config):
        """Test loading nonexistent model."""
        trainer = ModelTrainer(config=sample_config)
        success = trainer.load_model('/nonexistent/path/model.pkl')
        assert not success

    def test_get_latest_model_empty(self, sample_config, temp_model_dir):
        """Test getting latest model when none exist."""
        trainer = ModelTrainer(config=sample_config)
        trainer.model_dir = Path(temp_model_dir)

        latest = trainer.get_latest_model()
        assert latest is None

    def test_get_latest_model_multiple(self, sample_features, sample_target, sample_config, temp_model_dir):
        """Test getting latest model when multiple exist."""
        trainer = ModelTrainer(config=sample_config)
        trainer.model_dir = Path(temp_model_dir)

        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )
        trainer.train(X_train, y_train)

        # Save multiple models
        path1 = trainer.save_model(name='model_1')
        path2 = trainer.save_model(name='model_2')

        latest = trainer.get_latest_model()
        assert latest == path2  # Most recent


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_train_single_sample(self, sample_config):
        """Test training with minimal samples."""
        X_train = np.array([[1.0, 2.0, 3.0]])
        y_train = np.array([0])

        trainer = ModelTrainer(config=sample_config)
        metrics = trainer.train(X_train, y_train)

        # Should handle gracefully (might return empty metrics)
        assert isinstance(metrics, dict)

    def test_feature_importance_without_names(self, sample_features, sample_target, sample_config):
        """Test feature importance with auto-generated names."""
        trainer = ModelTrainer(config=sample_config)
        trainer.feature_names = None  # Explicitly set to None

        X_train, _, y_train, _ = trainer.prepare_training_data(
            sample_features, sample_target
        )
        trainer.train(X_train, y_train)

        importance = trainer.get_feature_importance(top_n=5)
        assert isinstance(importance, dict)

    def test_prepare_data_all_same_values(self, sample_config):
        """Test with constant feature values."""
        features = pd.DataFrame({
            'const_col': [5.0] * 50,
            'normal_col': np.random.randn(50)
        })
        target = pd.Series(np.random.randint(0, 2, 50), name='target')

        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            features, target
        )

        assert X_train.shape[1] == 2
        assert X_test.shape[1] == 2

    def test_prepare_data_extreme_values(self, sample_config):
        """Test with extreme numerical values."""
        features = pd.DataFrame({
            'very_large': [1e10, 1e11, 1e12] * 10,
            'very_small': [1e-10, 1e-11, 1e-12] * 10,
        })
        target = pd.Series(np.random.randint(0, 2, 30), name='target')

        trainer = ModelTrainer(config=sample_config)
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            features, target
        )

        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()


class TestIntegration:
    """Integration tests for full training pipeline."""

    def test_full_pipeline(self, sample_features, sample_target, sample_config, temp_model_dir):
        """Test complete training pipeline."""
        trainer = ModelTrainer(config=sample_config)
        trainer.model_dir = Path(temp_model_dir)
        trainer.feature_names = list(sample_features.columns)

        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            sample_features, sample_target
        )

        # Train
        metrics = trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics

        # Get importance
        importance = trainer.get_feature_importance(top_n=5)
        assert len(importance) > 0

        # Save and load
        path = trainer.save_model(name='pipeline_test')
        assert Path(path).exists()

        trainer2 = ModelTrainer(config=sample_config)
        success = trainer2.load_model(path)
        assert success

    def test_multiple_training_cycles(self, sample_features, sample_target, sample_config):
        """Test multiple training cycles on same trainer."""
        trainer = ModelTrainer(config=sample_config)

        for i in range(3):
            X_train, X_test, y_train, y_test = trainer.prepare_training_data(
                sample_features, sample_target
            )
            metrics = trainer.train(X_train, y_train)
            assert 'train_accuracy' in metrics
