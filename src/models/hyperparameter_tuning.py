"""Hyperparameter tuning for machine learning models."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """Tune hyperparameters for XGBoost models."""

    def __init__(self, model_dir: str = 'data/models', seed: int = 42):
        """
        Initialize hyperparameter tuner.

        Args:
            model_dir: Directory to save tuning results
            seed: Random seed for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.tuning_history = []
        self.best_params = None
        self.best_score = None

    def tune_xgboost_grid(self, X: np.ndarray, y: np.ndarray,
                          param_grid: Optional[Dict[str, List]] = None,
                          cv: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Tune XGBoost using grid search.

        Args:
            X: Training features
            y: Training targets
            param_grid: Parameter grid to search
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Dictionary with tuning results
        """
        if param_grid is None:
            param_grid = self._get_default_xgboost_grid()

        try:
            logger.info(f"Starting grid search with {len(param_grid)} combinations")

            base_model = xgb.XGBClassifier(
                random_state=self.seed,
                verbosity=0,
                n_jobs=-1
            )

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X, y)

            results = {
                'method': 'grid_search',
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'cv_results': grid_search.cv_results_,
                'timestamp': datetime.utcnow().isoformat(),
                'total_combinations': len(grid_search.cv_results_['params'])
            }

            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            self.tuning_history.append(results)

            logger.info(f"Grid search complete. Best score: {self.best_score:.4f}")
            return results

        except Exception as e:
            logger.error(f"Grid search failed: {e}")
            return {}

    def tune_xgboost_random(self, X: np.ndarray, y: np.ndarray,
                           param_dist: Optional[Dict[str, List]] = None,
                           n_iter: int = 50, cv: int = 5,
                           scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Tune XGBoost using randomized search.

        Args:
            X: Training features
            y: Training targets
            param_dist: Parameter distribution to search
            n_iter: Number of iterations
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Dictionary with tuning results
        """
        if param_dist is None:
            param_dist = self._get_default_xgboost_distribution()

        try:
            logger.info(f"Starting randomized search with {n_iter} iterations")

            base_model = xgb.XGBClassifier(
                random_state=self.seed,
                verbosity=0,
                n_jobs=-1
            )

            random_search = RandomizedSearchCV(
                base_model,
                param_dist,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=self.seed,
                n_jobs=-1,
                verbose=1
            )

            random_search.fit(X, y)

            results = {
                'method': 'random_search',
                'best_params': random_search.best_params_,
                'best_score': float(random_search.best_score_),
                'cv_results': random_search.cv_results_,
                'timestamp': datetime.utcnow().isoformat(),
                'iterations': n_iter
            }

            self.best_params = random_search.best_params_
            self.best_score = random_search.best_score_
            self.tuning_history.append(results)

            logger.info(f"Random search complete. Best score: {self.best_score:.4f}")
            return results

        except Exception as e:
            logger.error(f"Random search failed: {e}")
            return {}

    def evaluate_params(self, X: np.ndarray, y: np.ndarray,
                       params: Dict[str, Any], cv: int = 5,
                       scoring: str = 'roc_auc') -> Dict[str, float]:
        """
        Evaluate a specific set of hyperparameters.

        Args:
            X: Training features
            y: Training targets
            params: Hyperparameters to evaluate
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Dictionary with evaluation results
        """
        try:
            model = xgb.XGBClassifier(**params, random_state=self.seed, verbosity=0)

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            results = {
                'params': params,
                'cv_mean': float(scores.mean()),
                'cv_std': float(scores.std()),
                'cv_scores': scores.tolist(),
                'timestamp': datetime.utcnow().isoformat()
            }

            logger.info(f"Params evaluation: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
            return results

        except Exception as e:
            logger.error(f"Parameter evaluation failed: {e}")
            return {}

    def compare_params(self, X: np.ndarray, y: np.ndarray,
                      param_configs: List[Dict[str, Any]],
                      cv: int = 5, scoring: str = 'roc_auc') -> List[Dict[str, Any]]:
        """
        Compare multiple hyperparameter configurations.

        Args:
            X: Training features
            y: Training targets
            param_configs: List of parameter configurations
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            List of evaluation results, sorted by score
        """
        results = []
        for i, params in enumerate(param_configs):
            logger.info(f"Evaluating config {i+1}/{len(param_configs)}")
            eval_result = self.evaluate_params(X, y, params, cv, scoring)
            if eval_result:
                results.append(eval_result)

        # Sort by score descending
        results.sort(key=lambda x: x['cv_mean'], reverse=True)

        logger.info(f"Top 3 configurations:")
        for i, result in enumerate(results[:3]):
            logger.info(f"  {i+1}. Score: {result['cv_mean']:.4f}, Params: {result['params']}")

        return results

    def save_tuning_results(self, filename: Optional[str] = None) -> str:
        """
        Save tuning history to file.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.model_dir / filename

        try:
            # Convert numpy types to native Python types for JSON serialization
            serializable_history = []
            for record in self.tuning_history:
                record_copy = record.copy()
                if 'cv_results' in record_copy:
                    # Simplify cv_results for storage
                    cv_results = record_copy.pop('cv_results')
                    record_copy['cv_results_keys'] = list(cv_results.keys())
                serializable_history.append(record_copy)

            with open(filepath, 'w') as f:
                json.dump(serializable_history, f, indent=2)

            logger.info(f"Tuning results saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save tuning results: {e}")
            return ""

    def load_tuning_results(self, filepath: str) -> bool:
        """
        Load tuning history from file.

        Args:
            filepath: Path to tuning results file

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                self.tuning_history = json.load(f)

            logger.info(f"Loaded {len(self.tuning_history)} tuning records")
            return True

        except Exception as e:
            logger.error(f"Failed to load tuning results: {e}")
            return False

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters found during tuning."""
        return self.best_params

    def get_best_score(self) -> Optional[float]:
        """Get best cross-validation score found during tuning."""
        return self.best_score

    def _get_default_xgboost_grid(self) -> Dict[str, List]:
        """Get default grid for XGBoost grid search."""
        return {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

    def _get_default_xgboost_distribution(self) -> Dict[str, List]:
        """Get default distribution for XGBoost random search."""
        from scipy.stats import randint, uniform

        return {
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'n_estimators': randint(50, 500),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'gamma': uniform(0, 5),
            'min_child_weight': randint(1, 10)
        }

    def create_param_combinations(self, base_params: Dict[str, Any],
                                 variations: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Create parameter combinations from base params and variations.

        Args:
            base_params: Base parameters
            variations: Dictionary of parameter variations to apply

        Returns:
            List of parameter configurations
        """
        combinations = []
        keys = list(variations.keys())
        values = list(variations.values())

        from itertools import product
        for combo in product(*values):
            params = base_params.copy()
            for key, value in zip(keys, combo):
                params[key] = value
            combinations.append(params)

        logger.info(f"Created {len(combinations)} parameter combinations")
        return combinations
