"""Advanced feature selection techniques."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import xgboost as xgb
from sklearn.feature_selection import (
    RFE, SelectKBest, f_classif, mutual_info_classif,
    SelectFromModel, SequentialFeatureSelector
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Advanced feature selection methods."""

    def __init__(self):
        """Initialize feature selector."""
        self.selected_features = None
        self.feature_scores = None

    def select_recursive_elimination(self, X: pd.DataFrame, y: pd.Series,
                                    n_features: int = 10,
                                    step: int = 1) -> Dict[str, any]:
        """
        Select features using Recursive Feature Elimination (RFE).

        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select
            step: Features to eliminate per iteration

        Returns:
            Dictionary with selected features and scores
        """
        try:
            logger.info(f"Starting RFE with {n_features} target features")

            # Use XGBoost as estimator
            estimator = xgb.XGBClassifier(
                max_depth=5,
                n_estimators=100,
                random_state=42,
                verbosity=0
            )

            rfe = RFE(estimator, n_features_to_select=n_features, step=step)
            rfe.fit(X, y)

            selected = X.columns[rfe.support_].tolist()
            self.selected_features = selected

            results = {
                'method': 'rfe',
                'selected_features': selected,
                'n_features': len(selected),
                'ranking': dict(zip(X.columns, rfe.ranking_))
            }

            logger.info(f"RFE selected {len(selected)} features")
            return results

        except Exception as e:
            logger.error(f"RFE failed: {e}")
            return {}

    def select_correlation(self, X: pd.DataFrame, y: pd.Series,
                          n_features: int = 10,
                          threshold: float = 0.1) -> Dict[str, any]:
        """
        Select features based on correlation with target.

        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select
            threshold: Minimum correlation threshold

        Returns:
            Dictionary with selected features and correlations
        """
        try:
            logger.info(f"Starting correlation-based selection")

            # Ensure y is a Series with proper alignment
            if not isinstance(y, pd.Series):
                y = pd.Series(y, name='target')

            # Calculate correlations - handle the case where y.name is None
            if hasattr(y, 'name') and y.name:
                # Try to get correlation by name
                try:
                    correlations = X.corr(numeric_only=True)[y.name].abs()
                except KeyError:
                    # y.name not in X, compute directly
                    correlations = X.corrwith(y).abs()
            else:
                # No name, compute directly
                correlations = X.corrwith(y).abs()

            correlations = correlations.dropna().sort_values(ascending=False)

            # Filter by threshold
            valid_corr = correlations[correlations > threshold]
            selected = valid_corr.head(n_features).index.tolist()

            self.selected_features = selected
            self.feature_scores = valid_corr.to_dict()

            results = {
                'method': 'correlation',
                'selected_features': selected,
                'n_features': len(selected),
                'correlations': self.feature_scores
            }

            logger.info(f"Correlation selection found {len(selected)} features above threshold {threshold}")
            return results

        except Exception as e:
            logger.error(f"Correlation selection failed: {e}")
            return {}

    def select_mutual_information(self, X: pd.DataFrame, y: pd.Series,
                                 n_features: int = 10) -> Dict[str, any]:
        """
        Select features using mutual information with target.

        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select

        Returns:
            Dictionary with selected features and scores
        """
        try:
            logger.info(f"Starting mutual information selection")

            selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
            selector.fit(X, y)

            scores = dict(zip(X.columns, selector.scores_))
            selected_indices = selector.get_support()
            selected = X.columns[selected_indices].tolist()

            self.selected_features = selected
            self.feature_scores = scores

            results = {
                'method': 'mutual_information',
                'selected_features': selected,
                'n_features': len(selected),
                'scores': scores
            }

            logger.info(f"Mutual information selection selected {len(selected)} features")
            return results

        except Exception as e:
            logger.error(f"Mutual information selection failed: {e}")
            return {}

    def select_model_based(self, X: pd.DataFrame, y: pd.Series,
                          n_features: int = 10,
                          model_type: str = 'xgboost') -> Dict[str, any]:
        """
        Select features based on model importance scores.

        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select
            model_type: 'xgboost' or 'random_forest'

        Returns:
            Dictionary with selected features and importances
        """
        try:
            logger.info(f"Starting model-based feature selection ({model_type})")

            if model_type == 'xgboost':
                model = xgb.XGBClassifier(
                    max_depth=5,
                    n_estimators=100,
                    random_state=42,
                    verbosity=0
                )
            elif model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.fit(X, y)

            importances = dict(zip(X.columns, model.feature_importances_))
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            selected = [f[0] for f in sorted_features[:n_features]]

            self.selected_features = selected
            self.feature_scores = importances

            results = {
                'method': f'model_based_{model_type}',
                'selected_features': selected,
                'n_features': len(selected),
                'importances': importances
            }

            logger.info(f"Model-based selection selected {len(selected)} features")
            return results

        except Exception as e:
            logger.error(f"Model-based selection failed: {e}")
            return {}

    def select_sequential(self, X: pd.DataFrame, y: pd.Series,
                         n_features: int = 10,
                         direction: str = 'forward') -> Dict[str, any]:
        """
        Select features using Sequential Feature Selection.

        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select
            direction: 'forward' or 'backward'

        Returns:
            Dictionary with selected features
        """
        try:
            logger.info(f"Starting sequential feature selection ({direction})")

            estimator = xgb.XGBClassifier(
                max_depth=5,
                n_estimators=50,
                random_state=42,
                verbosity=0
            )

            sfs = SequentialFeatureSelector(
                estimator,
                n_features_to_select=min(n_features, X.shape[1]),
                direction=direction,
                cv=3,
                n_jobs=-1
            )

            sfs.fit(X, y)
            selected = X.columns[sfs.get_support()].tolist()

            self.selected_features = selected

            results = {
                'method': f'sequential_{direction}',
                'selected_features': selected,
                'n_features': len(selected)
            }

            logger.info(f"Sequential selection ({direction}) selected {len(selected)} features")
            return results

        except Exception as e:
            logger.error(f"Sequential selection failed: {e}")
            return {}

    def compare_methods(self, X: pd.DataFrame, y: pd.Series,
                       n_features: int = 10) -> Dict[str, any]:
        """
        Compare multiple feature selection methods.

        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select per method

        Returns:
            Dictionary with results from all methods
        """
        results = {}

        logger.info("Running all feature selection methods for comparison")

        # Correlation
        results['correlation'] = self.select_correlation(X, y, n_features)

        # Mutual information
        results['mutual_information'] = self.select_mutual_information(X, y, n_features)

        # XGBoost
        results['xgboost'] = self.select_model_based(X, y, n_features, 'xgboost')

        # Random Forest
        results['random_forest'] = self.select_model_based(X, y, n_features, 'random_forest')

        # RFE
        results['rfe'] = self.select_recursive_elimination(X, y, n_features)

        # Sequential forward
        results['sequential_forward'] = self.select_sequential(X, y, n_features, 'forward')

        # Calculate consensus (features selected by multiple methods)
        all_selected = []
        for method_result in results.values():
            if 'selected_features' in method_result:
                all_selected.extend(method_result['selected_features'])

        from collections import Counter
        feature_counts = Counter(all_selected)
        consensus_features = [f for f, count in feature_counts.items() if count >= 3]

        results['consensus'] = {
            'method': 'consensus',
            'selected_features': consensus_features,
            'n_features': len(consensus_features),
            'consensus_threshold': 3,
            'feature_agreement': dict(feature_counts)
        }

        logger.info(f"Consensus selection found {len(consensus_features)} features agreed by 3+ methods")

        return results

    def get_selected_features(self) -> Optional[List[str]]:
        """Get last selected features."""
        return self.selected_features

    def get_feature_scores(self) -> Optional[Dict[str, float]]:
        """Get last computed feature scores."""
        return self.feature_scores
