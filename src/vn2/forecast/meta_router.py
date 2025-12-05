"""
Meta-Router: Dynamic Model Selection for Forecast-Based Optimization.

The meta-router learns which forecasting model works best for each SKU based on
historical cost outcomes. Unlike static model selection (which picks one model per
SKU based on average performance), the meta-router can adapt its selection based
on current conditions (recent CV, stockout history, trend).

Key Features:
- Learns from realized costs, not just forecast accuracy
- Uses SKU features (CV, zero_rate, stockout_history, trend) for selection
- Can output probability distributions over models for soft routing
- Supports ensemble blending when confidence is low

Usage:
    router = MetaRouter(model_names=['zinb', 'slurp_bootstrap', 'lightgbm_quantile'])
    router.fit(historical_costs_df, sku_features_df)
    selections = router.predict(current_features_df)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score


@dataclass
class RouterConfig:
    """Configuration for meta-router."""
    model_names: List[str]
    classifier_type: str = 'random_forest'  # 'random_forest', 'gradient_boosting'
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_leaf: int = 5
    confidence_threshold: float = 0.6  # Below this, use ensemble
    random_state: int = 42


class MetaRouter:
    """Dynamic model selector based on learned cost patterns.
    
    The router learns to predict which model will have the lowest cost for a
    given SKU-week based on features like:
    - CV (coefficient of variation)
    - Zero rate (proportion of zero-demand periods)
    - Recent stockout rate
    - Trend direction
    - Mean demand level
    - Intermittency pattern
    """
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.model_names = config.model_names
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.classifier = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.feature_importances: Dict[str, float] = {}
    
    def _create_classifier(self):
        """Create the underlying classifier."""
        if self.config.classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.classifier_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.config.classifier_type}")
    
    def fit(
        self,
        cost_history: pd.DataFrame,
        features: pd.DataFrame
    ) -> 'MetaRouter':
        """Fit the router from historical cost data.
        
        Args:
            cost_history: DataFrame with columns [store, product, week, model, realized_cost]
            features: DataFrame with columns [store, product, week, ...features...]
        
        Returns:
            Self for chaining
        """
        # Find best model per SKU-week
        best_models = cost_history.loc[
            cost_history.groupby(['store', 'product', 'week'])['realized_cost'].idxmin()
        ][['store', 'product', 'week', 'model']].copy()
        
        # Merge with features
        data = best_models.merge(
            features,
            on=['store', 'product', 'week'],
            how='inner'
        )
        
        if len(data) < 50:
            raise ValueError(f"Insufficient training data: {len(data)} samples")
        
        # Prepare features
        self.feature_names = [c for c in features.columns 
                             if c not in ['store', 'product', 'week']]
        
        X = data[self.feature_names].values
        y = data['model'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier = self._create_classifier()
        self.classifier.fit(X_scaled, y_encoded)
        
        # Store feature importances
        if hasattr(self.classifier, 'feature_importances_'):
            self.feature_importances = dict(zip(
                self.feature_names,
                self.classifier.feature_importances_
            ))
        
        self.is_fitted = True
        return self
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict best model for each SKU.
        
        Args:
            features: DataFrame with columns [store, product, ...features...]
        
        Returns:
            DataFrame with columns [store, product, selected_model, confidence]
        """
        if not self.is_fitted:
            raise ValueError("Router must be fitted before predict")
        
        X = features[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        y_pred = self.classifier.predict(X_scaled)
        y_proba = self.classifier.predict_proba(X_scaled)
        
        # Decode labels
        models_pred = self.label_encoder.inverse_transform(y_pred)
        confidences = y_proba.max(axis=1)
        
        result = features[['store', 'product']].copy()
        result['selected_model'] = models_pred
        result['confidence'] = confidences
        
        # Flag low-confidence predictions
        result['use_ensemble'] = confidences < self.config.confidence_threshold
        
        return result
    
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get probability distribution over models for each SKU.
        
        Args:
            features: DataFrame with SKU features
        
        Returns:
            DataFrame with columns [store, product, model_name_1, model_name_2, ...]
        """
        if not self.is_fitted:
            raise ValueError("Router must be fitted before predict_proba")
        
        X = features[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        y_proba = self.classifier.predict_proba(X_scaled)
        
        result = features[['store', 'product']].copy()
        for i, model_name in enumerate(self.label_encoder.classes_):
            result[f'prob_{model_name}'] = y_proba[:, i]
        
        return result
    
    def get_ensemble_weights(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get ensemble weights when confidence is low.
        
        For SKUs where the router is uncertain, we blend forecasts from multiple
        models weighted by their predicted probabilities.
        
        Args:
            features: DataFrame with SKU features
        
        Returns:
            DataFrame with model weights per SKU
        """
        proba_df = self.predict_proba(features)
        
        # Only include probability columns
        prob_cols = [c for c in proba_df.columns if c.startswith('prob_')]
        
        # Normalize weights to sum to 1
        weights = proba_df[prob_cols].values
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        result = proba_df[['store', 'product']].copy()
        for i, col in enumerate(prob_cols):
            result[col.replace('prob_', 'weight_')] = weights[:, i]
        
        return result
    
    def cross_validate(
        self,
        cost_history: pd.DataFrame,
        features: pd.DataFrame,
        cv: int = 5
    ) -> Dict[str, float]:
        """Cross-validate the router on historical data."""
        # Prepare data (same as fit)
        best_models = cost_history.loc[
            cost_history.groupby(['store', 'product', 'week'])['realized_cost'].idxmin()
        ][['store', 'product', 'week', 'model']].copy()
        
        data = best_models.merge(features, on=['store', 'product', 'week'], how='inner')
        
        feature_names = [c for c in features.columns 
                        if c not in ['store', 'product', 'week']]
        
        X = data[feature_names].values
        y = data['model'].values
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        classifier = self._create_classifier()
        
        scores = cross_val_score(classifier, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        
        return {
            'mean_accuracy': float(scores.mean()),
            'std_accuracy': float(scores.std()),
            'min_accuracy': float(scores.min()),
            'max_accuracy': float(scores.max())
        }
    
    def save(self, path: Path) -> None:
        """Save router to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'classifier': self.classifier,
                'feature_names': self.feature_names,
                'feature_importances': self.feature_importances
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> 'MetaRouter':
        """Load router from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        router = cls(data['config'])
        router.label_encoder = data['label_encoder']
        router.scaler = data['scaler']
        router.classifier = data['classifier']
        router.feature_names = data['feature_names']
        router.feature_importances = data['feature_importances']
        router.is_fitted = True
        return router
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get feature importances as a sorted DataFrame."""
        if not self.feature_importances:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in sorted(
                self.feature_importances.items(),
                key=lambda x: -x[1]
            )
        ])


def compute_sku_features(
    demand_df: pd.DataFrame,
    lookback_weeks: int = 12
) -> pd.DataFrame:
    """Compute SKU features for meta-router from demand history.
    
    Args:
        demand_df: DataFrame with columns [store, product, week, demand]
        lookback_weeks: Number of weeks to use for feature computation
    
    Returns:
        DataFrame with computed features per SKU
    """
    features = []
    
    for (store, product), group in demand_df.groupby(['store', 'product']):
        group = group.sort_values('week')
        y = group['demand'].values[-lookback_weeks:] if len(group) >= lookback_weeks else group['demand'].values
        
        if len(y) < 2:
            continue
        
        # Basic statistics
        mean_demand = np.mean(y)
        std_demand = np.std(y)
        cv = std_demand / max(mean_demand, 0.01)
        
        # Zero/intermittency features
        zero_rate = np.mean(y == 0)
        nonzero_count = np.sum(y > 0)
        
        # Trend (linear regression slope)
        if len(y) > 2:
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            trend_direction = 1 if slope > 0.1 else (-1 if slope < -0.1 else 0)
        else:
            slope = 0
            trend_direction = 0
        
        # Demand level category
        demand_level = 'high' if mean_demand > 10 else ('medium' if mean_demand > 2 else 'low')
        
        # Volatility pattern
        if len(y) > 4:
            recent_cv = np.std(y[-4:]) / max(np.mean(y[-4:]), 0.01)
            cv_change = recent_cv - cv
        else:
            cv_change = 0
        
        # Intermittency pattern (ADI - average demand interval)
        if nonzero_count > 0:
            adi = len(y) / nonzero_count
        else:
            adi = len(y)
        
        features.append({
            'store': store,
            'product': product,
            'mean_demand': mean_demand,
            'std_demand': std_demand,
            'cv': cv,
            'zero_rate': zero_rate,
            'nonzero_count': nonzero_count,
            'trend_slope': slope,
            'trend_direction': trend_direction,
            'demand_level_high': 1 if demand_level == 'high' else 0,
            'demand_level_medium': 1 if demand_level == 'medium' else 0,
            'cv_change': cv_change,
            'adi': adi,
            'is_intermittent': 1 if adi > 1.32 else 0,  # SBC threshold
        })
    
    return pd.DataFrame(features)

