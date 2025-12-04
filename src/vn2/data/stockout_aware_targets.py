"""
Stockout-Aware Targets for Non-SLURP Models.

When demand is censored (stockout periods where we observe sales = available stock),
the true demand is unknown. Traditional approaches impute a single point value,
which collapses uncertainty and can bias forecasts.

This module provides methods to create stockout-aware training targets that:
1. Preserve uncertainty during stockout periods as INTERVALS instead of points
2. Allow quantile regression models to learn from the uncertainty
3. Can be used to create weighted loss functions that account for censoring

Key Insight: Instead of imputing stockout periods as y = observed_sales + estimated_miss,
we create an interval target [observed_sales, observed_sales + upper_bound] and train
models to minimize loss over the interval.

Usage:
    targets = StockoutAwareTargets()
    targets.fit(demand_df, in_stock_flags)
    interval_df = targets.transform(demand_df)
    # Then use interval_df to train quantile models with interval-aware loss
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class IntervalTarget:
    """Represents a target as an interval [lower, upper]."""
    lower: float
    upper: float
    is_censored: bool
    weight: float = 1.0  # Higher weight for non-censored observations


class StockoutAwareTargets:
    """Create interval-based targets that preserve stockout uncertainty.
    
    For non-stockout periods: interval is [y, y] (point)
    For stockout periods: interval is [observed, observed + estimate]
    
    The upper bound estimate is based on historical patterns:
    - For sporadic stockouts: use rolling mean + k*std
    - For persistent stockouts: use neighbor SKU patterns
    """
    
    def __init__(
        self,
        upper_multiplier: float = 3.0,
        min_upper_bound: float = 1.0,
        weight_non_censored: float = 1.0,
        weight_censored: float = 0.5
    ):
        """Initialize stockout-aware target creator.
        
        Args:
            upper_multiplier: Multiplier for std to create upper bound (default 3.0 = 3σ)
            min_upper_bound: Minimum additional units for upper bound
            weight_non_censored: Sample weight for non-censored observations
            weight_censored: Sample weight for censored observations (lower = more uncertain)
        """
        self.upper_multiplier = upper_multiplier
        self.min_upper_bound = min_upper_bound
        self.weight_non_censored = weight_non_censored
        self.weight_censored = weight_censored
        
        self.sku_stats: Dict[Tuple[int, int], Dict] = {}
    
    def fit(
        self,
        demand_df: pd.DataFrame,
        in_stock_col: str = 'in_stock'
    ) -> 'StockoutAwareTargets':
        """Fit statistics from historical demand.
        
        Args:
            demand_df: DataFrame with columns [store, product, week, demand, in_stock]
            in_stock_col: Name of column indicating non-stockout periods (1 = in stock)
        
        Returns:
            Self for chaining
        """
        for (store, product), group in demand_df.groupby(['store', 'product']):
            # Get non-censored observations only
            non_censored = group[group[in_stock_col] == 1]['demand']
            
            if len(non_censored) >= 2:
                mean = non_censored.mean()
                std = non_censored.std()
            else:
                mean = group['demand'].mean()
                std = group['demand'].std() if len(group) >= 2 else mean * 0.5
            
            self.sku_stats[(store, product)] = {
                'mean': float(mean),
                'std': float(std),
                'max_observed': float(group['demand'].max()),
                'n_censored': int((group[in_stock_col] == 0).sum()),
                'n_total': len(group)
            }
        
        return self
    
    def transform(
        self,
        demand_df: pd.DataFrame,
        in_stock_col: str = 'in_stock'
    ) -> pd.DataFrame:
        """Create interval targets from demand data.
        
        Args:
            demand_df: DataFrame with columns [store, product, week, demand, in_stock]
            in_stock_col: Name of column indicating non-stockout periods
        
        Returns:
            DataFrame with columns:
            - All original columns
            - target_lower: Lower bound of interval
            - target_upper: Upper bound of interval
            - is_censored: Whether this observation is censored
            - sample_weight: Weight for this observation
        """
        result = demand_df.copy()
        
        target_lower = []
        target_upper = []
        is_censored = []
        sample_weights = []
        
        for _, row in demand_df.iterrows():
            store = row['store']
            product = row['product']
            observed = row['demand']
            in_stock = row[in_stock_col]
            
            if in_stock == 1:
                # Not censored: interval is [y, y]
                target_lower.append(observed)
                target_upper.append(observed)
                is_censored.append(False)
                sample_weights.append(self.weight_non_censored)
            else:
                # Censored: interval is [observed, observed + upper_estimate]
                stats = self.sku_stats.get((store, product), {})
                mean = stats.get('mean', observed)
                std = stats.get('std', observed * 0.5)
                
                # Upper bound: observed + multiplier * std, at least min_upper_bound more
                upper_add = max(self.upper_multiplier * std, self.min_upper_bound)
                upper = observed + upper_add
                
                target_lower.append(observed)
                target_upper.append(upper)
                is_censored.append(True)
                sample_weights.append(self.weight_censored)
        
        result['target_lower'] = target_lower
        result['target_upper'] = target_upper
        result['is_censored'] = is_censored
        result['sample_weight'] = sample_weights
        
        return result
    
    def get_interval_pinball_loss(
        self,
        y_pred: float,
        lower: float,
        upper: float,
        quantile: float
    ) -> float:
        """Compute pinball loss for interval target.
        
        For interval [lower, upper]:
        - If y_pred is within interval, loss is 0
        - If y_pred < lower, loss is standard underprediction penalty
        - If y_pred > upper, loss is standard overprediction penalty
        
        Args:
            y_pred: Predicted quantile value
            lower: Lower bound of target interval
            upper: Upper bound of target interval
            quantile: Quantile level (e.g., 0.8333)
        
        Returns:
            Pinball loss value
        """
        if y_pred < lower:
            # Underprediction: we know true demand >= lower
            return quantile * (lower - y_pred)
        elif y_pred > upper:
            # Overprediction: we're confident true demand <= upper
            return (1 - quantile) * (y_pred - upper)
        else:
            # Within interval: no penalty
            return 0.0


def create_interval_targets(
    demand_df: pd.DataFrame,
    in_stock_col: str = 'in_stock',
    upper_multiplier: float = 3.0,
    min_upper_bound: float = 1.0
) -> pd.DataFrame:
    """Convenience function to create interval targets.
    
    Args:
        demand_df: DataFrame with demand and in_stock columns
        in_stock_col: Name of in_stock column
        upper_multiplier: Multiplier for std to create upper bound
        min_upper_bound: Minimum additional units for upper bound
    
    Returns:
        DataFrame with interval target columns added
    """
    targets = StockoutAwareTargets(
        upper_multiplier=upper_multiplier,
        min_upper_bound=min_upper_bound
    )
    targets.fit(demand_df, in_stock_col)
    return targets.transform(demand_df, in_stock_col)


def create_weighted_loss_targets(
    demand_df: pd.DataFrame,
    in_stock_col: str = 'in_stock',
    censored_weight: float = 0.5,
    non_censored_weight: float = 1.0
) -> pd.DataFrame:
    """Create weighted targets for censoring-aware training.
    
    This is a simpler approach that doesn't use intervals, but instead
    assigns lower sample weights to censored observations.
    
    Args:
        demand_df: DataFrame with demand and in_stock columns
        in_stock_col: Name of in_stock column
        censored_weight: Weight for censored observations
        non_censored_weight: Weight for non-censored observations
    
    Returns:
        DataFrame with sample_weight column added
    """
    result = demand_df.copy()
    result['sample_weight'] = np.where(
        result[in_stock_col] == 1,
        non_censored_weight,
        censored_weight
    )
    return result


class IntervalQuantileLoss:
    """Loss function for training quantile models with interval targets.
    
    This loss function:
    - Uses standard pinball loss for point targets (non-censored)
    - Uses interval-aware pinball loss for interval targets (censored)
    - Supports sample weighting
    
    Usage with sklearn-compatible models:
        loss = IntervalQuantileLoss(quantile=0.8333)
        # Custom training loop with this loss
    """
    
    def __init__(self, quantile: float = 0.8333):
        self.quantile = quantile
    
    def __call__(
        self,
        y_pred: np.ndarray,
        target_lower: np.ndarray,
        target_upper: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Compute interval-aware pinball loss.
        
        Args:
            y_pred: Predicted values
            target_lower: Lower bounds of intervals
            target_upper: Upper bounds of intervals
            sample_weight: Optional sample weights
        
        Returns:
            Weighted average loss
        """
        n = len(y_pred)
        losses = np.zeros(n)
        
        for i in range(n):
            pred = y_pred[i]
            lower = target_lower[i]
            upper = target_upper[i]
            
            if pred < lower:
                losses[i] = self.quantile * (lower - pred)
            elif pred > upper:
                losses[i] = (1 - self.quantile) * (pred - upper)
            # else: within interval, loss = 0
        
        if sample_weight is not None:
            return np.average(losses, weights=sample_weight)
        else:
            return np.mean(losses)
    
    def gradient(
        self,
        y_pred: np.ndarray,
        target_lower: np.ndarray,
        target_upper: np.ndarray
    ) -> np.ndarray:
        """Compute gradient of interval-aware pinball loss.
        
        For gradient-based optimization (e.g., custom LightGBM objective).
        """
        n = len(y_pred)
        grad = np.zeros(n)
        
        for i in range(n):
            pred = y_pred[i]
            lower = target_lower[i]
            upper = target_upper[i]
            
            if pred < lower:
                grad[i] = -self.quantile
            elif pred > upper:
                grad[i] = 1 - self.quantile
            # else: within interval, gradient = 0
        
        return grad

