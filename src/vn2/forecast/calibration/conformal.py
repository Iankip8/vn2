"""
Conformalized Quantile Calibration for Forecast Distributions.

This module implements post-hoc calibration of quantile forecasts using
conformal prediction methods. The key idea is to adjust quantile forecasts
based on observed residuals from a holdout set, ensuring that the calibrated
quantiles achieve their nominal coverage.

For inventory optimization with critical fractile τ* = 0.8333, proper
calibration around this quantile is essential for minimizing expected cost.

References:
- Romano, Patterson, Candès (2019). "Conformalized Quantile Regression"
- Kuleshov et al. (2018). "Accurate Uncertainties for Deep Learning"
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@dataclass
class CalibrationResult:
    """Result of calibration for a single quantile."""
    nominal: float
    empirical_before: float
    empirical_after: float
    adjustment: float


class ConformalQuantileCalibrator:
    """Post-hoc calibration of quantile forecasts using conformal methods.
    
    The calibrator learns adjustment factors from a holdout set where we have
    both forecasts and actuals. It then applies these adjustments to future
    forecasts to improve coverage.
    
    Attributes:
        quantile_levels: Array of quantile levels being calibrated
        adjustments: Dict mapping quantile level -> additive adjustment
        is_fitted: Whether the calibrator has been fitted
    """
    
    def __init__(self, quantile_levels: np.ndarray):
        """Initialize calibrator.
        
        Args:
            quantile_levels: Array of quantile levels (e.g., [0.1, 0.5, 0.9])
        """
        self.quantile_levels = np.array(quantile_levels)
        self.adjustments: Dict[float, float] = {}
        self.scale_factors: Dict[float, float] = {}
        self.is_fitted = False
        self.calibration_results: List[CalibrationResult] = []
    
    def fit(
        self,
        forecasts: np.ndarray,
        actuals: np.ndarray,
        method: str = 'isotonic'
    ) -> 'ConformalQuantileCalibrator':
        """Fit calibration adjustments from holdout data.
        
        Args:
            forecasts: Array of shape (n_samples, n_quantiles) with quantile forecasts
            actuals: Array of shape (n_samples,) with actual values
            method: Calibration method ('isotonic', 'scaling', or 'additive')
        
        Returns:
            Self for chaining
        """
        n_samples = len(actuals)
        
        for i, q in enumerate(self.quantile_levels):
            q_forecasts = forecasts[:, i]
            
            # Compute empirical coverage before calibration
            empirical_before = np.mean(actuals <= q_forecasts)
            
            if method == 'isotonic':
                # Isotonic regression approach: find adjustment that achieves target coverage
                residuals = actuals - q_forecasts
                sorted_residuals = np.sort(residuals)
                
                # Find the residual at the nominal quantile
                # This is the conformity score that achieves q coverage
                idx = int(np.ceil((n_samples + 1) * q)) - 1
                idx = np.clip(idx, 0, n_samples - 1)
                adjustment = sorted_residuals[idx]
                
                self.adjustments[q] = adjustment
                self.scale_factors[q] = 1.0
                
            elif method == 'scaling':
                # Scale approach: multiply forecasts by a factor
                median_forecast = np.median(q_forecasts[q_forecasts > 0])
                median_actual = np.median(actuals[actuals > 0])
                
                if median_forecast > 0:
                    scale = median_actual / median_forecast
                else:
                    scale = 1.0
                
                self.adjustments[q] = 0.0
                self.scale_factors[q] = scale
                
            else:  # additive
                # Simple additive: shift by median residual at this quantile
                residuals = actuals - q_forecasts
                target_residual = np.quantile(residuals, q)
                
                self.adjustments[q] = target_residual
                self.scale_factors[q] = 1.0
            
            # Compute empirical coverage after calibration
            calibrated = q_forecasts * self.scale_factors[q] + self.adjustments[q]
            empirical_after = np.mean(actuals <= calibrated)
            
            self.calibration_results.append(CalibrationResult(
                nominal=q,
                empirical_before=empirical_before,
                empirical_after=empirical_after,
                adjustment=self.adjustments[q]
            ))
        
        self.is_fitted = True
        return self
    
    def transform(self, forecasts: np.ndarray) -> np.ndarray:
        """Apply calibration to new forecasts.
        
        Args:
            forecasts: Array of shape (n_samples, n_quantiles) or (n_quantiles,)
        
        Returns:
            Calibrated forecasts with same shape
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        is_1d = forecasts.ndim == 1
        if is_1d:
            forecasts = forecasts.reshape(1, -1)
        
        calibrated = np.zeros_like(forecasts)
        for i, q in enumerate(self.quantile_levels):
            scale = self.scale_factors.get(q, 1.0)
            adjustment = self.adjustments.get(q, 0.0)
            calibrated[:, i] = forecasts[:, i] * scale + adjustment
        
        # Ensure monotonicity (higher quantiles >= lower quantiles)
        for row in range(calibrated.shape[0]):
            calibrated[row] = np.maximum.accumulate(calibrated[row])
        
        # Ensure non-negative (for demand forecasts)
        calibrated = np.maximum(calibrated, 0)
        
        if is_1d:
            calibrated = calibrated.flatten()
        
        return calibrated
    
    def fit_transform(
        self,
        forecasts: np.ndarray,
        actuals: np.ndarray,
        method: str = 'isotonic'
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(forecasts, actuals, method)
        return self.transform(forecasts)
    
    def get_calibration_summary(self) -> pd.DataFrame:
        """Get summary of calibration results."""
        if not self.calibration_results:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'quantile': r.nominal,
                'coverage_before': r.empirical_before,
                'coverage_after': r.empirical_after,
                'adjustment': r.adjustment,
                'improvement': abs(r.nominal - r.empirical_after) < abs(r.nominal - r.empirical_before)
            }
            for r in self.calibration_results
        ])
    
    def save(self, path: Path) -> None:
        """Save calibrator to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'quantile_levels': self.quantile_levels,
                'adjustments': self.adjustments,
                'scale_factors': self.scale_factors,
                'calibration_results': self.calibration_results
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> 'ConformalQuantileCalibrator':
        """Load calibrator from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        calibrator = cls(data['quantile_levels'])
        calibrator.adjustments = data['adjustments']
        calibrator.scale_factors = data['scale_factors']
        calibrator.calibration_results = data.get('calibration_results', [])
        calibrator.is_fitted = True
        return calibrator


def calibrate_model_quantiles(
    model_name: str,
    checkpoints_dir: Path,
    actuals_df: pd.DataFrame,
    output_dir: Path,
    quantile_levels: np.ndarray,
    method: str = 'isotonic',
    holdout_folds: List[int] = None
) -> Dict[str, any]:
    """Calibrate a model's quantile forecasts and save calibrated checkpoints.
    
    This function:
    1. Loads forecasts from existing checkpoints
    2. Compares to actuals to fit calibration
    3. Applies calibration to all forecasts
    4. Saves calibrated checkpoints with prefix 'conformal_'
    
    Args:
        model_name: Name of model to calibrate (e.g., 'zinb')
        checkpoints_dir: Directory containing model checkpoints
        actuals_df: DataFrame with columns [store, product, week, actual]
        output_dir: Directory to save calibrated checkpoints
        quantile_levels: Quantile levels in the forecasts
        method: Calibration method
        holdout_folds: Fold indices to use for fitting calibration
    
    Returns:
        Dict with calibration statistics
    """
    model_dir = checkpoints_dir / model_name
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")
    
    # Collect forecasts and actuals for calibration fitting
    all_forecasts = []
    all_actuals = []
    
    holdout_folds = holdout_folds or [0, 1, 2]  # Default: first 3 folds
    
    for sku_dir in model_dir.iterdir():
        if not sku_dir.is_dir():
            continue
        
        try:
            store, product = map(int, sku_dir.name.split('_'))
        except:
            continue
        
        sku_actuals = actuals_df[
            (actuals_df['store'] == store) & 
            (actuals_df['product'] == product)
        ]
        
        for fold_file in sku_dir.glob('fold_*.pkl'):
            fold_idx = int(fold_file.stem.replace('fold_', ''))
            
            if fold_idx not in holdout_folds:
                continue
            
            with open(fold_file, 'rb') as f:
                data = pickle.load(f)
            
            qdf = data.get('quantiles')
            if qdf is None or qdf.empty:
                continue
            
            # Get h=1 forecasts and corresponding actuals
            if 1 in qdf.index:
                q_forecast = qdf.loc[1].values
                # Find actual for this week (fold_idx + 1)
                actual_row = sku_actuals[sku_actuals['week'] == fold_idx + 1]
                if not actual_row.empty:
                    all_forecasts.append(q_forecast)
                    all_actuals.append(actual_row['actual'].iloc[0])
    
    if len(all_forecasts) < 50:
        return {'error': 'Insufficient data for calibration', 'n_samples': len(all_forecasts)}
    
    forecasts_arr = np.array(all_forecasts)
    actuals_arr = np.array(all_actuals)
    
    # Fit calibrator
    calibrator = ConformalQuantileCalibrator(quantile_levels)
    calibrator.fit(forecasts_arr, actuals_arr, method=method)
    
    # Save calibrator
    calibrator_path = output_dir / f'calibrator_{model_name}.pkl'
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrator.save(calibrator_path)
    
    # Apply calibration to all checkpoints and save
    conformal_model_dir = output_dir / f'conformal_{model_name}'
    conformal_model_dir.mkdir(parents=True, exist_ok=True)
    
    n_calibrated = 0
    for sku_dir in model_dir.iterdir():
        if not sku_dir.is_dir():
            continue
        
        sku_output = conformal_model_dir / sku_dir.name
        sku_output.mkdir(parents=True, exist_ok=True)
        
        for fold_file in sku_dir.glob('fold_*.pkl'):
            with open(fold_file, 'rb') as f:
                data = pickle.load(f)
            
            qdf = data.get('quantiles')
            if qdf is None or qdf.empty:
                continue
            
            # Calibrate each horizon
            calibrated_qdf = qdf.copy()
            for h in qdf.index:
                original = qdf.loc[h].values
                calibrated = calibrator.transform(original)
                calibrated_qdf.loc[h] = calibrated
            
            # Save calibrated checkpoint
            output_path = sku_output / fold_file.name
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'quantiles': calibrated_qdf,
                    'original_model': model_name,
                    'calibration_method': method
                }, f)
            
            n_calibrated += 1
    
    return {
        'model': model_name,
        'n_calibration_samples': len(all_forecasts),
        'n_checkpoints_calibrated': n_calibrated,
        'calibrator_path': str(calibrator_path),
        'output_dir': str(conformal_model_dir),
        'summary': calibrator.get_calibration_summary().to_dict()
    }

