"""
VN2 Benchmark Forecaster

Implements the exact approach used in the VN2 competition benchmark:
1. Compute seasonal factors (52-week multiplicative)
2. De-seasonalize demand
3. 13-week moving average of de-seasonalized demand
4. Re-seasonalize forecast for future periods
5. Return quantiles by assuming distribution around forecast

This is a simple, adaptive approach that:
- Uses only recent 13-week data (adapts quickly to demand changes)
- Handles seasonality explicitly
- No complex ML models
- Target cost: €5,248
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VN2BenchmarkForecaster:
    """
    VN2 Benchmark forecaster using seasonal moving average.
    
    Approach from VN2.py generate_benchmark_order():
    - Seasonal factors: 52-week multiplicative weekly patterns
    - Base forecast: 13-week MA of de-seasonalized demand
    - Re-seasonalize for future weeks
    """
    
    def __init__(self, forecast_config):
        self.forecast_config = forecast_config
        self.name = "vn2_benchmark"
        self.quantiles = forecast_config.quantiles
        self.horizons = list(range(1, forecast_config.horizon + 1))
        
    def fit(self, sku_id: tuple, demand_history: pd.Series) -> bool:
        """
        VN2 approach doesn't require training - it's pure online computation.
        
        Args:
            sku_id: (Store, Product) tuple
            demand_history: Historical demand series with DatetimeIndex
            
        Returns:
            True (always succeeds)
        """
        return True
    
    def predict(
        self, 
        sku_id: tuple, 
        demand_history: pd.Series,
        forecast_date: pd.Timestamp,
        exog_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate VN2-style forecast: seasonal 13-week MA.
        
        Args:
            sku_id: (Store, Product) tuple
            demand_history: Historical demand with DatetimeIndex
            forecast_date: Date to forecast from
            exog_future: Not used (VN2 approach is univariate)
            
        Returns:
            DataFrame with columns [horizon, quantile, value]
        """
        # Step 1: Compute seasonal factors (52-week multiplicative)
        seasonal_factors = self._compute_seasonal_factors(demand_history)
        
        # Step 2: De-seasonalize demand
        demand_deseason = self._deseasonalize(demand_history, seasonal_factors)
        
        # Step 3: 13-week moving average of de-seasonalized demand
        # Use last 13 weeks of de-seasonalized data
        recent_deseason = demand_deseason.iloc[-13:] if len(demand_deseason) >= 13 else demand_deseason
        base_forecast = recent_deseason.mean()
        
        # Handle edge case: no historical data or all zeros
        if pd.isna(base_forecast) or base_forecast < 0:
            base_forecast = 0.0
            
        # Step 4: Re-seasonalize for future periods
        forecast_periods = pd.date_range(
            start=forecast_date + pd.Timedelta(weeks=1),
            periods=max(self.horizons),
            freq='W-MON'
        )
        
        # Get seasonal factors for forecast weeks
        forecast_week_numbers = forecast_periods.isocalendar().week
        seasonal_multipliers = seasonal_factors.loc[forecast_week_numbers].values
        
        # Step 5: Generate forecasts for each horizon
        results = []
        for h in self.horizons:
            # Seasonalized point forecast
            point_forecast = base_forecast * seasonal_multipliers[h - 1]
            
            # Generate quantiles by assuming Poisson-like distribution
            # (simple approach: scale based on forecast level)
            for q in self.quantiles:
                if point_forecast <= 0:
                    # If forecast is zero/negative, all quantiles are 0
                    quantile_value = 0.0
                else:
                    # Simple heuristic: assume CV (coefficient of variation) of 1.0
                    # which is reasonable for intermittent demand
                    # Use Gaussian approximation: Q(p) ≈ μ + σ*Φ^(-1)(p)
                    std_dev = np.sqrt(point_forecast)  # Poisson-like
                    z_score = self._normal_ppf(q)
                    quantile_value = max(0, point_forecast + std_dev * z_score)
                
                results.append({
                    'horizon': h,
                    'quantile': q,
                    'value': quantile_value
                })
        
        return pd.DataFrame(results)
    
    def _compute_seasonal_factors(self, demand_history: pd.Series) -> pd.Series:
        """
        Compute multiplicative seasonal factors per ISO week number.
        
        Args:
            demand_history: Historical demand with DatetimeIndex
            
        Returns:
            Series indexed by week number (1-52) with seasonal factors
        """
        if len(demand_history) < 4:
            # Not enough data for seasonality - return neutral factors
            return pd.Series(1.0, index=range(1, 53))
        
        # Create DataFrame with demand and week numbers
        season_df = pd.DataFrame({
            'demand': demand_history.values,
            'week_num': demand_history.index.isocalendar().week
        })
        
        # Average demand per week number
        weekly_avg = season_df.groupby('week_num')['demand'].mean()
        
        # Normalize to mean of 1.0 (multiplicative factors)
        overall_mean = weekly_avg.mean()
        if overall_mean > 0:
            seasonal_factors = weekly_avg / overall_mean
        else:
            seasonal_factors = pd.Series(1.0, index=weekly_avg.index)
        
        # Fill missing weeks with 1.0
        all_weeks = pd.Series(1.0, index=range(1, 53))
        all_weeks.update(seasonal_factors)
        
        return all_weeks
    
    def _deseasonalize(
        self, 
        demand_history: pd.Series, 
        seasonal_factors: pd.Series
    ) -> pd.Series:
        """
        Remove seasonality from demand history.
        
        Args:
            demand_history: Historical demand with DatetimeIndex
            seasonal_factors: Seasonal factors indexed by week number
            
        Returns:
            De-seasonalized demand series
        """
        week_numbers = demand_history.index.isocalendar().week
        factors = seasonal_factors.loc[week_numbers].values
        
        # Divide by seasonal factors (multiplicative de-seasonalization)
        demand_deseason = demand_history / factors
        
        return demand_deseason
    
    def _normal_ppf(self, q: float) -> float:
        """
        Approximate inverse CDF (percent point function) of standard normal.
        
        Args:
            q: Quantile level (0 to 1)
            
        Returns:
            Z-score corresponding to quantile
        """
        # Use scipy if available, otherwise simple approximation
        try:
            from scipy.stats import norm
            return norm.ppf(q)
        except ImportError:
            # Simple approximation for common quantiles
            # This is a rough approximation - good enough for demo
            if q <= 0.01:
                return -2.33
            elif q <= 0.05:
                return -1.64
            elif q <= 0.1:
                return -1.28
            elif q <= 0.2:
                return -0.84
            elif q <= 0.3:
                return -0.52
            elif q <= 0.4:
                return -0.25
            elif q <= 0.5:
                return 0.0
            elif q <= 0.6:
                return 0.25
            elif q <= 0.7:
                return 0.52
            elif q <= 0.8:
                return 0.84
            elif q <= 0.9:
                return 1.28
            elif q <= 0.95:
                return 1.64
            elif q <= 0.99:
                return 2.33
            else:
                return 2.58
    
    def get_coverage(self) -> Dict[str, int]:
        """Return coverage statistics (not applicable for VN2 - all SKUs covered)."""
        return {"total": -1, "fitted": -1}
