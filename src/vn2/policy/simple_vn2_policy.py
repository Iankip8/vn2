"""
Simple VN2-Style Ordering Policy

Implements the simple order-up-to policy used by VN2:
- order_up_to = coverage_weeks × median_forecast
- order = max(0, order_up_to - net_inventory)

This is simpler than the newsvendor approach and more robust to forecast bias.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class SimpleVN2Policy:
    """
    Simple VN2-style order-up-to policy.
    
    Policy:
        order_up_to = coverage_weeks × avg(median(h3), median(h4), median(h5))
        order = max(0, order_up_to - net_inventory)
    
    This approach:
    - Uses median (50th percentile) instead of high percentiles
    - Simple coverage multiplier instead of complex optimization
    - More robust to forecast bias
    """
    
    def __init__(
        self,
        coverage_weeks: float = 4.0,
        horizons: list = [3, 4, 5],
        holding_cost: float = 0.2,
        shortage_cost: float = 1.0
    ):
        """
        Initialize Simple VN2 Policy.
        
        Args:
            coverage_weeks: Number of weeks of coverage to maintain
            horizons: Forecast horizons to use (typically [3,4,5] for T=3 protection)
            holding_cost: Cost per unit per week of holding inventory
            shortage_cost: Cost per unit of unmet demand
        """
        self.coverage_weeks = coverage_weeks
        self.horizons = horizons
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.name = f"simple_vn2_coverage_{coverage_weeks}"
        
        logger.info(f"Initialized SimpleVN2Policy:")
        logger.info(f"  Coverage: {coverage_weeks} weeks")
        logger.info(f"  Horizons: {horizons}")
        logger.info(f"  Costs: H={holding_cost}, S={shortage_cost}")
    
    def compute_order(
        self,
        sku_id: Tuple,
        forecasts: pd.DataFrame,
        current_state: Dict
    ) -> Dict:
        """
        Compute order quantity using simple order-up-to policy.
        
        Args:
            sku_id: (store, product) tuple
            forecasts: DataFrame with columns [horizon, quantile, value]
            current_state: Dict with keys:
                - on_hand: Current on-hand inventory
                - in_transit_w1: Arriving in 1 week
                - in_transit_w2: Arriving in 2 weeks
                
        Returns:
            Dict with keys:
                - order_quantity: Units to order
                - order_up_to_level: Target inventory position
                - net_inventory: Current inventory position
                - median_forecast: Average median forecast used
        """
        # Extract median forecasts for specified horizons
        medians = []
        for h in self.horizons:
            h_forecasts = forecasts[forecasts['horizon'] == h]
            if len(h_forecasts) == 0:
                logger.warning(f"SKU {sku_id}: Missing horizon {h}, using 0")
                medians.append(0.0)
                continue
            
            # Get median (q=0.50)
            median_row = h_forecasts[h_forecasts['quantile'] == 0.5]
            if len(median_row) == 0:
                # If exact 0.5 not available, interpolate
                quantiles = h_forecasts.sort_values('quantile')
                median_val = np.interp(0.5, quantiles['quantile'], quantiles['value'])
            else:
                median_val = median_row['value'].iloc[0]
            
            medians.append(max(0, median_val))
        
        # Average the medians
        avg_median = np.mean(medians)
        
        # Compute order-up-to level
        order_up_to = self.coverage_weeks * avg_median
        
        # Compute net inventory position
        net_inventory = (
            current_state.get('on_hand', 0) +
            current_state.get('in_transit_w1', 0) +
            current_state.get('in_transit_w2', 0)
        )
        
        # Compute order quantity
        order_quantity = max(0, order_up_to - net_inventory)
        
        return {
            'order_quantity': order_quantity,
            'order_up_to_level': order_up_to,
            'net_inventory': net_inventory,
            'median_forecast': avg_median,
            'medians_by_horizon': {h: m for h, m in zip(self.horizons, medians)}
        }
    
    def __str__(self):
        return f"SimpleVN2Policy(coverage={self.coverage_weeks})"
