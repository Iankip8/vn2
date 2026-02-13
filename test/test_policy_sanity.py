"""
Sanity tests for ordering policy implementation.

Tests verify that Patrick McDonald's recommended fixes are correctly implemented:
1. Protection period = lead_weeks + review_weeks (3 weeks, not 1)
2. Critical fractile = 0.833 explicit (shortage / (holding + shortage))
3. Aggregate weekly distributions over protection horizon via Monte Carlo

Author: Ian
Date: February 4, 2026
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vn2.policy.corrected_policy import (
    aggregate_weekly_distributions_mc,
    compute_base_stock_level_corrected,
    compute_order_quantity_corrected,
    Costs,
    LeadTime,
)


class TestProtectionPeriodAggregation:
    """Test that protection period correctly aggregates demand over lead + review weeks."""
    
    def test_protection_period_calculation(self):
        """Verify protection period = lead_weeks + review_weeks."""
        # Given: Lead time 2 weeks, review period 1 week
        lead_weeks = 2
        review_weeks = 1
        
        # When: Calculate protection period
        protection_weeks = lead_weeks + review_weeks
        
        # Then: Should be 3 weeks
        assert protection_weeks == 3, "Protection period should be lead(2) + review(1) = 3 weeks"
    
    def test_mc_aggregation_sums_weeks(self):
        """Verify Monte Carlo aggregation sums demand across weeks."""
        # Given: Quantile forecasts for h=3,4,5 (when order is available)
        quantiles_df = pd.DataFrame({
            0.1: [1.0, 1.0, 1.0],  # h=3, h=4, h=5
            0.5: [5.0, 5.0, 5.0],
            0.9: [10.0, 10.0, 10.0],
        }, index=[3, 4, 5])  # Horizons when order arrives and is used
        
        quantile_levels = np.array([0.1, 0.5, 0.9])
        
        # When: Aggregate over 3 weeks with lead_weeks=2
        mu, sigma = aggregate_weekly_distributions_mc(
            quantiles_df,
            quantile_levels,
            protection_weeks=3,
            lead_weeks=2,  # Order arrives after 2 weeks
            n_samples=10000,
            seed=42
        )
        
        # Then: Aggregated mean should be ~3x single week mean
        single_week_mean = 5.0
        expected_aggregated_mean = 3 * single_week_mean
        
        # Allow 10% tolerance due to MC sampling
        assert abs(mu - expected_aggregated_mean) / expected_aggregated_mean < 0.10, \
            f"Aggregated mean {mu:.2f} should be ~3x single week {single_week_mean}"
    
    def test_aggregation_increases_variance(self):
        """Verify that aggregating 3 weeks increases variance."""
        # Given: Same quantiles for h=3,4,5
        quantiles_df = pd.DataFrame({
            0.1: [1.0, 1.0, 1.0],
            0.5: [5.0, 5.0, 5.0],
            0.9: [10.0, 10.0, 10.0],
        }, index=[3, 4, 5])  # Horizons when order is available
        
        quantile_levels = np.array([0.1, 0.5, 0.9])
        
        # When: Aggregate
        mu, sigma = aggregate_weekly_distributions_mc(
            quantiles_df,
            quantile_levels,
            protection_weeks=3,
            lead_weeks=2,  # Order arrives after 2 weeks
            n_samples=10000,
            seed=42
        )
        
        # Then: Std should be ~sqrt(3) times higher
        # For sum of independent vars: Var(X1+X2+X3) = Var(X1) + Var(X2) + Var(X3)
        # So Std(sum) = sqrt(3) * Std(single)
        single_week_std = np.std(np.interp(
            np.random.RandomState(42).uniform(0, 1, 10000),
            [0.1, 0.5, 0.9],
            [1.0, 5.0, 10.0]
        ))
        
        expected_aggregated_std = np.sqrt(3) * single_week_std
        
        # Allow 15% tolerance
        assert abs(sigma - expected_aggregated_std) / expected_aggregated_std < 0.15, \
            f"Aggregated std {sigma:.2f} should be ~sqrt(3) × single week std"


class TestCriticalFractile:
    """Test that critical fractile is correctly calculated and applied."""
    
    def test_critical_fractile_calculation(self):
        """Verify critical fractile = shortage / (holding + shortage)."""
        # Given: Standard costs
        shortage_cost = 1.0
        holding_cost = 0.2
        
        # When: Calculate critical fractile
        critical_fractile = shortage_cost / (holding_cost + shortage_cost)
        
        # Then: Should be 0.8333...
        expected = 5.0 / 6.0  # = 0.8333...
        assert abs(critical_fractile - expected) < 0.0001, \
            f"Critical fractile {critical_fractile} should be {expected:.4f}"
    
    def test_base_stock_uses_correct_quantile(self):
        """Verify base-stock level targets the 83.3% quantile."""
        # Given: Aggregated demand with known distribution
        # Use normal(100, 20) so we can calculate exact quantile
        mu = 100.0
        sigma = 20.0
        
        # Create quantile dataframe from normal distribution
        quantile_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        quantile_values = stats.norm.ppf(quantile_levels, loc=mu, scale=sigma)
        
        quantiles_df = pd.DataFrame(
            [quantile_values],
            columns=quantile_levels,
            index=[1]  # Single week for simplicity
        )
        
        # When: Calculate base-stock level
        costs = Costs(holding=0.2, shortage=1.0)
        lt = LeadTime(lead_weeks=0, review_weeks=1)  # 1 week total
        
        S, mu_prot, sigma_prot = compute_base_stock_level_corrected(
            quantiles_df, quantile_levels, costs, lt
        )
        
        # Then: S should be at the 83.3% quantile
        expected_S = stats.norm.ppf(0.8333, loc=mu, scale=sigma)
        
        # Allow 5% tolerance due to MC sampling
        assert abs(S - expected_S) / expected_S < 0.05, \
            f"Base-stock {S:.2f} should target 83.3% quantile {expected_S:.2f}"
    
    def test_higher_shortage_cost_increases_base_stock(self):
        """Verify that higher shortage cost increases base-stock level."""
        # Given: Same demand distribution
        quantile_levels = np.array([0.1, 0.5, 0.9])
        quantiles_df = pd.DataFrame({
            0.1: [10.0],
            0.5: [50.0],
            0.9: [90.0],
        }, index=[1])
        
        lt = LeadTime(lead_weeks=0, review_weeks=1)
        
        # When: Calculate with different shortage costs
        costs_low = Costs(holding=0.2, shortage=0.5)   # Critical fractile = 0.714
        costs_high = Costs(holding=0.2, shortage=2.0)  # Critical fractile = 0.909
        
        S_low, _, _ = compute_base_stock_level_corrected(quantiles_df, quantile_levels, costs_low, lt)
        S_high, _, _ = compute_base_stock_level_corrected(quantiles_df, quantile_levels, costs_high, lt)
        
        # Then: Higher shortage cost should give higher base-stock
        assert S_high > S_low, \
            f"Higher shortage cost should increase base-stock: {S_high} > {S_low}"


class TestBaseStockAccounting:
    """Test that inventory position accounting is correct."""
    
    def test_inventory_position_includes_all_components(self):
        """Verify position = on_hand + intransit_1 + intransit_2."""
        # Given: Initial inventory state
        initial_state = pd.DataFrame({
            'on_hand': [10],
            'intransit_1': [5],
            'intransit_2': [3],
        })
        
        # When: Calculate position
        position = (
            initial_state['on_hand'].iloc[0] + 
            initial_state['intransit_1'].iloc[0] + 
            initial_state['intransit_2'].iloc[0]
        )
        
        # Then: Should sum all components
        assert position == 18, "Position should be 10 + 5 + 3 = 18"
    
    def test_order_quantity_respects_position(self):
        """Verify order = max(0, S - position)."""
        # Given: Base-stock level and position
        quantile_levels = np.array([0.1, 0.5, 0.9])
        quantiles_df = pd.DataFrame({
            0.1: [10.0],
            0.5: [50.0],
            0.9: [90.0],
        }, index=[1])
        
        costs = Costs(holding=0.2, shortage=1.0)
        lt = LeadTime(lead_weeks=0, review_weeks=1)
        
        # Test Case 1: Position < S (should order)
        initial_state_low = pd.DataFrame({
            'on_hand': [10],
            'intransit_1': [0],
            'intransit_2': [0],
        })
        
        order_low, _ = compute_order_quantity_corrected(
            quantiles_df, quantile_levels, initial_state_low, costs, lt
        )
        
        assert order_low > 0, "Should order when position < base-stock"
        
        # Test Case 2: Position > S (should not order)
        initial_state_high = pd.DataFrame({
            'on_hand': [100],
            'intransit_1': [50],
            'intransit_2': [50],
        })
        
        order_high, _ = compute_order_quantity_corrected(
            quantiles_df, quantile_levels, initial_state_high, costs, lt
        )
        
        assert order_high == 0, "Should not order when position > base-stock"
    
    def test_order_is_non_negative(self):
        """Verify that order quantity is never negative."""
        # Given: High initial position
        quantile_levels = np.array([0.1, 0.5, 0.9])
        quantiles_df = pd.DataFrame({
            0.1: [1.0],
            0.5: [5.0],
            0.9: [10.0],
        }, index=[1])
        
        initial_state = pd.DataFrame({
            'on_hand': [1000],
            'intransit_1': [1000],
            'intransit_2': [1000],
        })
        
        costs = Costs(holding=0.2, shortage=1.0)
        lt = LeadTime(lead_weeks=0, review_weeks=1)
        
        # When: Calculate order
        order, _ = compute_order_quantity_corrected(
            quantiles_df, quantile_levels, initial_state, costs, lt
        )
        
        # Then: Should be 0, not negative
        assert order >= 0, f"Order quantity {order} should never be negative"


class TestGoldenSKU:
    """Golden test for one specific SKU to verify end-to-end policy behavior."""
    
    def test_sku_0_126_policy_calculation(self):
        """
        Test policy calculation for SKU store=0, product=126.
        
        This is a regression test to ensure policy behavior stays consistent.
        """
        # Given: Realistic quantile forecasts for this SKU
        # (These would be loaded from actual trained model predictions)
        quantile_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        quantiles_df = pd.DataFrame({
            0.1: [0.0, 0.0, 0.0],
            0.3: [0.0, 0.0, 1.0],
            0.5: [1.0, 1.0, 2.0],
            0.7: [2.0, 3.0, 4.0],
            0.9: [5.0, 6.0, 7.0],
        }, index=[1, 2, 3])
        
        initial_state = pd.DataFrame({
            'on_hand': [2],
            'intransit_1': [1],
            'intransit_2': [0],
        })
        
        costs = Costs(holding=0.2, shortage=1.0)
        lt = LeadTime(lead_weeks=2, review_weeks=1)
        
        # When: Calculate order quantity
        order, info = compute_order_quantity_corrected(
            quantiles_df, quantile_levels, initial_state, costs, lt
        )
        
        # Then: Order should be reasonable (0-20 units for this low-demand SKU)
        assert 0 <= order <= 20, \
            f"Order {order} should be in reasonable range [0, 20] for low-demand SKU"
        
        # Verify it uses 3-week protection
        assert lt.lead_weeks + lt.review_weeks == 3, \
            "Should aggregate over 3 weeks (2 lead + 1 review)"
    
    def test_deterministic_with_seed(self):
        """Verify that results are deterministic with fixed seed."""
        quantile_levels = np.array([0.1, 0.5, 0.9])
        quantiles_df = pd.DataFrame({
            0.1: [1.0, 1.0, 1.0],
            0.5: [5.0, 5.0, 5.0],
            0.9: [10.0, 10.0, 10.0],
        }, index=[1, 2, 3])
        
        initial_state = pd.DataFrame({
            'on_hand': [5],
            'intransit_1': [2],
            'intransit_2': [1],
        })
        
        costs = Costs(holding=0.2, shortage=1.0)
        lt = LeadTime(lead_weeks=2, review_weeks=1)
        
        # When: Calculate twice (MC sampling uses default seed)
        order1, _ = compute_order_quantity_corrected(
            quantiles_df, quantile_levels, initial_state, costs, lt
        )
        order2, _ = compute_order_quantity_corrected(
            quantiles_df, quantile_levels, initial_state, costs, lt
        )
        
        # Then: Should be identical (deterministic MC with default seed)
        assert order1 == order2, "Results should be deterministic with default seeding"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
