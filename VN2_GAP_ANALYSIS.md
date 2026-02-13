# VN2 Approach Analysis - Why We Can't Match €5,248

## The Critical Differences

### VN2's Actual Approach (from VN2.py)

```python
# Step 1: Forecast 10 future periods (not just 3-5!)
f_periods = pd.date_range(start=sales.columns[-1], periods=10, 
                          inclusive="neither", freq="W-MON")

# Step 2: Create forecast for ALL 10 periods
forecast = base_forecast × seasonal_factors  # Same forecast for all 10 weeks

# Step 3: Order up to = SUM OF FIRST 4 WEEKS
order_up_to = forecast.iloc[:,:4].sum(axis=1)  # Sum weeks 1,2,3,4

# Step 4: Simple order-up-to policy
order = max(0, order_up_to - net_inventory)
```

**Key Points:**
1. Forecasts **10 periods ahead** (weeks 1-10)
2. Uses **deterministic point forecast** (no PMFs, no quantiles!)
3. **Sums first 4 weeks** of forecast: `forecast.iloc[:,:4].sum(axis=1)`
4. **Direct subtraction** from net inventory

### Our Approach (What We're Doing)

```python
# Step 1: Generate forecasts for h=1 to 12
forecasts = forecaster.predict(...)  # Returns quantiles for h=1..12

# Step 2: Extract h=3,4 PMFs (only 2 horizons!)
h3_pmf = quantiles_to_pmf(h3_quantiles)
h4_pmf = quantiles_to_pmf(h4_quantiles)

# Step 3: Run complex backtest with PMF-based optimization
result = run_12week_backtest(
    forecasts_h1=h3_pmf,  # Only h=3
    forecasts_h2=h4_pmf,  # Only h=4
    ...
)
```

**Key Points:**
1. We use **h=3,4** (only 2 periods, not 4!)
2. We convert to **PMFs** (probabilistic, not deterministic)
3. We use **complex backtest** with optimization
4. We're missing h=5,6 (the 3rd and 4th coverage weeks!)

---

## The Root Cause of €9,193 vs €5,248

### Problem 1: Wrong Horizons ❌

**VN2 covers weeks t+1, t+2, t+3, t+4:**
```
Week t (order placed):
  - Order arrives at t+2
  - Covers demand at t+1, t+2, t+3, t+4
  - 4 weeks total coverage
```

**We cover only h=3,4:**
```
Week t (order placed):
  - h=3: Arrival week (t+2 seen from t-1)
  - h=4: Next week (t+3 seen from t-1)
  - Missing h=1,2,5,6!
```

**We're only covering 2 weeks when VN2 covers 4!**

### Problem 2: Deterministic vs Probabilistic ❌

**VN2:**
- Uses **single point forecast** (13-week MA)
- Direct sum: `forecast[week1] + forecast[week2] + forecast[week3] + forecast[week4]`
- Simple, deterministic

**We:**
- Convert to **quantile distributions**
- Convert quantiles to **PMFs**
- Run **probabilistic optimization**
- Much more complex, loses VN2's simplicity

### Problem 3: Week Alignment ❌

**VN2's timeline:**
```
Current week: t
Order placed: start of week t
Order arrives: start of week t+2
Coverage needed: weeks t+1, t+2, t+3, t+4 (from placement perspective)

From VN2.py:
f_periods = date_range(start=sales.columns[-1], periods=10, ...)
# This is AFTER the last observed week
# So f_periods[0] = t+1, f_periods[1] = t+2, etc.
order_up_to = forecast.iloc[:,:4].sum()  # Sums t+1, t+2, t+3, t+4
```

**Our timeline:**
```
We use h=3,4 where:
- h=3 is supposed to be "arrival week"
- h=4 is "next week"
But this doesn't align with VN2's simple "next 4 weeks" logic
```

---

## Why This Matters - Cost Impact

### VN2 (€5,248):
```python
# Example SKU:
base_forecast = 0.77  # 13-week recent average
order_up_to = 4 × 0.77 = 3.08
order = max(0, 3.08 - net_inventory)
```

### Our Approach (€9,193):
```python
# We're using:
# - Wrong horizons (h=3,4 instead of next 4 weeks)
# - PMF conversion adds noise
# - Complex optimization over-orders
# Result: ~2x higher order quantities
```

---

## How to Fix It

### Option 1: Pure Deterministic VN2 Replication ✓

```python
# In our forecaster.predict():
# Return DETERMINISTIC forecasts for h=1,2,3,4 (next 4 weeks)
# No quantiles, no PMFs!

def predict_vn2_exact(self, history, forecast_date):
    # De-seasonalize
    deseason = history / seasonal_factors
    
    # 13-week MA
    base = deseason[-13:].mean()
    
    # Re-seasonalize for next 4 weeks
    next_4_weeks = [forecast_date + timedelta(weeks=i) for i in range(1,5)]
    forecasts = [base × seasonal_factor[week] for week in next_4_weeks]
    
    # Order up to = sum
    order_up_to = sum(forecasts)
    
    # Simple policy
    order = max(0, order_up_to - net_inventory)
    
    return order  # Not PMFs!
```

### Option 2: Skip PMF Conversion

The issue is we're forcing everything through the PMF framework when VN2 doesn't use PMFs at all!

**VN2 is purely deterministic:**
- One forecast value per week
- Sum 4 weeks
- Done

**We're making it probabilistic:**
- 13 quantiles per week
- Convert to PMF
- Optimize over distribution
- Loses VN2's simplicity

---

## The Fundamental Mismatch

**VN2.py Structure:**
```
For each decision week:
  1. Get historical sales up to now
  2. Compute 13-week MA (de-seasonalized)
  3. Project forward 10 weeks (re-seasonalized)
  4. Sum first 4: order_up_to
  5. Order = max(0, order_up_to - inventory)
  6. Update state with actual demand
  7. Repeat
```

**Our Structure:**
```
For each decision week:
  1. Generate quantile forecasts (h=1..12)
  2. Extract h=3,4,5 (wrong!)
  3. Convert to PMFs
  4. Run complex optimization
  5. Get order from BacktestState
  6. Repeat
```

**They're completely different algorithms!**

---

## Why €9,193 is Consistent

Our result of €9,193 is very close to:
- Simple 3-week: €9,543
- Simple 4-week: €10,194
- Our models: €10,045

All these are in the **€9K-10K range** because they all:
- Use model forecasts (not pure 13-week MA)
- Use some form of probabilistic/complex ordering
- Cover 3-4 weeks (close to VN2's 4)

The €5,248 is different because:
- Pure 13-week MA (adapts fastest to demand changes)
- Deterministic (no variance explosion)
- Exact 4-week coverage
- Simplest possible policy

---

## Recommendation

To match €5,248, we need to:

1. **Bypass PMF framework entirely**
2. **Implement pure deterministic VN2 logic**
3. **Use next 4 weeks (h=1,2,3,4 from decision point)**
4. **Simple sum, no optimization**

The PMF-based backtest framework is designed for sophisticated models.
VN2's winning approach deliberately avoids sophistication!

**Bottom line:** We can't match VN2 using run_12week_backtest() because that function is fundamentally designed for PMF-based probabilistic forecasting, while VN2 uses pure deterministic forecasting with simple arithmetic.
