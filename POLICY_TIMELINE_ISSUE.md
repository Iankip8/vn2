# CRITICAL FINDING: Policy Implementation Issue

**Date:** February 4, 2026

## The Problem

After examining the evaluation code in `model_eval.py` and the simulator in `sim/core.py`, I found a **critical mismatch** between how the policy is implemented and how the simulation works.

---

## How the Simulation Actually Works

From `sim/core.py` lines 40-80:

### Timeline in the Simulator:

**Week t (current week):**
1. **Start of week**: Receive `intransit_1` (order placed 2 weeks ago at t-2)
2. **During week**: Demand arrives, sell from `on_hand`
3. **End of week**: Place order → goes to `intransit_2`

**Week t+1 (next week):**
1. `intransit_2` → `intransit_1` (order moves closer)
2. Receive nothing (order placed at t-2 hasn't arrived yet... wait that's wrong)

Wait, let me trace this more carefully:

```python
# Week t: Current week
received = state["intransit_1"]  # Receive order from t-2
on_hand = state["on_hand"] + received
sold = min(demand, on_hand)
on_hand = on_hand - sold

# Pipeline update:
next_state["intransit_1"] = state["intransit_2"]  # t-1 order moves up
next_state["intransit_2"] = order  # Today's order enters pipeline
```

So the pipeline is:
- **intransit_2**: Arrives in 2 weeks (lead_weeks=2)
- **intransit_1**: Arrives in 1 week (lead_weeks=1)
- **Order today** → intransit_2 → intransit_1 → on_hand

This means lead time is EXACTLY 2 weeks.

---

## How the Policy is Applied in Evaluation

From `model_eval.py` lines 500-520:

```python
for step in range(horizon):
    demand_t = y_true[step]
    
    # Week 0: place computed order; after: zero recourse
    if step == 0:
        order_t = order_qty  # Place our one-time order
    else:
        order_t = 0.0        # NO MORE ORDERS!
    
    state, cost = sim.step(state, demand_t, order_t)
```

**THIS IS THE ISSUE!**

The evaluation:
1. Places ONE order at step=0
2. Then places ZERO orders for all future steps
3. Evaluates costs over the entire horizon (12 weeks typically)

But the corrected policy:
- Aggregates demand over 3 weeks (h=1, h=2, h=3)
- Orders to protect those 3 weeks
- **But the order won't arrive until week 2** (lead_time=2)!

---

## The Timeline Mismatch

### What the Policy Thinks:

```
Week 0: Place order for protection_weeks = 3
  → Need to cover demand in weeks 1, 2, 3
  → Aggregate D₁ + D₂ + D₃
  → Order Q = 0.833 quantile of (D₁ + D₂ + D₃)
```

### What Actually Happens:

```
Week 0: Place order Q
Week 1: Order in intransit_2, demand D₁ happens
        → Must satisfy from existing on_hand
        → Order Q doesn't help!
Week 2: Order moves to intransit_1, demand D₂ happens
        → Still must satisfy from existing on_hand
        → Order Q still doesn't help!
Week 3: Order Q arrives! Now on_hand increases
        → But demand D₁ and D₂ already caused shortages
        → Order only helps for D₃, D₄, D₅... onwards
```

---

## Why This Breaks Everything

### Buggy Policy (1-week protection):
- Aggregates only D₁ (h=1)
- Orders small amount
- By week 3, order arrives and... wait, the order is too small!
- But horizon is only evaluated for 2 steps (h=1, h=2)
- So the order never arrives during evaluation!

### Corrected Policy (3-week protection):
- Aggregates D₁ + D₂ + D₃
- Orders LARGE amount (3x more)
- But order arrives at week 3
- Evaluation runs over 12 weeks
- Weeks 1-2: Massive shortages (order not arrived yet)
- Week 3+: Massive holding costs (big order sitting)
- **Total cost is WORSE!**

---

## The Real Question

**What should the protection period be?**

### Option 1: Protection = Review Period Only (R=1)
- Next review at week 1
- Next order arrives at week 3
- Current order covers weeks 3-3 = just week 3
- **Protection = 1 week**

But this doesn't make sense with (R,S) policy theory...

### Option 2: Protection = Lead + Review (L+R=3)  
- Next review at week 1
- Next order arrives at week 3
- Current order covers weeks 3-4-5
- **Protection = 3 weeks starting from when order arrives**

But then we need to aggregate h=3, h=4, h=5, not h=1, h=2, h=3!

### Option 3: The Evaluation is Wrong
- Should we be placing orders every week (not just at step=0)?
- Is this a "single-period newsvendor" evaluation instead of multi-period (R,S)?

---

## Why Seasonal Naive was Best with Buggy Policy

```
Buggy policy:
- Used only h=1 forecast
- Seasonal naive is excellent at h=1
- Ordered small amount based on h=1
- Order never arrived during 2-step evaluation
- Low holding cost (small order)
- Moderate shortage cost
- Total: 10,251

Corrected policy:
- Used h=1+h=2+h=3 forecast  
- Seasonal naive degrades at h=2, h=3
- Ordered 3x more based on aggregated demand
- Order arrives week 3, holds through week 12
- Massive holding cost
- Shortage cost still present (weeks 1-2)
- Total: 36,868 (+260%!)
```

---

## The Critical Insight

**The evaluation is simulating a SINGLE ORDER decision, not a periodic review system!**

This is more like a **newsvendor problem** than a true **(R,S) policy**.

In newsvendor:
- Place one order
- It arrives after lead time L
- Covers demand for the next period

So the protection should be:
- **1 period** (the next review period R=1)
- But starting from when the order arrives (week L+1 = week 3)
- So aggregate h=3 only, not h=1,2,3!

OR we need to change the evaluation to place orders every week, not just once.

---

## Next Steps

1. **Check with Patrick**: Which interpretation is correct?
   - Is this newsvendor (single order)?
   - Or periodic review (order every week)?

2. **Test the h=3 hypothesis**:
   - Modify policy to use h=L+R (h=3 only)
   - See if this improves service level

3. **Or fix evaluation**:
   - Place orders every step, not just step=0
   - Implement true (R,S) policy

4. **Re-check Patrick's recommendations**:
   - Maybe his "3-week protection" assumes orders are placed every week?
   - Or he meant something different?
