# Forecast Metrics Theory: Wasserstein Distance and Asymmetry in Inventory Planning

## Overview

This document covers theoretical foundations for evaluating density forecasts in inventory planning contexts, with emphasis on:

1. **Wasserstein (Earth Mover's) Distance** as a metric for comparing probability distributions
2. **Asymmetry** in inventory planning and how forecast metrics should align with value functions
3. **Practical recommendations** for metric selection and diagnostic tools

---

## Part I: Wasserstein Distance for Forecast Metrics

### Introduction

The Wasserstein distance (often referred to as earth mover's distance) is a metric for comparing probability distributions and can be directly applied to assess the quality of density forecasts by measuring how "close" the forecasted distribution is to the true empirical distribution. While classic metrics like accuracy, precision, and recall are designed for classification problems, analogous metrics for probabilistic or density forecasts are typically derived from the properties of such distances, and can be designed to reflect calibration, sharpness, and discrimination power of forecast densities.

### Accuracy, Precision, and Recall Analogues

#### Accuracy

In density forecasting, accuracy could be measured as the Wasserstein distance between the forecasted and observed distributions—smaller distances mean greater accuracy. This "distance to truth" is a natural generalization of point forecast error.

#### Precision and Recall Analogues

Precision and recall, in the classification context, relate to false positives and negatives. For density forecasts, similar concepts can be modeled by how mass is allocated relative to observed outcomes. For instance:

- **Calibration**: Whether the predicted probabilities match observed frequencies
- **Sharpness**: How concentrated the forecast distribution is

These can serve as analogues, and Wasserstein distance can be adapted to reflect both by examining portions of forecast distributions around the observed values.

### Value for Pinball Distributions

#### Pinball Loss vs. Wasserstein Distance

- The **pinball loss** is standard in quantile regression and forecasts, focusing on getting quantiles correct (e.g., the 90% quantile should be exceeded by observations 10% of the time).

- The **Wasserstein distance**, being a global metric, differs from pinball—the latter is local (quantiles), while the former is holistic (entire distributions). However, aggregating pinball losses across quantiles helps create consistency with Wasserstein-type metrics.

- A **value-weighted pinball distribution** could incorporate the magnitude of forecast errors into the loss, making it somewhat closer to Wasserstein's spirit, but the two measure different aspects:
  - **Wasserstein**: The "cost" of moving mass between distributions
  - **Pinball**: Asymmetric penalties for over- or under-prediction

- Thus, Wasserstein distance would be more useful if the goal is to assess the overall goodness of fit for the entire predictive distribution, while pinball loss is more focused on specific quantile coverage.

### Practical Usage and Recommendations

- **Wasserstein distance** offers a way to compare full forecast densities and would be useful for diagnosing systematic biases or over/under-confidence in forecast distributions.

- **Calibration plots** (PIT histograms) and **sharpness measures** could be combined with Wasserstein distances to create composite scoring metrics resembling "precision" and "recall" for probabilistic forecasts.

- **Pinball loss** remains indispensable for quantile-based prediction and evaluation, while Wasserstein offers a complementary global perspective.

- Using a **value-weighted pinball loss** could be meaningful if large errors are disproportionately more costly, but this would depart from the interpretability of sharpness and calibration in favor of a more utility-driven approach.

### Summary

Wasserstein distance can be applied to create analogues for accuracy (distance to truth), while extensions for precision/recall would require careful definitions based on calibration/sharpness or conditional densities. For pinball distributions, Wasserstein is more appropriate for distributional comparisons, while value-weighted pinball loss may serve specialized needs.

---

## Part II: Asymmetry in Inventory Planning

### Introduction

Asymmetry in inventory planning is a critical issue—stockouts often cause significant gross profit losses compared to the less severe marginal net profit reductions from excessive inventory. This effect is accentuated by the curvature of the profit/density function, as described by Jensen's Inequality in the newsvendor problem. Standard point forecast metrics (e.g., MAE, RMSE) do not capture this curvature or asymmetry, but advanced metrics and diagnostic tools can be leveraged for practical forecasting and inventory optimization.

### Pinball Loss and Value Alignment

Pinball loss (quantile loss) is inherently asymmetric and particularly suited to inventory forecasting because it penalizes under- and over-prediction differently, mirroring the real financial consequences faced in stockout and overstock scenarios. By optimizing for quantile-based forecasts (e.g., setting service levels tied to desired percentiles), you can tailor your forecasting model to align loss minimization with actual value functions—explicitly weighting the severe cost of stockouts versus the milder cost of holding excess stock.

#### Key Points

- The "reorder point" is essentially a quantile forecast, and pinball loss directly evaluates its accuracy through the lens of desired service levels and associated financial impacts.

- A **value-weighted pinball loss** or **threshold-weighted scoring rule** could help prioritize decisions around more extreme outcomes (chronic stockouts or clearance risk) by upweighting losses where economic impact is most pronounced.

#### Critical Fractile in Newsvendor Context

For inventory planning with:
- **Shortage cost** (cu) = 1.0€/unit
- **Holding cost** (co) = 0.2€/unit/week

The optimal service level (critical fractile) is:
```
τ* = cu / (cu + co) = 1.0 / (1.0 + 0.2) = 0.8333 (83.3rd percentile)
```

This means the optimal order quantity should target the 83.3rd percentile of the demand distribution, where the expected cost of a stockout equals the expected cost of overstocking.

### Precision, Recall, and Categorization

Density forecasts can be used to define thresholds (stockout, extreme overstock, etc.), transforming regression-style forecasts into binary or multi-category events:

- **Precision and recall**, although from classification, can be adapted by defining "positive" as a stockout (did the forecast correctly predict a stockout occurrence?) and "negative" as sufficient stock (did it avoid false positives for stockouts?).

- More nuanced classification—like distinguishing gradual and extreme stockouts or various degrees of overstock—allows different forecast thresholds along the continuum, reflecting the value function's breakpoints (customer loss, markdown, obsolescence).

### Sharpness and PIT Diagnostics

Sharpness and Probability Integral Transform (PIT) histograms are powerful diagnostic tools for density forecasts:

#### Sharpness

**Sharpness** reflects the concentration of forecasted distributions—crucial for inventory, where sharp (certain) predictions enable leaner stocking policies, but excessive sharpness increases risk if calibration is poor.

#### PIT Histograms

**PIT histograms** assess probabilistic calibration (do observed outcomes match forecast quantiles?), helping diagnose systematic biases and mismatches that might lead to unexpected stockouts or gluts.

#### Combined Use

Using sharpness and PIT in tandem allows for nuanced model comparison, ensuring that highly "sharp" forecasts are also well-calibrated, minimizing costly mismatches.

### Applied Forecasting Recommendations

1. **Adopt pinball loss (quantile loss)** as the central metric for evaluating and optimizing inventory forecasts, and experiment with value-weighted versions to capture economic nuances more faithfully.

2. **Regularly assess forecast calibration and sharpness** through PIT histograms and sharpness diagrams, and use these diagnostics to refine model parameters and service level selections.

3. **Employ precision and recall-type metrics** at defined decision thresholds—stockouts, extreme events, and overstock breakpoints—to measure the practical effectiveness of density forecasts in categorizing valuable inventory events.

4. **Consider asymmetric loss functions** (including modified pinball and custom threshold-weighted rules) to align forecast performance directly to profit functions and real business risk.

### Summary

These approaches provide robust, value-sensitive forecast metrics and diagnostics, helping ensure that inventory management is strategically optimized for both gross profit and risk management in practical, high-stakes supply chain environments.

---

## References

### Wasserstein Distance

1. [Probability Density Metrics](https://www.fabriziomusacchio.com/blog/2023-07-28-probability_density_metrics/)
2. [Earth Mover's Distance](https://www.johndcook.com/blog/2023/11/06/earth-movers-distance/)
3. [Integral Probability Metric](https://en.wikipedia.org/wiki/Integral_probability_metric)
4. [Calibrated Regression](https://proceedings.mlr.press/v162/kuleshov22a/kuleshov22a.pdf)
5. [Introduction to Quantile Loss](https://towardsdatascience.com/an-introduction-to-quantile-loss-a-k-a-the-pinball-loss-33cccac378a9/)
6. [Quantile Regression](https://mindfulmodeler.substack.com/p/how-i-made-peace-with-quantile-regression)
7. [Probabilistic Forecasting](https://www.annualreviews.org/doi/10.1146/annurev-statistics-032921-020240)
8. [Quantile Aggregation](https://www.stat.cmu.edu/~ryantibs/papers/quantagg.pdf)
9. [Value-Weighted Forecasts](https://arxiv.org/pdf/2404.17487.pdf)
10. [Calibrated Anomaly Detection](http://papers.neurips.cc/paper/7422-a-loss-framework-for-calibrated-anomaly-detection.pdf)
11. [Probability Distances](https://statistics.uchicago.edu/~lekheng/work/probdist.pdf)
12. [KLIC and Forecast Evaluation](https://faculty.ucr.edu/~taelee/paper/BaoLeeSaltoglu_KLIC.pdf)

### Asymmetry in Inventory Planning

1. [Asymmetric Loss Functions](https://arxiv.org/html/2505.00937v1)
2. [Asymmetric Loss in Forecasting](https://rady.ucsd.edu/_files/faculty-research/timmermann/asymmetric-loss.pdf)
3. [Quantile Loss Introduction](https://towardsdatascience.com/an-introduction-to-quantile-loss-a-k-a-the-pinball-loss-33cccac378a9/)
4. [Reorder Point Definition](https://www.lokad.com/reorder-point-definition/)
5. [Quantile Regression](https://mindfulmodeler.substack.com/p/how-i-made-peace-with-quantile-regression)
6. [Pinball Loss Function](https://www.lokad.com/pinball-loss-function-definition/)
7. [Value-Weighted Metrics](https://pubmed.ncbi.nlm.nih.gov/35707067/)
8. [Forecast Classification](https://www.sciencedirect.com/science/article/pii/S0377221721006500)
9. [Probabilistic Forecasts](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf)
10. [PIT Diagnostics](https://www.scss.tcd.ie/publications/tech-reports/reports.06/TCD-CS-2006-21.pdf)
11. [Calibration and Sharpness](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID1907799_code698671.pdf?abstractid=1907799&mirid=1)
12. [Forecast Evaluation](https://www.philadelphiafed.org/-/media/frbp/assets/working-papers/2011/wp11-5.pdf)

---

## Implementation Notes

### Current System Usage

This project implements several of these concepts:

- **Pinball loss** at critical fractile (0.8333) for model selection
- **Gaussian-weighted pinball** around the critical fractile (0.73-0.93 band) for challenger comparisons
- **PIT diagnostics** for calibration assessment
- **Coverage metrics** at multiple quantile levels (50%, 90%, 95%)
- **Cost-based evaluation** using expected cost under newsvendor optimization

See `docs/operations/model_evaluation.md` for implementation details.

### Future Enhancements

Potential additions based on this theory:

1. **Wasserstein distance** calculation for full distribution comparisons
2. **Value-weighted pinball loss** with custom weighting functions
3. **Threshold-based precision/recall** metrics for stockout classification
4. **Sharpness diagrams** alongside PIT histograms
5. **Composite scoring metrics** combining calibration, sharpness, and Wasserstein distance

