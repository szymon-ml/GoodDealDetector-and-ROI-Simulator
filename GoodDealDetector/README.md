# GoodDealDetector: Identifying Houses with High Flipping Potential

## A tool to narrow searches for profitable real estate flips, simulate ROI from renovations, and help homeowners value their homes.

---

## 1. What This Project Does

This project tackles a common challenge for real estate investors: identifying houses that are undervalued and have high renovation potential. It works in two stages:

**Regression:** An XGBoost model predicts the intrinsic value of a house based on its features.  

**Classification:** Using the predicted intrinsic value, a binary `Flipping_Potential` target is engineered to label houses likely to be profitable flips. A classifier then predicts which properties meet this criteria.

The result is a model that can flag high-potential flips from hundreds of listings, reducing time and risk for investors.

---

## 2. Business Value / Impact

- **For Realtors & Investors:** Quickly prioritize house with high ROI potential.  
- **For Buyers & Renovators:** Provide data-backed ROI simulation and insight into which renovations are likely to yield the highest returns. (for the ROI Simulator, check out the other folder in this repository)
- **Practical Deployment:** Although trained on 2006–2010 Ames Housing data, the pipeline is generalizable. With updated, modern data, the methodology can support current market decisions.  

This tool turns raw housing data into actionable investment insights, serving as a filtering layer before human evaluation.

---

## 3. Key Results

The final `XGBClassifier` focuses on the minority “Good Flip” class:

- **Precision (Good Flip):** 87%  
- **Recall (Good Flip):** 100%  
- **F1 Score (Good Flip):** 93%  

**Interpretation:**

- **Recall = 100%** → All potential flips are captured, minimizing missed opportunities.  
- **Precision = 87%** → When flagged as a flip, there’s a high likelihood it’s actually profitable, reducing manual review effort.

---

## 4. Limitations / Assumptions

- The `Flipping_Potential` target is engineered using regression output, not direct investor labeling. This inflates metrics like F1, precision, and recall.  
- Without a mathematically derived target, metrics would likely be lower. The model mainly learned to reverse-engineer the formula used to define a “good flip.” In practice, an actual investor would provide these labels, allowing the model to capture subjective decisions and nuanced preferences that aren’t purely mathematical.  
- Feature contributions are dominated by the `Underpriced` variable, reflecting the engineered logic behind the target. Metrics primarily measure the model’s ability to reproduce domain-informed signals, not purely raw data signals.  
- The model assumes market conditions at purchase resemble those at sale. Real-world deployment requires up-to-date sales data.

---

## 5. Process Summary

The pipeline begins with exploratory data analysis on key house characteristics (overall quality, seasonality, neighborhood, renovation potential). An XGBoost regression model estimates intrinsic value. Residuals and engineered features are then used to derive the `Flipping_Potential` target. Classification models (Logistic Regression, Random Forest, XGBoost) are trained using custom scoring metrics to handle class imbalance and emphasize correct identification of profitable flips. The final model combines domain-informed features and raw data, producing actionable, investor-focused predictions.

---

## 6. Key Vizualizations

### 6.1 Intrinsic Value vs Market Price Scatter (predicted vs actual scores of the regressor)

### Regression Model Feature Importance

###

### 6.1 Classification Model's Confusion Matrix

### 6.2 Classification Model's Ability to Separate Classes

### 6.3 Classification Model Feature Importance

### 6.4