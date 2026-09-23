# Farmer Income Prediction

**IBM SkillsBuild Virtual Internship Project**  
**Student:** Aditya Chaudhary

---

## Project Objective

Build a machine learning regression model to predict the total income of farmers (agriculture + non-agriculture) based on land, regional, demographic, and environmental features.

---

## Dataset

- **Source:** Kaggle — [manishpatil115/farmer-income-prediction-dataset](https://www.kaggle.com/datasets/manishpatil115/farmer-income-prediction-dataset)
- **Training Samples:** 47,970 rows, 105 columns
- **Target Variable:** `Target_Variable/Total Income` — Total annual income in INR

---

## Technologies Used

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- kagglehub

---

## Project Workflow

1. Load dataset using KaggleHub
2. Understand columns and target variable
3. Basic data cleaning (missing values, duplicates, drop non-informative columns)
4. Exploratory Data Analysis (5 plots)
5. Label encode categorical variables
6. Train/test split (80/20)
7. Train 3 models
8. Evaluate with MAE, RMSE, R²
9. Compare models and select best
10. Sample prediction
11. Conclusion

---

## Models Used

| Model | Description |
|---|---|
| Linear Regression | Baseline linear model |
| Decision Tree Regressor | Tree-based model capturing non-linearity |
| Random Forest Regressor | Ensemble of 100 trees (best model) |

---

## Evaluation Metrics

| Model | MAE (INR) | RMSE (INR) | R² |
|---|---|---|---|
| Linear Regression | 3,53,131.80 | 6,30,068.62 | 0.9100 |
| Decision Tree | 3,41,366.79 | 7,70,110.95 | 0.8656 |
| **Random Forest** | **2,60,292.32** | **5,22,759.80** | **0.9381** |

**Best Model: Random Forest Regressor (R² = 0.9381)**

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```bash
   jupyter notebook AdityaChaudhary_FarmerIncomePrediction.ipynb
   ```
3. Run all cells in order. The dataset is downloaded automatically via KaggleHub (requires Kaggle API credentials set up, or uses local cache).

---

## Conclusion

Random Forest Regressor is the best model with R² = 0.9381, meaning it explains ~94% of the variance in farmer income. Non-Agriculture Income and Total Land for Agriculture are the most important predictors. This model can support financial institutions in data-driven farmer loan assessment.
