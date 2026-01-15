# Customer Churn Prediction (ML Model Comparison)

## Overview
This project predicts **customer churn** using machine learning and compares multiple classification models to identify which performs best. Churn prediction helps businesses retain customers by identifying those likely to leave.

## Tech Stack
- Python  
- pandas, numpy  
- matplotlib  
- scikit-learn  
- xgboost  

## Dataset
- File: `customer_churn_data.csv`
- Type: Tabular customer data
- Target column: `Churn`
  - 1 → Customer churned
  - 0 → Customer did not churn

### Example Features
- tenure
- MonthlyCharges
- TotalCharges
- Contract type
- Payment method
- Internet service

Categorical features are converted using **one-hot encoding**.


## Models Used
- Logistic Regression  
- Random Forest  
- XGBoost  

All models are trained on the same train–test split for fair comparison.

## Evaluation Metric
- **F1-score**  
Chosen because the dataset is imbalanced and accuracy alone is misleading.

## Results
A bar chart is used to compare the **F1-scores** of all three models.

- Logistic Regression → baseline performance  
- Random Forest → better handling of non-linear patterns  
- XGBoost → generally performs best due to boosting  

F1 Scores:
Logistic Regression : 0.472
Random Forest : 0.471
XGBoost : 0.499

![alt text](<Screenshot 2026-01-15 135339.png>)

It serves as a **baseline churn prediction project** that can be improved further.


