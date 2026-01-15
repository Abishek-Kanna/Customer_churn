import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("customer_churn_data.csv")

df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

LogisticRegression = LogisticRegression(max_iter=1000, class_weight="balanced")
RandomForest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

LogisticRegression.fit(X_train_scaled, y_train)
RandomForest.fit(X_train, y_train)
xgb.fit(X_train_scaled, y_train)

lr_prob = LogisticRegression.predict_proba(X_test_scaled)[:, 1]
rf_prob = RandomForest.predict_proba(X_test)[:, 1]
xgb_prob = xgb.predict_proba(X_test_scaled)[:, 1]

lr_pred = (lr_prob >= 0.5).astype(int)
rf_pred = (rf_prob >= 0.5).astype(int)
xgb_pred = (xgb_prob >= 0.5).astype(int)

lr_f1 = f1_score(y_test, lr_pred)
rf_f1 = f1_score(y_test, rf_pred)
xgb_f1 = f1_score(y_test, xgb_pred)


models = ["Logistic Regression", "Random Forest", "XGBoost"]
f1_scores = [lr_f1, rf_f1, xgb_f1]

plt.figure()
plt.bar(models, f1_scores)
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.show()

print("F1 Scores:")
for m, s in zip(models, f1_scores):
    print(m, ":", round(s, 3))


