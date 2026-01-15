import pandas as pd

data = {
    'recency_days': [0, 2, 3, 5, 1, 4],
    'frequency': [5, 3, 1, 1, 4, 1],
    'monetary': [1000, 500, 0, 0, 800, 0],
    'churn': [0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
print(df)

x = df[['recency_days', 'frequency', 'monetary']]
y = df['churn']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Predicted churn values:", y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))