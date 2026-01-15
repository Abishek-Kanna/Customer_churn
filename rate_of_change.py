import pandas as pd

grades = {
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun'],
    'grades': [88, 92, 85, 90, 95, 93]
}

df = pd.DataFrame(grades)
print(df)
df['rateof_change'] = df['grades'].diff()
print(df)