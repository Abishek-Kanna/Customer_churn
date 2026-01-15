import pandas as pd

usage_logs = {
    'log_id': [101,102,103,104],
    'customer_id': [1,1,2,3],
    'event_type': ['login','purchase','login','login'],
    'event_time': [ '2024-02-01',
        '2024-02-05',
        '2024-02-03',
        '2024-02-02'
        ]
}

df_logs = pd.DataFrame(usage_logs)
df_logs['event_time'] = pd.to_datetime(df_logs['event_time'])

reference_date = df_logs['event_time'].max()

recency = df_logs.groupby('customer_id')['event_time'].max().apply(lambda x: (reference_date - x).days).reset_index(name = 'recency_days') # lower recency is higher activity and higher recency is lower activity

frequency = df_logs.groupby('customer_id').size().reset_index(name='frequency')

value_map = {'login': 0, 'purchase': 500} # monetary value assigned to each event type

df_logs['monetary_value'] = df_logs['event_type'].map(value_map) # map event types to monetary values

monetary = df_logs.groupby('customer_id')['monetary_value'].sum().reset_index(name = 'monetary')

rfm = recency.merge(frequency, on='customer_id').merge(monetary, on='customer_id')


print(rfm)