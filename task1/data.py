# data.py
import pandas as pd
import os

# 1. Load the CSV
df = pd.read_csv("mock_kaggle.csv")

# 2. Drop missing values
df = df.dropna()

# 3. Convert 'data' to datetime
df['data'] = pd.to_datetime(df['data'], errors='coerce')

# 4. Remove rows with invalid dates
df = df.dropna(subset=['data'])

# 5. Group by month
monthly_sales = (
    df.groupby(df['data'].dt.to_period('M'))
    .agg({
        'venda': 'sum',
        'estoque': 'sum',
        'preco': 'mean'
    })
    .reset_index()
)
monthly_sales['data'] = monthly_sales['data'].dt.to_timestamp()

# 6. Feature Engineering
monthly_sales['venda_monthly_avg'] = monthly_sales['venda'].mean()
monthly_sales['venda_rolling_3m'] = monthly_sales['venda'].rolling(window=3, min_periods=1).mean()
monthly_sales['month_name'] = monthly_sales['data'].dt.month_name()
monthly_sales['weekday'] = monthly_sales['data'].dt.day_name()


# Seasons mapping
season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
monthly_sales['season'] = monthly_sales['data'].dt.month % 12 // 3 + 1
monthly_sales['season'] = monthly_sales['season'].map(season_map)

# Holiday months
holiday_dates = pd.to_datetime([
    '2022-01-01', '2022-10-24', '2022-12-25',
    '2023-01-01', '2023-10-12', '2023-12-25'
])
monthly_sales['is_holiday_month'] = monthly_sales['data'].isin(holiday_dates).astype(int)

# 7. Save processed file
os.makedirs("data/processed", exist_ok=True)
monthly_sales.to_csv("data/processed/sales_clean.csv", index=False)

print("✅ Data saved with features:", monthly_sales.head())
 