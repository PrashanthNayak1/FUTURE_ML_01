# merge.py
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load actual and forecast data
actual_df = pd.read_csv("data/processed/sales_clean.csv")
forecast_df = pd.read_csv("data/processed/sales_forecast.csv")

# 2. Standardize column names
actual_df.rename(columns={"data": "date"}, inplace=True)
forecast_df.rename(columns={"ds": "date"}, inplace=True)

# 3. Convert date column to datetime
actual_df['date'] = pd.to_datetime(actual_df['date'])
forecast_df['date'] = pd.to_datetime(forecast_df['date'])

# 4. Merge datasets
merged_df = pd.merge(
    actual_df,
    forecast_df[['date', 'yhat']],
    on='date',
    how='inner'
)

# 5. Add date-based features
merged_df['weekday'] = merged_df['date'].dt.day_name()
merged_df['month_name'] = merged_df['date'].dt.month_name()

# Function to assign seasons (Northern Hemisphere)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

merged_df['season'] = merged_df['date'].dt.month.apply(get_season)

# 6. Plot Actual vs Forecast
plt.figure(figsize=(12, 6))
plt.plot(merged_df['date'], merged_df['venda'], label='Actual', marker='o')
plt.plot(merged_df['date'], merged_df['yhat'], label='Forecast', linestyle='--', marker='x')
plt.xlabel("Date")
plt.ylabel("Sales (venda)")
plt.title("Actual vs Forecast Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Save merged data for Power BI
merged_df.to_csv("data/processed/actual_vs_forecast.csv", index=False)
print("✅ Saved 'data/processed/actual_vs_forecast.csv' for Power BI")
