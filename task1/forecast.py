 # forecast.py
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/sales_clean.csv")

# 2. Prepare for Prophet
prophet_df = df.rename(columns={'data': 'ds', 'venda': 'y'})
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

# 3. Train Prophet model
m = Prophet()
m.fit(prophet_df)

# 4. Forecast next 12 months
future = m.make_future_dataframe(periods=12, freq='MS')
forecast = m.predict(future)

# 5. Plot forecast
m.plot(forecast)
plt.show()

# 6. Plot trend & seasonality
m.plot_components(forecast)
plt.show()

# 7. Save forecast results
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("data/processed/sales_forecast.csv", index=False)
print("✅ Forecast saved to data/processed/sales_forecast.csv")
