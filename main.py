# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2. LOAD DATA
temp_df = pd.read_csv("data/GLB.Ts+dSST.csv", skiprows=1)
co2_df = pd.read_csv("data/owid-co2-data.csv")

# 3. CLEAN TEMPERATURE DATA
# Keep only Year + annual temp anomaly
temp_df = temp_df[['Year', 'J-D']]   # J-D = yearly anomaly
temp_df.columns = ['year', 'temp']

temp_df = temp_df.dropna()

# 4. CLEAN OWID DATA
# Use global data only
co2_df = co2_df[co2_df['country'] == 'World']

# Select useful features
co2_df = co2_df[[
    'year',
    'co2',
    'co2_growth_prct',
    'methane',
    'nitrous_oxide',
    'energy_per_capita',
    'co2_per_capita'
]]

co2_df = co2_df.dropna()

# 5. MERGE DATA
df = pd.merge(temp_df, co2_df, on='year')

# 6. FEATURE ENGINEERING
# Growth features
df['co2_growth'] = df['co2'].diff()
df['methane_growth'] = df['methane'].diff()

# Moving averages
df['co2_ma'] = df['co2'].rolling(5).mean()
df['temp_ma'] = df['temp'].rolling(5).mean()

# Time feature
df['years_since_1950'] = df['year'] - 1950

# Drop NaNs after rolling/diff
df = df.dropna()

# 7. DEFINE FEATURES / TARGET
X = df.drop(columns=['temp', 'year'])
y = df['temp']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 9. LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Linear Regression Results")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2:", r2_score(y_test, y_pred_lr))

# Feature importance (coefficients)
lr_importance = pd.Series(lr.coef_, index=X.columns).sort_values(key=abs, ascending=False)

print("\nLinear Regression Feature Importance:")
print(lr_importance)

# 10. RANDOM FOREST
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2:", r2_score(y_test, y_pred_rf))

rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance)

# 11. VISUALIZATIONS
# ---- Time Series ----
plt.figure()
plt.plot(df['year'], df['temp'], label='Temperature')
plt.plot(df['year'], df['co2'], label='CO2')
plt.legend()
plt.title("Temperature vs CO2 Over Time")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

# ---- Feature Importance (LR) ----
plt.figure()
lr_importance.head(10).plot(kind='bar')
plt.title("Top Features (Linear Regression)")
plt.xticks(rotation=45)
plt.show()

# ---- Feature Importance (RF) ----
plt.figure()
rf_importance.head(10).plot(kind='bar')
plt.title("Top Features (Random Forest)")
plt.xticks(rotation=45)
plt.show()

# ---- Scatter Plot ----
plt.figure()
plt.scatter(df['co2'], df['temp'])
plt.xlabel("CO2")
plt.ylabel("Temperature")
plt.title("CO2 vs Temperature")
plt.show()

# ===============================
# 12. SAVE PROCESSED DATA
df.to_csv("processed_climate_data.csv", index=False)