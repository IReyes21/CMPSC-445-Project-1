import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# 1. LOAD DATA
# ==============================
temp_df = pd.read_csv("data/GLB.Ts+dSST.csv", skiprows=1)
noaa_df = pd.read_csv("data/co2_mm_mlo.csv", comment='#')
owid_df = pd.read_csv("data/owid-co2-data.csv")

# ==============================
# 2. CLEAN NASA TEMP (MONTHLY)
# ==============================
temp_df = temp_df.melt(
    id_vars=['Year'],
    value_vars=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    var_name='month_name',
    value_name='temp'
)

temp_df['temp'] = pd.to_numeric(temp_df['temp'], errors='coerce')

month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
             'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

temp_df['month'] = temp_df['month_name'].map(month_map)
temp_df['date'] = pd.to_datetime(dict(year=temp_df['Year'], month=temp_df['month'], day=1))

temp_df = temp_df[['date','temp']].dropna()

# ==============================
# 3. CLEAN NOAA CO2
# ==============================
noaa_df.columns = noaa_df.columns.str.strip().str.lower()

noaa_df = noaa_df[['year','month','average']]
noaa_df = noaa_df.rename(columns={'average':'co2_noaa'})
noaa_df = noaa_df[noaa_df['co2_noaa'] > 0]

noaa_df['date'] = pd.to_datetime(dict(year=noaa_df['year'], month=noaa_df['month'], day=1))
noaa_df = noaa_df[['date','co2_noaa']]

# ==============================
# 4. CLEAN OWID
# ==============================
owid_world = owid_df[owid_df['country'] == 'World'].copy()

owid_cols = ['year', 'co2', 'co2_growth_prct', 'methane', 'nitrous_oxide', 'energy_per_capita']
owid_world = owid_world[owid_cols].dropna()

# ==============================
# 5. MERGE DATA
# ==============================
df = pd.merge(temp_df, noaa_df, on='date', how='inner')

df['year'] = df['date'].dt.year
df = pd.merge(df, owid_world, on='year', how='inner')

df = df.sort_values('date').reset_index(drop=True)

print("Total samples BEFORE features:", len(df))

# ==============================
# 6. FEATURE ENGINEERING
# ==============================
df['co2_growth'] = df['co2_noaa'].pct_change()
df['methane_growth'] = df['methane'].pct_change()

df['co2_ma12'] = df['co2_noaa'].rolling(12).mean()
df['temp_ma12'] = df['temp'].rolling(12).mean()

df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

df = df.dropna()

print("Total samples AFTER cleaning:", len(df))

# ==============================
# 7. MODELING
# ==============================
X = df.drop(columns=['temp', 'date', 'year'])
y = df['temp']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ==============================
# 8. LINEAR REGRESSION
# ==============================
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

# Feature importance (LR)
lr_importance = pd.Series(lr.coef_, index=X.columns).sort_values(key=abs, ascending=False)

# ==============================
# 9. RANDOM FOREST
# ==============================
rf = RandomForestRegressor(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# ==============================
# 10. RESULTS
# ==============================
print("\n===== MODEL PERFORMANCE =====")

print("\nLinear Regression:")
print("R2:", r2_score(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))

print("\nRandom Forest:")
print("R2:", r2_score(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))

print("\n===== FEATURE IMPORTANCE =====")

print("\nTop Features (Linear Regression):")
print(lr_importance.head(10))

print("\nTop Features (Random Forest):")
print(rf_importance.head(10))

print("\nTop Driver of Temperature (RF):")
print(rf_importance.idxmax())

# ==============================
# 11. VISUALIZATIONS
# ==============================

# ---- Dual Axis Plot ----
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Date')
ax1.set_ylabel('Temp Anomaly', color='red')
ax1.plot(df['date'], df['temp'], color='red', alpha=0.5)

ax2 = ax1.twinx()
ax2.set_ylabel('CO2 (ppm)', color='blue')
ax2.plot(df['date'], df['co2_noaa'], color='blue')

plt.title("Temperature vs CO2 (Dual Axis)")
plt.show()

# ---- Smoothed Trends ----
plt.figure(figsize=(10,5))
plt.plot(df['date'], df['temp_ma12'], label='Temp (12-mo avg)')
plt.plot(df['date'], df['co2_ma12'], label='CO2 (12-mo avg)')
plt.legend()
plt.title("Smoothed Trends (12-Month Moving Average)")
plt.show()

# ---- Feature Importance Plot ----
plt.figure(figsize=(10,5))
rf_importance.head(10).sort_values().plot(kind='barh')
plt.title("Top Features (Random Forest)")
plt.show()

# ---- Scatter Plot ----
plt.figure()
plt.scatter(df['co2_noaa'], df['temp'])
plt.xlabel("CO2")
plt.ylabel("Temperature")
plt.title("CO2 vs Temperature")
plt.show()

# ---- Correlation Heatmap ----
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# ==============================
# 12. SAVE DATA
# ==============================
df.to_csv("processed_climate_data.csv", index=False)
