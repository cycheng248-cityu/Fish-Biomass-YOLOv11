# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:51:17 2026

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# 1. Load the data
df = pd.read_csv('fish_physical_measurement.csv')
df.columns = df.columns.str.strip()

# Constant for water/fish tissue density (approx 1.0 g/cm^3)
rho = 1.0  

# 2. Calculate Dimensionless Ratios
df['Aspect_Ratio'] = df['Height'] / df['Length']
df['Thickness_Ratio'] = df['Thickness'] / df['Length']

# 3. Calculate Theoretical Volume & Baseline Mass
df['Theoretical_Volume'] = (np.pi / 6) * df['Length'] * df['Height'] * df['Thickness']

# 4. Calculate Perfect K for every individual fish
df['Perfect_K'] = df['True_Mass'] / (rho * df['Theoretical_Volume'])

# 5. Set up ML Data for Ridge Regression
X = df[['Aspect_Ratio', 'Thickness_Ratio']].values
y = df['Perfect_K'].values

ridge_model = Ridge(alpha=1.0)
loo = LeaveOneOut()
predicted_masses = []

# ==========================================
# PART 1: LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
# ==========================================
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model on the remaining 9 fish
    ridge_model.fit(X_train, y_train)
    
    # Predict the K for the 1 hidden fish
    predicted_K = ridge_model.predict(X_test)[0]
    
    # Convert that predicted K back into a predicted Mass
    hidden_fish_vol = df.iloc[test_index[0]]['Theoretical_Volume']
    predicted_mass = predicted_K * rho * hidden_fish_vol
    predicted_masses.append(predicted_mass)

# Add the estimated masses to our DataFrame
df['Estimated_Mass_With_K'] = predicted_masses

# ==========================================
# PART 2: METRICS CALCULATION
# ==========================================
mape = mean_absolute_percentage_error(df['True_Mass'], df['Estimated_Mass_With_K'])
rmse = np.sqrt(mean_squared_error(df['True_Mass'], df['Estimated_Mass_With_K']))
r2 = r2_score(df['True_Mass'], df['Estimated_Mass_With_K'])

print("--- VALIDATION METRICS ---")
print(f"MAPE (Mean Absolute Percentage Error): {mape * 100:.2f}%")
print(f"RMSE (Root Mean Square Error):         {rmse:.2f} g")
print(f"R-squared (Coefficient of Det.):       {r2:.2f}")

# ==========================================
# PART 3: THE FINAL DEPLOYMENT EQUATION
# ==========================================
# Train on ALL 10 fish to get the final formula for your mobile app
ridge_model.fit(X, y)

alpha = ridge_model.coef_[0]
beta = ridge_model.coef_[1]
gamma = ridge_model.intercept_

print(f"\n--- FINAL DYNAMIC SHAPE FACTOR EQUATION ---")
print(f"K = ({alpha:.4f} * Aspect_Ratio) + ({beta:.4f} * Thickness_Ratio) + {gamma:.4f}")

# ==========================================
# PART 4: SAVE NEW CSV FILE
# ==========================================
output_filename = 'fish_mass_estimation_results.csv'
df.to_csv(output_filename, index=False)
print(f"\n Success: Predictions and metrics appended. Saved to '{output_filename}'")


#%%

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


# 2. Logarithmic Transformation for the Power Curve (W = a * L^b)
# We linearize the equation to: log(W) = log(a) + b * log(L)
df['log_Length'] = np.log(df['Length'])
df['log_Mass'] = np.log(df['True_Mass'])

# 3. Set up ML Data for Linear Regression
X = df[['log_Length']].values
y = df['log_Mass'].values

# Standard Linear Regression to solve the linearized power curve
linear_model = LinearRegression()
loo = LeaveOneOut()
predicted_masses_power = []

# ==========================================
# PART 1: LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
# ==========================================
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the power curve on the remaining 9 fish
    linear_model.fit(X_train, y_train)
    
    # Predict log(Mass) for the 1 hidden fish
    predicted_log_mass = linear_model.predict(X_test)[0]
    
    # Reverse the log transform (exponentiate) to get actual predicted Mass in grams
    predicted_mass = np.exp(predicted_log_mass)
    predicted_masses_power.append(predicted_mass)

# Add the estimated masses to our DataFrame
df['Estimated_Mass_PowerCurve'] = predicted_masses_power

# ==========================================
# PART 2: METRICS CALCULATION
# ==========================================
mape = mean_absolute_percentage_error(df['True_Mass'], df['Estimated_Mass_PowerCurve'])
rmse = np.sqrt(mean_squared_error(df['True_Mass'], df['Estimated_Mass_PowerCurve']))
r2 = r2_score(df['True_Mass'], df['Estimated_Mass_PowerCurve'])

print("--- TRADITIONAL POWER CURVE METRICS ---")
print(f"MAPE (Mean Absolute Percentage Error): {mape * 100:.2f}%")
print(f"RMSE (Root Mean Square Error):         {rmse:.2f} g")
print(f"R-squared (Coefficient of Det.):       {r2:.2f}")

# ==========================================
# PART 3: THE FINAL DEPLOYMENT EQUATION
# ==========================================
# Train on ALL 10 fish to extract the final 'a' and 'b' coefficients
linear_model.fit(X, y)

# The slope is 'b'
b = linear_model.coef_[0]
# The intercept is log(a), so we exponentiate it to get 'a'
a = np.exp(linear_model.intercept_)

print(f"\n--- FINAL TRADITIONAL POWER CURVE EQUATION ---")
print(f"Mass = {a:.4f} * Length ^ {b:.4f}")

# ==========================================
# PART 4: SAVE NEW CSV FILE
# ==========================================
output_filename = 'fish_power_curve_results.csv'
df.to_csv(output_filename, index=False)
print(f"\n Success: Predictions and metrics appended. Saved to '{output_filename}'")













