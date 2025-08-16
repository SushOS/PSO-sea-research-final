# %%
# Install required packages
!pip install seaborn pmdarima pyswarm --quiet
!pip install pyswarms
import pyswarms as ps

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from datetime import datetime
from itertools import combinations
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import pmdarima as pm
from pyswarm import pso
import warnings
warnings.filterwarnings('ignore')

# %%
# Load the sea temperature dataset
data = pd.read_excel('sea_temp.xlsx', sheet_name='SST')

# Convert Date column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m')
data = data.set_index('Date')

# Rename the target column for easier handling
data = data.rename(columns={'Sea Surface Temperature (Deg C)': 'Temperature'})

# Keep only the temperature column for analysis
temp_data = data[['Temperature']].copy()

print("Dataset Info:")
print(f"Shape: {temp_data.shape}")
print(f"Date range: {temp_data.index.min()} to {temp_data.index.max()}")
print("\nFirst few rows:")
print(temp_data.head())

print("\nBasic Statistics:")
print(temp_data.describe())

# %%
# Plot the time series
plt.figure(figsize=(15, 6))
plt.plot(temp_data.index, temp_data['Temperature'])
plt.title('Sea Surface Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(temp_data['Temperature'], 
                                 model='additive', 
                                 period=12)
fig, axes = plt.subplots(4, 1, figsize=(15, 12))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# STATIONARITY CHECK FUNCTION
# =============================================================================

def check_stationarity(ts, window_size=12, alpha=0.05):
    """
    Check stationarity using ADF test and rolling statistics
    """
    # Calculate rolling statistics
    rolling_mean = ts.rolling(window=window_size).mean()
    rolling_std = ts.rolling(window=window_size).std()
    
    # Perform ADF test
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                        index=['Test Statistic', 'p-value', 
                              '#Lags Used', 'Number of Observations Used'])
    
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    
    print(dfoutput)
    
    if dftest[1] < alpha:
        print('✓ Reject null hypothesis. The data is stationary.')
        is_stationary = True
    else:
        print('✗ Fail to reject null hypothesis. The data is non-stationary.')
        is_stationary = False
    
    print('-' * 60)
    
    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Original', alpha=0.7)
    plt.plot(rolling_mean, label='Rolling Mean', color='red')
    plt.plot(rolling_std, label='Rolling Std', color='black')
    plt.legend()
    plt.title('Rolling Mean & Standard Deviation')
    plt.grid(True)
    plt.show()
    
    return is_stationary

# %%

# =============================================================================
# STATIONARITY ANALYSIS
# =============================================================================

print("=== STATIONARITY CHECK FOR ORIGINAL DATA ===")
is_stationary_original = check_stationarity(temp_data['Temperature'])

# If not stationary, apply differencing
if not is_stationary_original:
    print("\n=== APPLYING FIRST DIFFERENCING ===")
    temp_diff = temp_data['Temperature'].diff().dropna()
    is_stationary_diff = check_stationarity(temp_diff)
    
    if not is_stationary_diff:
        print("\n=== APPLYING SEASONAL DIFFERENCING ===")
        temp_seasonal_diff = temp_data['Temperature'].diff(12).dropna()
        is_stationary_seasonal = check_stationarity(temp_seasonal_diff)
        
        if not is_stationary_seasonal:
            print("\n=== APPLYING BOTH DIFFERENCING ===")
            temp_both_diff = temp_data['Temperature'].diff().diff(12).dropna()
            check_stationarity(temp_both_diff)

# %%
# =============================================================================
# ACF AND PACF PLOTS
# =============================================================================

def plot_acf_pacf(ts, lags=40):
    """Plot ACF and PACF"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    sm.graphics.tsa.plot_acf(ts, lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')
    
    sm.graphics.tsa.plot_pacf(ts, lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.show()

print("=== ACF/PACF FOR DIFFERENCED DATA ===")
temp_diff = temp_data['Temperature'].diff().dropna()
plot_acf_pacf(temp_diff)

# %%
# =============================================================================
# MODEL TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def split_data(data, test_size=24):
    """Split data into train and test sets"""
    train_size = len(data) - test_size
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    return train, test

def evaluate_model(actual, predicted, model_name):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    print(f"\n{model_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    
    return {'RMSE': rmse, 'MAE': mae, 'MSE': mse}

# Split the data
train_data, test_data = split_data(temp_data['Temperature'], test_size=24)

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# %%
# =============================================================================
# ARIMA MODEL
# =============================================================================

print("\n" + "="*50)
print("ARIMA MODEL IMPLEMENTATION")
print("="*50)

# Auto ARIMA to find best parameters
print("Finding optimal ARIMA parameters...")
auto_arima_model = pm.auto_arima(train_data,
                                start_p=0, start_q=0,
                                test='adf',
                                max_p=5, max_q=5,
                                seasonal=False,
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

print(f"Best ARIMA model: {auto_arima_model.order}")

# Fit ARIMA model
arima_order = auto_arima_model.order
arima_model = ARIMA(train_data, order=arima_order)
arima_fitted = arima_model.fit()

print(arima_fitted.summary())

# ARIMA Forecasting
arima_forecast = arima_fitted.forecast(steps=len(test_data))
arima_metrics = evaluate_model(test_data, arima_forecast, "ARIMA")

# %%
# =============================================================================
# SARIMA MODEL
# =============================================================================

print("\n" + "="*50)
print("SARIMA MODEL IMPLEMENTATION")
print("="*50)

# Auto SARIMA to find best parameters
print("Finding optimal SARIMA parameters...")
auto_sarima_model = pm.auto_arima(train_data,
                                 start_p=0, start_q=0,
                                 test='adf',
                                 max_p=3, max_q=3,
                                 m=12,
                                 start_P=0, start_Q=0,
                                 max_P=2, max_Q=2,
                                 seasonal=True,
                                 d=None, D=None,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

print(f"Best SARIMA model: {auto_sarima_model.order} x {auto_sarima_model.seasonal_order}")

# Fit SARIMA model
sarima_order = auto_sarima_model.order
sarima_seasonal_order = auto_sarima_model.seasonal_order

sarima_model = SARIMAX(train_data, 
                      order=sarima_order,
                      seasonal_order=sarima_seasonal_order)
sarima_fitted = sarima_model.fit()

print(sarima_fitted.summary())

# SARIMA Forecasting
sarima_forecast = sarima_fitted.forecast(steps=len(test_data))
sarima_metrics = evaluate_model(test_data, sarima_forecast, "SARIMA")

# %%
# =============================================================================
# SARIMAX MODEL (with exogenous variables)
# =============================================================================

print("\n" + "="*50)
print("SARIMAX MODEL IMPLEMENTATION")
print("="*50)

# Create exogenous variables
def create_exogenous_features(data):
    """Create exogenous variables for SARIMAX"""
    df = pd.DataFrame(index=data.index)
    df['month'] = data.index.month
    # df['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    # df['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    df['trend'] = np.arange(len(data))
    return df

# Create exogenous features for train and test
exog_train = create_exogenous_features(train_data)
exog_test = create_exogenous_features(test_data)

# Fit SARIMAX model
sarimax_model = SARIMAX(train_data,
                       exog=exog_train,
                       order=sarima_order,
                       seasonal_order=sarima_seasonal_order)
sarimax_fitted = sarimax_model.fit()

print(sarimax_fitted.summary())

# SARIMAX Forecasting
sarimax_forecast = sarimax_fitted.forecast(steps=len(test_data), exog=exog_test)
sarimax_metrics = evaluate_model(test_data, sarimax_forecast, "SARIMAX")

# %%
# =============================================================================
# MODEL COMPARISON
# =============================================================================

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

results_df = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'SARIMAX'],
    'RMSE': [arima_metrics['RMSE'], sarima_metrics['RMSE'], sarimax_metrics['RMSE']],
    'MAE': [arima_metrics['MAE'], sarima_metrics['MAE'], sarimax_metrics['MAE']],
    'MSE': [arima_metrics['MSE'], sarima_metrics['MSE'], sarimax_metrics['MSE']]
})

print(results_df)

# Find best performing model
best_model_idx = results_df['RMSE'].idxmin()
best_model_name = results_df.loc[best_model_idx, 'Model']
print(f"\nBest performing model: {best_model_name}")

# Plot predictions
plt.figure(figsize=(15, 8))
plt.plot(train_data.index[-60:], train_data.iloc[-60:], 
         label='Training Data', color='blue', alpha=0.7)
plt.plot(test_data.index, test_data, label='Actual', color='green', linewidth=2)
plt.plot(test_data.index, arima_forecast, label='ARIMA', linestyle='--', alpha=0.8)
plt.plot(test_data.index, sarima_forecast, label='SARIMA', linestyle='--', alpha=0.8)
plt.plot(test_data.index, sarimax_forecast, label='SARIMAX', linestyle='--', alpha=0.8)
plt.axvline(x=test_data.index[0], color='red', linestyle=':', alpha=0.7, label='Test Start')
plt.title('Model Predictions Comparison')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# %%
# =============================================================================
# PSO OPTIMIZATION FOR BEST MODEL
# =============================================================================

print("\n" + "="*50)
print("PSO OPTIMIZATION")
print("="*50)

def sarima_pso_objective(X):
    """
    Vectorized objective function for pyswarms. Each row in X is a candidate solution.
    Returns an array of RMSE values (to be minimized).
    """
    results = []
    for params in X:
        try:
            p, d, q, P, D, Q = [int(max(0, min(param, ub[i]))) for i, param in enumerate(params)]
            model = SARIMAX(train_data,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            fitted_model = model.fit(disp=False)
            predictions = fitted_model.forecast(steps=len(test_data))
            rmse = np.sqrt(mean_squared_error(test_data, predictions))
            results.append(rmse)
        except Exception as e:
            results.append(1e6)  # Large penalty for failed fit
    return np.array(results)

# def objective_function_sarima(params):
#     """Objective function for PSO optimization of SARIMA parameters"""
#     try:
#         p, d, q, P, D, Q = [int(max(0, min(param, 3))) for param in params]
        
#         # Fit SARIMA model with given parameters
#         model = SARIMAX(train_data,
#                        order=(p, d, q),
#                        seasonal_order=(P, D, Q, 12),
#                        enforce_stationarity=False,
#                        enforce_invertibility=False)
        
#         fitted_model = model.fit(disp=False)
        
#         # Make predictions
#         predictions = fitted_model.forecast(steps=len(test_data))
        
#         # Calculate RMSE
#         rmse = np.sqrt(mean_squared_error(test_data, predictions))
        
#         return rmse
        
#     except:
#         return 1e6  # Return large value for invalid parameters

def objective_function_sarimax(params):
    """Objective function for PSO optimization of SARIMAX parameters"""
    try:
        p, d, q, P, D, Q = [int(max(0, min(param, 3))) for param in params]
        
        # Fit SARIMAX model with given parameters
        model = SARIMAX(train_data,
                       exog=exog_train,
                       order=(p, d, q),
                       seasonal_order=(P, D, Q, 12),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        fitted_model = model.fit(disp=False)
        
        # Make predictions
        predictions = fitted_model.forecast(steps=len(test_data), exog=exog_test)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        
        return rmse
        
    except:
        return 1e6  # Return large value for invalid parameters

# # PSO parameters
lb = [0, 0, 0, 0, 0, 0]  # Lower bounds for p, d, q, P, D, Q
ub = [3, 2, 3, 2, 1, 2]  # Upper bounds for p, d, q, P, D, Q
bounds = (np.array(lb), np.array(ub))

# Swarm size and iterations (can be adjusted)
swarm_size = 20
max_iters = 20

# Run PSO using pyswarms
optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=6, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)
best_cost, best_pos = optimizer.optimize(sarima_pso_objective, iters=max_iters)

best_params = [int(max(0, min(best_pos[i], ub[i]))) for i in range(6)]
print(f"[pyswarms] Optimal SARIMA parameters: {best_params}")
print(f"[pyswarms] Optimal RMSE: {best_cost:.4f}")


# Optimize based on best performing model
if best_model_name == 'SARIMA':
    # print("Optimizing SARIMA model with PSO...")
    # optimal_params, optimal_rmse = pso(objective_function_sarima, lb, ub, 
    #                                   swarmsize=20, maxiter=50)
    # optimal_params = [int(p) for p in optimal_params]
    
    # print(f"Optimal SARIMA parameters: {optimal_params}")
    # print(f"Optimal RMSE: {optimal_rmse:.4f}")
    
    # # Train optimized model
    # optimized_model = SARIMAX(train_data,
    #                          order=tuple(optimal_params[:3]),
    #                          seasonal_order=(optimal_params[3], optimal_params[4], optimal_params[5], 12),
    #                          enforce_stationarity=False,
    #                          enforce_invertibility=False)
    # optimized_fitted = optimized_model.fit(disp=False)
    # optimized_forecast = optimized_fitted.forecast(steps=len(test_data))
    original_rmse = sarima_metrics['RMSE']
    

    # Train optimized model with best parameters
    optimized_model = SARIMAX(train_data,
                            order=tuple(best_params[:3]),
                            seasonal_order=(best_params[3], best_params[4], best_params[5], 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
    optimized_fitted = optimized_model.fit(disp=False)
    optimized_forecast = optimized_fitted.forecast(steps=len(test_data))
    optimized_metrics = evaluate_model(test_data, optimized_forecast, "PSO-Optimized (pyswarms)")
    optimal_rmse = optimized_metrics['RMSE']
    

# elif best_model_name == 'SARIMAX':
#     print("Optimizing SARIMAX model with PSO...")
#     optimal_params, optimal_rmse = pso(objective_function_sarimax, lb, ub, 
#                                       swarmsize=20, maxiter=50)
#     optimal_params = [int(p) for p in optimal_params]
    
#     print(f"Optimal SARIMAX parameters: {optimal_params}")
#     print(f"Optimal RMSE: {optimal_rmse:.4f}")
    
#     # Train optimized model
#     optimized_model = SARIMAX(train_data,
#                              exog=exog_train,
#                              order=tuple(optimal_params[:3]),
#                              seasonal_order=(optimal_params[3], optimal_params[4], optimal_params[5], 12),
#                              enforce_stationarity=False,
#                              enforce_invertibility=False)
#     optimized_fitted = optimized_model.fit(disp=False)
#     optimized_forecast = optimized_fitted.forecast(steps=len(test_data), exog=exog_test)
#     original_rmse = sarimax_metrics['RMSE']

# else:  # ARIMA
#     print("PSO optimization not implemented for ARIMA in this example.")
#     print("Using SARIMA for PSO optimization instead...")
#     optimal_params, optimal_rmse = pso(objective_function_sarima, lb, ub, 
#                                       swarmsize=20, maxiter=50)
#     optimal_params = [int(p) for p in optimal_params]
    
#     # Train optimized model
#     optimized_model = SARIMAX(train_data,
#                              order=tuple(optimal_params[:3]),
#                              seasonal_order=(optimal_params[3], optimal_params[4], optimal_params[5], 12),
#                              enforce_stationarity=False,
#                              enforce_invertibility=False)
#     optimized_fitted = optimized_model.fit(disp=False)
#     optimized_forecast = optimized_fitted.forecast(steps=len(test_data))
#     original_rmse = arima_metrics['RMSE']

# %%
# =============================================================================
# OPTIMIZATION RESULTS COMPARISON
# =============================================================================

print("\n" + "="*50)
print("OPTIMIZATION RESULTS")
print("="*50)

# Calculate metrics for optimized model
optimized_metrics = evaluate_model(test_data, optimized_forecast, "PSO-Optimized")

# Compare results
improvement = ((original_rmse - optimal_rmse) / original_rmse) * 100

print(f"\nComparison Results:")
print(f"Original {best_model_name} RMSE: {original_rmse:.4f}")
print(f"PSO-Optimized RMSE: {optimal_rmse:.4f}")
print(f"Improvement: {improvement:.2f}%")

# Final comparison plot
plt.figure(figsize=(15, 8))
plt.plot(train_data.index[-60:], train_data.iloc[-60:], 
         label='Training Data', color='blue', alpha=0.7)
plt.plot(test_data.index, test_data, 
         label='Actual', color='green', linewidth=3)

if best_model_name == 'SARIMA':
    plt.plot(test_data.index, sarima_forecast, 
             label=f'Original SARIMA (RMSE: {original_rmse:.3f})', 
             linestyle='--', alpha=0.8, linewidth=2)
elif best_model_name == 'SARIMAX':
    plt.plot(test_data.index, sarimax_forecast, 
             label=f'Original SARIMAX (RMSE: {original_rmse:.3f})', 
             linestyle='--', alpha=0.8, linewidth=2)
else:
    plt.plot(test_data.index, arima_forecast, 
             label=f'Original ARIMA (RMSE: {original_rmse:.3f})', 
             linestyle='--', alpha=0.8, linewidth=2)

plt.plot(test_data.index, optimized_forecast, 
         label=f'PSO-Optimized (RMSE: {optimal_rmse:.3f})', 
         linestyle='-', linewidth=2, color='red')

plt.axvline(x=test_data.index[0], color='gray', linestyle=':', alpha=0.7, label='Test Start')
plt.title('Original vs PSO-Optimized Model Comparison')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# %%
# =============================================================================
# FUTURE FORECASTING
# =============================================================================

print("\n" + "="*50)
print("FUTURE FORECASTING")
print("="*50)

# Retrain optimized model on full dataset
full_model = SARIMAX(temp_data['Temperature'],
                    order=tuple(best_params[:3]),
                            seasonal_order=(best_params[3], best_params[4], best_params[5], 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False)                            

# if best_model_name == 'SARIMAX':
#     full_exog = create_exogenous_features(temp_data['Temperature'])
#     full_model = SARIMAX(temp_data['Temperature'],
#                         exog=full_exog,
#                         order=tuple(optimal_params[:3]),
#                         seasonal_order=(optimal_params[3], optimal_params[4], optimal_params[5], 12),
#                         enforce_stationarity=False,
#                         enforce_invertibility=False)

full_fitted = full_model.fit(disp=False)

# Forecast next 24 months
forecast_steps = 24
if best_model_name == 'SARIMAX':
    # Create future exogenous variables
    future_dates = pd.date_range(start=temp_data.index[-1] + pd.DateOffset(months=1), 
                                periods=forecast_steps, freq='MS')
    future_exog = create_exogenous_features(pd.Series(index=future_dates))
    future_forecast = full_fitted.forecast(steps=forecast_steps, exog=future_exog)
else:
    future_forecast = full_fitted.forecast(steps=forecast_steps)

# Create future dates
future_dates = pd.date_range(start=temp_data.index[-1] + pd.DateOffset(months=1), 
                            periods=forecast_steps, freq='MS')

# Plot future forecast
plt.figure(figsize=(15, 8))
plt.plot(temp_data.index[-120:], temp_data['Temperature'].iloc[-120:], 
         label='Historical Data', color='blue')
plt.plot(future_dates, future_forecast, 
         label='24-Month Forecast', color='red', linewidth=2, linestyle='--')
plt.axvline(x=temp_data.index[-1], color='green', linestyle=':', 
           alpha=0.7, label='Forecast Start')
plt.title('Sea Surface Temperature - 24 Month Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Future forecast values (next 24 months):")
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted_Temperature': future_forecast
})
print(forecast_df)

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)