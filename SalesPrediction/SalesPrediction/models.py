import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import holidays
import warnings

# Suppress harmless warnings from the models
warnings.filterwarnings("ignore", category=UserWarning)

def get_safe_plot_data():
    """Returns a default empty structure for plot data to prevent errors."""
    return {
        'historical_dates': [], 'historical_values': [],
        'forecast_dates': [], 'forecast_values': [],
        'lower_ci': [], 'upper_ci': []
    }

def create_features(df, is_sparse=False):
    """Creates a rich set of time series features from a datetime index."""
    df = df.set_index('date')
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekofyear'] = df.index.isocalendar().week.astype(int)

    malaysia_holidays = holidays.MY()
    df['is_holiday'] = df.index.to_series().apply(lambda date: date in malaysia_holidays).astype(int)

    if is_sparse:
        # --- Features for Intermittent Data ---
        df['is_zero_sale'] = (df['sales'] == 0).astype(int)
        df['days_since_sale'] = df['is_zero_sale'].cumsum()
        df['days_since_sale'] = df['days_since_sale'] - df['days_since_sale'][df['is_zero_sale'] == 0].groupby(df['is_zero_sale'].cumsum()).transform('max')
    else:
        # --- Features for Flowing Data ---
        for window in [2, 4, 8]: # Added longer window
            df[f'rolling_mean_{window}'] = df['sales'].shift(1).rolling(window=window).mean()
        for lag in [1, 2, 4]:
            df[f'lag_{lag}'] = df['sales'].shift(lag)

    df.fillna(0, inplace=True)
    return df

def run_xgboost_prediction(df, forecast_horizon=30, time_freq='W'):
    try:
        if len(df) < 20:
            raise ValueError("Not enough data for XGBoost (requires at least 20 data points).")

        sparsity = (df['sales'] == 0).sum() / len(df)
        is_sparse = sparsity > 0.4

        objective = 'count:poisson' if is_sparse else 'reg:squarederror'
        print(f"--- Running XGBoost with {'Poisson (Sparse)' if is_sparse else 'Standard (Flowing)'} objective ---")

        df_featured = create_features(df.copy(), is_sparse=is_sparse)

        test_size = max(forecast_horizon, int(len(df_featured) * 0.25))
        train = df_featured.iloc[:-test_size].copy()
        test = df_featured.iloc[-test_size:].copy()

        features = [col for col in df_featured.columns if col != 'sales']
        target = 'sales'

        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]

        reg = xgb.XGBRegressor(objective=objective, n_estimators=1000, 
                               max_depth=5, learning_rate=0.01, subsample=0.8, 
                               colsample_bytree=0.8, random_state=42,
                               early_stopping_rounds=50)

        reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        last_date = df_featured.index.max()
        future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=time_freq)[1:]

        future_df = pd.DataFrame(index=future_dates)
        future_df['sales'] = np.nan

        full_df = pd.concat([df_featured, future_df])
        full_featured_df = create_features(full_df.reset_index(), is_sparse=is_sparse)

        future_features = full_featured_df.set_index('date').loc[future_dates]

        predictions = reg.predict(future_features[features])
        predictions[predictions < 0] = 0

        residuals = y_test - reg.predict(X_test)
        std_dev = np.std(residuals)
        lower_bound = predictions - 1.96 * std_dev
        upper_bound = predictions + 1.96 * std_dev
        lower_bound[lower_bound < 0] = 0

        plot_data = {
            'historical_dates': df_featured.index.strftime('%Y-%m-%d').tolist(),
            'historical_values': df_featured['sales'].tolist(),
            'forecast_dates': future_dates.strftime('%Y-%m-%d').tolist(),
            'forecast_values': predictions.tolist(),
            'lower_ci': lower_bound.tolist(), 
            'upper_ci': upper_bound.tolist()
        }

        test_predictions = reg.predict(X_test)
        mae = mean_absolute_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)

        return {'mae': mae, 'r2': r2, 'plot_data': plot_data}
    except Exception as e:
        print(f"XGBoost Error: {e}")
        return {'mae': float('inf'), 'r2': float('-inf'), 'plot_data': get_safe_plot_data()}

def run_arima_prediction(df, forecast_horizon=30, time_freq='W'):
    try:
        if len(df) < 15:
            raise ValueError("Not enough data for ARIMA (requires at least 15 data points).")

        df_series = df.set_index('date')['sales'].asfreq(time_freq, fill_value=0)

        # --- UPGRADE: Stabilize log transform ---
        non_zero_median = df_series[df_series > 0].median()
        if pd.isna(non_zero_median) or non_zero_median == 0: non_zero_median = 0.1
        df_series_stabilized = df_series.replace(0, non_zero_median * 0.1)
        log_series = np.log1p(df_series_stabilized)

        seasonal_period = 52 if time_freq == 'W' else 12
        best_aic, best_order, best_seasonal_order = float("inf"), None, None

        orders_to_try = [((1,1,1), (0,0,0,0)), ((1,1,0), (0,0,0,0))]
        if len(log_series) > seasonal_period:
            orders_to_try.append(((1,1,1), (1,1,1,seasonal_period)))

        for order, seasonal_order in orders_to_try:
            try:
                model = SARIMAX(log_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False).fit(disp=False)
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
            except Exception:
                continue

        if not best_order: best_order, best_seasonal_order = (1,1,0), (0,0,0,0)

        model_fit = SARIMAX(log_series, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False).fit(disp=False)

        forecast_result = model_fit.get_forecast(steps=forecast_horizon)
        forecast = np.expm1(forecast_result.predicted_mean)
        conf_int = np.expm1(forecast_result.conf_int())

        forecast[forecast < 0] = 0
        conf_int[conf_int < 0] = 0

        plot_data = {
            'historical_dates': df_series.index.strftime('%Y-%m-%d').tolist(),
            'historical_values': df_series.tolist(),
            'forecast_dates': forecast.index.strftime('%Y-%m-%d').tolist(),
            'forecast_values': forecast.values.tolist(),
            'lower_ci': conf_int.iloc[:, 0].values.tolist(),
            'upper_ci': conf_int.iloc[:, 1].values.tolist()
        }

        train_size = int(len(df_series) * 0.8)
        train, test = df_series.iloc[0:train_size], df_series.iloc[train_size:len(df_series)]

        can_evaluate = len(train) > seasonal_period if best_seasonal_order[3] > 0 else len(train) > 10

        if can_evaluate and len(test) > 0:
            eval_log_model = SARIMAX(np.log1p(train.replace(0, non_zero_median * 0.1)), order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False).fit(disp=False)
            eval_log_forecast = eval_log_model.forecast(steps=len(test))
            eval_forecast = np.expm1(eval_log_forecast)

            mae = mean_absolute_error(test, eval_forecast)
            r2 = r2_score(test, eval_forecast)
        else:
            mae, r2 = float('inf'), float('-inf')

        return {'mae': mae, 'r2': r2, 'plot_data': plot_data}
    except Exception as e:
        print(f"ARIMA/SARIMAX Error: {e}")
        return {'mae': float('inf'), 'r2': float('-inf'), 'plot_data': get_safe_plot_data()}

