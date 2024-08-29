import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import zscore
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import base64



# Function to preprocess the data
def preprocess_time_series(data, date_col, target_col):
    # Drop missing values
    data = data.dropna()

    # Convert the date column to datetime
    data[date_col] = pd.to_datetime(data[date_col])

    # Normalize and scale the target column
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[target_col] = scaler.fit_transform(data[[target_col]])

    # Handle outliers using Z-score
    data['z_score'] = zscore(data[target_col])
    data = data[(data['z_score'].abs() < 3)].drop(columns=['z_score'])

    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != date_col:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    # Retain only the Date and target columns
    data = data[[date_col, target_col]]

    return data, scaler

# Function to load and prepare data
def load_and_prepare_data(file, target_col):
    df = pd.read_csv(file)
    df, scaler = preprocess_time_series(df, 'Date', target_col)
    df.set_index('Date', inplace=True)
    return df, scaler

# Function to prepare data for LSTM and GRU
def prepare_lstm_data(df, n_steps):
    scaled_data = df.values
    
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, 0])  # Features
        y.append(scaled_data[i, 0])  # Target
    
    return np.array(X), np.array(y), scaled_data

# Function to build and train LSTM model
def build_and_train_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0)
    return model

# Function to build and train GRU model
def build_and_train_gru(X_train, y_train):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0)
    return model

# Function to evaluate models
def evaluate_models(df, target_col, future_periods, scaler):
    n_steps = 5
    results = {}
    
    # LSTM Model
    X, y, scaled_data = prepare_lstm_data(df, n_steps)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    lstm_model = build_and_train_lstm(X_train, y_train)
    forecast = lstm_model.predict(X_test)
    forecast = scaler.inverse_transform(forecast)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, forecast)
    r2 = r2_score(actual, forecast)
    
    results['LSTM'] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

    # GRU Model
    gru_model = build_and_train_gru(X_train, y_train)
    forecast_gru = gru_model.predict(X_test)
    forecast_gru = scaler.inverse_transform(forecast_gru)
    actual_gru = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse_gru = mean_squared_error(actual_gru, forecast_gru)
    rmse_gru = np.sqrt(mse_gru)
    mae_gru = mean_absolute_error(actual_gru, forecast_gru)
    r2_gru = r2_score(actual_gru, forecast_gru)
    
    results['GRU'] = {'RMSE': rmse_gru, 'MAE': mae_gru, 'R2': r2_gru}

    # XGBoost Model
    X_train_xgb = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test_xgb = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train_xgb, y_train)
    forecast_xgb = xgb_model.predict(X_test_xgb)
    forecast_xgb = scaler.inverse_transform(forecast_xgb.reshape(-1, 1))
    actual_xgb = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse_xgb = mean_squared_error(actual_xgb, forecast_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    mae_xgb = mean_absolute_error(actual_xgb, forecast_xgb)
    r2_xgb = r2_score(actual_xgb, forecast_xgb)
    
    results['XGBoost'] = {'RMSE': rmse_xgb, 'MAE': mae_xgb, 'R2': r2_xgb}

    # ARIMA Model
    arima_model = ARIMA(df[target_col], order=(5,1,0))
    arima_result = arima_model.fit()
    forecast_arima = arima_result.forecast(steps=len(df) - split)
    forecast_arima = scaler.inverse_transform(forecast_arima.values.reshape(-1, 1))
    actual_arima = scaler.inverse_transform(df[target_col].values[split:].reshape(-1, 1))

    mse_arima = mean_squared_error(actual_arima, forecast_arima)
    rmse_arima = np.sqrt(mse_arima)
    mae_arima = mean_absolute_error(actual_arima, forecast_arima)
    r2_arima = r2_score(actual_arima, forecast_arima)
    
    results['ARIMA'] = {'RMSE': rmse_arima, 'MAE': mae_arima, 'R2': r2_arima}

    # SARIMA Model
    sarima_model = SARIMAX(df[target_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
    forecast_sarima = sarima_model.get_forecast(steps=len(df) - split).predicted_mean
    forecast_sarima = scaler.inverse_transform(forecast_sarima.values.reshape(-1, 1))
    actual_sarima = scaler.inverse_transform(df[target_col].values[split:].reshape(-1, 1))

    mse_sarima = mean_squared_error(actual_sarima, forecast_sarima)
    rmse_sarima = np.sqrt(mse_sarima)
    mae_sarima = mean_absolute_error(actual_sarima, forecast_sarima)
    r2_sarima = r2_score(actual_sarima, forecast_sarima)
    
    results['SARIMA'] = {'RMSE': rmse_sarima, 'MAE': mae_sarima, 'R2': r2_sarima}

    # Select the best model based on RMSE and MAE
    best_model = min(results, key=lambda k: (results[k]['RMSE'], results[k]['MAE']))

    return results, best_model

# Function to train the best model on the entire dataset
def train_best_model(best_model, df, target_col, scaler, future_periods):
    n_steps = 5

    if best_model in ['LSTM', 'GRU']:
        X, y, scaled_data = prepare_lstm_data(df, n_steps)
        X_train = np.reshape(X, (X.shape[0], X.shape[1], 1))

        if best_model == 'LSTM':
            model = build_and_train_lstm(X_train, y)
        elif best_model == 'GRU':
            model = build_and_train_gru(X_train, y)

        forecast = model.predict(X_train)
        forecast = scaler.inverse_transform(forecast)

    elif best_model == 'XGBoost':
        X, y, scaled_data = prepare_lstm_data(df, n_steps)
        X_train = np.reshape(X, (X.shape[0], X.shape[1]))

        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y)
        forecast = model.predict(X_train)
        forecast = scaler.inverse_transform(forecast.reshape(-1, 1))

    elif best_model == 'ARIMA':
        model = ARIMA(df[target_col], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=future_periods)
        forecast = scaler.inverse_transform(forecast.values.reshape(-1, 1))

    elif best_model == 'SARIMA':
        model = SARIMAX(df[target_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
        forecast = model.get_forecast(steps=future_periods).predicted_mean
        forecast = scaler.inverse_transform(forecast.values.reshape(-1, 1))

    return forecast


# Function to set background image and style the Streamlit app
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
        }}
        .css-18e3th9 {{
            color: #003366; /* Dark blue color for headings */
            font-size: 1.5em; /* Increase font size */
        }}
        .css-1wvnftk {{
            color: #003366; /* Dark blue color for subheadings */
            font-size: 1.2em; /* Increase font size */
        }}
        .css-1v0mbdj {{
            color: #003366; /* Dark blue color for input labels */
            font-size: 1.2em; /* Increase font size */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Updated Streamlit app function
def streamlit_app():
    set_background(r"C:\Users\Ramtej\Desktop\Sales_Forcasting\bg.jpg")  # Use raw string for the path

    st.title("Sales Forecasting Application")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        target_col = st.text_input("Enter the target column name:")
        future_periods = st.number_input("Enter the number of future periods to forecast:", min_value=1, max_value=365, value=30)

        if st.button("Forecast"):
            df, scaler = load_and_prepare_data(uploaded_file, target_col)
            results, best_model = evaluate_models(df, target_col, future_periods, scaler)

            # Convert the results dictionary to a DataFrame for tabular display
            results_df = pd.DataFrame(results).T  # Transpose for better readability
            st.write("Model Evaluation Results:")
            st.dataframe(results_df)  # Display as a table

            st.write(f"Best Model: {best_model}")
            forecast = train_best_model(best_model, df, target_col, scaler, future_periods)

            # Check if forecast length matches future periods
            forecast = forecast.flatten()  # Flatten the forecast to a 1D array
            if len(forecast) != future_periods:
                st.error(f"Forecast length ({len(forecast)}) does not match the number of future periods ({future_periods}).")
                return

            st.write(f"Forecast for the next {future_periods} periods:")
            st.write(forecast)

            # Visualize original and forecasted data
            fig, ax = plt.subplots()
            df[target_col].plot(ax=ax, label='Original Data', color='blue')
            future_dates = pd.date_range(start=df.index[-1], periods=future_periods + 1, freq='D')[1:]

            # Ensure forecast length matches future_dates length
            if len(forecast) > len(future_dates):
                forecast = forecast[:len(future_dates)]  # Trim forecast if too long
            elif len(future_dates) > len(forecast):
                future_dates = future_dates[:len(forecast)]  # Trim future_dates if too long

            forecast_series = pd.Series(forecast, index=future_dates)
            forecast_series.plot(ax=ax, label='Forecasted Data', color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales')
            ax.legend()
            st.pyplot(fig)

if __name__ == '__main__':
    streamlit_app()
