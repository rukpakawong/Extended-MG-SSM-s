import pandas as pd
import numpy as np
import requests
import holidays

# 1. Fetch Thai Power System Hourly Dataset from Zenodo
def fetch_zenodo_electricity_data():
    """
    Downloads Zenodo dataset ID 17109911:
    Thai Power System: Hourly Power Generation, Demand, and Cross-Border Flows
    """
    urls = [
        "https://zenodo.org/records/17109911/files/system_2023.csv",
        "https://zenodo.org/records/17109911/files/system_2024.csv"
    ]
    dfs = []
    for url in urls:
        print(f"Downloading electricity load data from {url}...")
        df = pd.read_csv(url)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Standardize time column with dayfirst=True
    time_col = [col for col in combined_df.columns if col in ['timestamp', 'datetime', 'time', 'Date']][0]
    combined_df['time'] = pd.to_datetime(combined_df[time_col], dayfirst=True)
    
    return combined_df

# 2. Fetch Bangkok Weather Features from Open-Meteo
def fetch_bangkok_weather(start_date="2023-01-01", end_date="2024-12-31"):
    """
    Fetches hourly historical temperature, humidity, and heat index for Bangkok (13.7563 N, 100.5018 E).
    """
    print("Fetching Bangkok weather features from Open-Meteo API...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 13.7563,
        "longitude": 100.5018,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature"],
        "timezone": "Asia/Bangkok"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    hourly_data = response.json()["hourly"]
    
    weather_df = pd.DataFrame({
        "time": pd.to_datetime(hourly_data["time"]),
        "temp_c": hourly_data["temperature_2m"],
        "humidity_pct": hourly_data["relative_humidity_2m"],
        "apparent_temp_c": hourly_data["apparent_temperature"]
    })
    return weather_df

# 3. Add Temporal Encodings & Thai Public Holidays
def add_calendar_features(df):
    """
    Generates sine/cosine cyclical time features and Thai official public holiday flags.
    """
    # Hour cyclical encoding (24-hour cycle)
    df['hour'] = df['time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of year cyclical encoding (365.25-day cycle)
    df['day_of_year'] = df['time'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Day of week / Weekend indicators
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Thai Official Holidays
    th_holidays = holidays.Thailand(years=[2023, 2024])
    df['is_thai_holiday'] = df['time'].dt.date.isin(th_holidays).astype(int)
    
    return df

# Run Execution Pipeline
if __name__ == "__main__":
    # Fetch datasets
    load_df = fetch_zenodo_electricity_data()
    weather_df = fetch_bangkok_weather(start_date="2023-01-01", end_date="2024-12-31")
    
    # Merge on timestamp
    merged_df = pd.merge(load_df, weather_df, on="time", how="inner")
    merged_df = add_calendar_features(merged_df)
    
    # Sort chronologically and reset index
    merged_df = merged_df.sort_values("time").reset_index(drop=True)
    
    # Save combined dataset for deep learning
    output_filename = "thailand_power_forecasting_dataset.csv"
    merged_df.to_csv(output_filename, index=False)
    
    print(f"\nPipeline Complete! Output saved to '{output_filename}'.")
    print(f"Dataset Shape: {merged_df.shape}")
    print("Features included:")
    for col in merged_df.columns:
        print(f" - {col}")