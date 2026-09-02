import io
import requests
import pandas as pd


# 1. FETCH DATA FROM NREL
def fetch_nsrdb_data(api_key, email, lat, lon, year="2020", interval="30", output_file="himawari_data.csv"):
    """Queries the NREL Himawari API and returns a pandas DataFrame."""
    url = "https://developer.nlr.gov/api/nsrdb/v2/solar/himawari-download.csv"
    
    payload = {
        'api_key': api_key,
        'email': email,
        'wkt': f'POINT({lon} {lat})', # The required format for Himawari coverage
        'names': year,
        'interval': interval,
        'leap_day': 'false',
        'utc': 'false',
        'attributes': 'ghi,dni,dhi,air_temperature,wind_speed,relative_humidity'
    }
    
    print(f"Requesting NSRDB data for Lat {lat}, Lon {lon}, Year: {year}...")
    response = requests.get(url, params=payload)
    if response.status_code == 200:
        # 1. Save the raw text to a CSV file directly
        with open(output_file, 'w') as file:
            file.write(response.text)
        print(f"Success! Data saved to {output_file}")
        
        # 2. Load into Pandas to verify the data
        # NSRDB CSVs always have 2 header rows of metadata before the tabular data
        df = pd.read_csv(io.StringIO(response.text), skiprows=2)
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])
        df = df.set_index('datetime')
        df.to_csv(output_file)
        print("\nData Preview:")
        print(df.head())
    
        return df
    else:
        print(f"API Error {response.status_code}: {response.text}")
        return None

# 4. EXECUTE PIPELINE
if __name__ == "__main__":
    # Replace with your actual credentials
    API_KEY = "IbPo8vswLtqpoczi9Ag3FRazKnWM46qKQX7hiYso" 
    EMAIL = "theppawan.k@gmail.com" 
    LAT, LON = 13.754, 100.501 # Example: NREL Golden, CO
    
    # 4a. Fetch & Preprocess
    raw_df = fetch_nsrdb_data(API_KEY, EMAIL, LAT, LON, output_file="bangkok_solar_data.csv")