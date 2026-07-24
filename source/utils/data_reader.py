import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        """
        Takes a continuous sequence of data and creates sliding windows.
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        # Subtract squence_length becaouse we need that many time steps
        # to form a single input window before we can predict the next step
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):

        # The input features: a window of past data
        x = self.data[idx: idx + self.sequence_length]

        # The target label: the single value occurring immediately after the window
        y = self.data[idx + self.sequence_length]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_and_prepare_time_series_data(filepath_or_url,
                                      target_column,
                                      date_column=None,
                                      seq_length=14,
                                      batch_size=32,
                                      train_split=0.8,
                                      fill_missing=True):
    """
    Downloads or read a CSV dataset, clean it, scales it, and return DataLoaders ready for training

    Arg:
        filepath_or_url (str): Path to local CSV or URL to a remote CSV.
        target_column (list of str): The name of the column you want to predict.
        date_column (str, optional): Column to sort chronologically
        seq_length (int, default=14): How many past steps to use for predicting the next step.
        batch_size (int, default=32): Size of the batches for the DataLoaders.
        train_split (float, default=0.8): Ratio of data to use for training (0 to 1)
        fill_missing (bool): If True, forward-fills and zero-fills missing data.
    """
    print(f"Fetching data from: {filepath_or_url}")

    try:
        df = pd.read_csv(filepath_or_url)
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {filepath_or_url}. Error: {e}")
    
    if date_column in df.columns:
        print(f"Sorting data chronologicall by column: {date_column}")
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
    
    # Extract the target column
    if fill_missing:
        # Forward fill carries the last known value forward, then fillna(0) catches any starting NaNs
        data = df[target_column].ffill().fillna(0).values
    else:
        # Drop rows with NaN values in the target column
        df = df.dropna(subset=target_column)
        data = df[target_column].values 
    
    print(f"Succesfully loaded {len(data)} sequential data points.")

    # Models require data scaled between 0 a nd 1 for stable gradient descent
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split into training and validation sets chronologically
    train_size = int(train_split * len(scaled_data))
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:]

    train_dataset = TimeSeriesDataset(train_data, seq_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length)

    # Shuffle traning data to prevent memorization of the timeline
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Prepared Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    return train_loader, val_loader, val_dataset, scaler