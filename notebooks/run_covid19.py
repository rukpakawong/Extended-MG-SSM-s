# Setup
import torch.nn as nn
import torch.optim as optim
import json

from utils.data_reader import load_and_prepare_time_series_data
from utils.trainer import Trainer
from models.baseline_models import LSTMModel, BiLSTMModel, GRUModel
from models.custom_models import MGSSMModel, MGSSMsModel, ExtendedMGSSMsModel
selected_country_codes = ['US', 'IN', 'BR', 'FR', 'DE',
                         'GB', 'RU', 'IT', 'TR', 'ES',
                         'VN', 'AR', 'AU', 'AT', 'BD',
                         'BE', 'BG', 'CA', 'CL', 'CN',
                         'CU', 'DK', 'FI', 'GE', 'GR',
                         'ID', 'JP', 'JO', 'KE', 'KR',
                         'LR', 'MY','ML', 'MX', 'NL',
                         'NO', 'PH','SE', 'CH', 'TH']

# Fetch and prepare the data for each country code
with open("/home/theppawan/nn-models/data/COVID19_url_data.json", "r") as f:
    dataset = json.load(f)

train_loader_dist = {}
val_loader_dist = {}
test_loader_dist = {}
scaler_dist = {}
for key in selected_country_codes:
    train_loader, val_loader, test_loader, scaler = load_and_prepare_time_series_data(
        filepath_or_url = dataset['region'].format(region=key),
        target_column=['cumulative_confirmed'],
        date_column="date",
        seq_length=14,
        batch_size=64,
        train_split=0.8,
        fill_missing=True)
    train_loader_dist[key] = train_loader
    val_loader_dist[key] = val_loader
    test_loader_dist[key] = test_loader
    scaler_dist[key] = scaler

# Setup model configurations
model_config_dict = {
    "Baseline LSTM": LSTMModel(input_size=1, hidden_size=256, num_layers=1, output_size=1),
    "Baseline BiLSTM": BiLSTMModel(input_size=1, hidden_size=256, num_layers=1, output_size=1),
    "Baseline GRU": GRUModel(input_size=1, hidden_size=128, num_layers=1, output_size=1),
    "MGSSM": MGSSMModel(input_size=1, hidden_size=64, num_layers=1, output_size=1, gate_size=32),
    "MGSSMs": MGSSMsModel(input_size=1, hidden_size=64, num_layers=1, output_size=1, gate_size=32),
    "ExtendedMGSSMs": ExtendedMGSSMsModel(input_size=1, hidden_size=64, num_layers=1, output_size=1, gate_size=32, p=2)
}

# Training parameters
model_trained = []
for model in model_config_dict.values():
    print(f"Training model: {model.__class__.__name__}")
    trainer = Trainer(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        device="cpu"
    )
    for key in selected_country_codes:
        train_loader = train_loader_dist[key]
        val_loader = val_loader_dist[key]
        print(f"Training on country code: {key}")

        trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=1000, patience=50, label=key)