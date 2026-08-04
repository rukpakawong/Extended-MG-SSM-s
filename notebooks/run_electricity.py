# Setup
import torch.nn as nn
import torch.optim as optim

from utils.data_reader import load_and_prepare_time_series_data
from utils.trainer import Trainer
from models.baseline_models import LSTMModel, BiLSTMModel, GRUModel
from models.custom_models import MGSSMModel, MGSSMsModel, ExtendedMGSSMsModel

train_loader, val_loader, test_loader, scaler = load_and_prepare_time_series_data(
    filepath_or_url = "/home/theppawan/nn-models/data/thailand_power_forecasting_dataset.csv",
    target_column=['metropolitan_demand'],
    date_column="time",
    seq_length=168,
    batch_size=64,
    train_split=0.8,
    fill_missing=True)

# Setup model configurations
model_config_dict = {
    "Baseline LSTM": LSTMModel(input_size=1, hidden_size=256, num_layers=1, output_size=1),
    "Baseline BiLSTM": BiLSTMModel(input_size=1, hidden_size=256, num_layers=1, output_size=1),
    "Baseline GRU": GRUModel(input_size=1, hidden_size=128, num_layers=1, output_size=1),
    "MGSSM": MGSSMModel(input_size=1, hidden_size=64, num_layers=1, output_size=1, gate_size=32),
    "MGSSMs": MGSSMsModel(input_size=1, hidden_size=64, num_layers=1, output_size=1, gate_size=32),
    "ExtendedMGSSMs": ExtendedMGSSMsModel(input_size=1, hidden_size=64, num_layers=1, output_size=1, gate_size=32, p=2)
}

model_trained = []
for model in model_config_dict.values():
    print(f"Training model: {model.__class__.__name__}")
    trainer = Trainer(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        device="cpu"
    )
    trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=1000, patience=50, label="elec")