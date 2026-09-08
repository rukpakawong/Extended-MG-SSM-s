import optuna
import torch.nn as nn
import torch.optim as optim
import json

from utils.data_reader import load_and_prepare_time_series_data
from utils.trainer import Trainer
from models.baseline_models import LSTMModel, BiLSTMModel, GRUModel
from models.custom_models import MGSSMsModel, ExtendedMGSSMsModel

def objective(trial):

    train_loader, val_loader, _, _ = load_and_prepare_time_series_data(
        filepath_or_url = "/home/theppawan/nn-models/data/bangkok_solar_data.csv",
        target_column=['GHI'],
        date_column='datetime',
        seq_length=24,
        batch_size=128,
        train_split=0.8,
        fill_missing=True
    )

    # 2. Treat the model architecture as a categorical hyperparameter
    model_type = trial.suggest_categorical("model_type", ["LSTM", "BiLSTM", "GRU", "MGSSMs", "ExtendedMGSSMs"])

    if model_type == "LSTM":
        hidden_size = trial.suggest_categorical("lstm_hidden_size", [16, 32, 64, 128, 256])
        num_layers = trial.suggest_categorical("lstm_num_layers", [1, 2, 3, 4, 5])
        model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)

    elif model_type == "GRU":
        hidden_size = trial.suggest_categorical("gru_hidden_size", [16, 32, 64, 128, 256])
        num_layers = trial.suggest_categorical("gru_num_layers", [1, 2, 3, 4, 5])
        model = GRUModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
        
    elif model_type == "MGSSMs":
        hidden_size = trial.suggest_categorical("ssm_state_dim", [16, 32, 64, 128, 256])
        num_layers = trial.suggest_categorical("ssm_num_layers", [1, 2, 3, 4, 5])
        gate_size = trial.suggest_categorical("ssm_gate_size", [16, 32, 64])
        model = MGSSMsModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1, gate_size=gate_size)

    elif model_type == "ExtendedMGSSMs":
        hidden_size = trial.suggest_categorical("extend_ssm_state_dim", [16, 32, 64, 128, 256])
        num_layers = trial.suggest_categorical("extend_ssm_num_layers", [1, 2, 3, 4, 5])
        gate_size = trial.suggest_categorical("extend_ssm_gate_size", [16, 32, 64])
        model = ExtendedMGSSMsModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1, gate_size=gate_size, p=2)

    # 3. Initialize your custom Trainer class
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion,
        save_dir=None
    )

    # 4. Train and evaluate
    _, val_losses = trainer.train(
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=100, 
        patience=10,
        save_best_model=False
    )

    return min(val_losses)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="multi_model_benchmark")
    study.optimize(objective, n_trials=50)
    
    # 1. Convert all trial data into a Pandas DataFrame
    df = study.trials_dataframe()
    
    # 2. Filter for only successfully completed trials (ignores pruned/failed trials)
    df = df[df['state'] == 'COMPLETE']
    
    # 3. Group by the model type and find the index of the minimum loss ('value') for each
    # Optuna automatically prepends 'params_' to hyperparameter column names
    best_trials_idx = df.groupby('params_model_type')['value'].idxmin()
    best_trials_per_model = df.loc[best_trials_idx]
    
    # Define output file path
    output_filename = "/home/theppawan/nn-models/results/study_results_solar.txt"
    
    # Open text file in write mode
    with open(output_filename, "w") as f:
        f.write("\n" + "="*40 + "\n")
        f.write("🏆 BEST HYPERPARAMETERS PER MODEL 🏆\n")
        f.write("="*40 + "\n")
        
        # 4. Loop through and write the optimal settings for each model
        for _, row in best_trials_per_model.iterrows():
            model_name = row['params_model_type']
            best_loss = row['value']
            trial_num = row['number']
            
            f.write(f"\nModel: {model_name} (Found in Trial {trial_num})\n")
            f.write(f"Lowest Validation Loss: {best_loss:.4f}\n")
            
            # Extract columns starting with 'params_', drop NaNs, and convert to dictionary
            raw_params = row.filter(like='params_').dropna().to_dict()
            
            # Clean up the keys for readability by removing the 'params_' prefix
            optimal_params = {k.replace('params_', ''): v for k, v in raw_params.items()}
            
            for param_name, param_value in optimal_params.items():
                f.write(f"  - {param_name}: {param_value}\n")
                
    print(f"Study results successfully saved to {output_filename}")