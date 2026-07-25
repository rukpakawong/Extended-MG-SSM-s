import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

class Evaluator:
    def __init__(self, device=None):
        """
        Initialize the Evaluator.

        Args:
            device: 'cuda', 'mps', or 'cpu'. If None, it auto-detects.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

    @torch.no_grad()
    def get_predictions(self, model, data_loader, scaler=None):
        """
        Get predictions from the model for the entire dataset in data_loader.

        Args:
            model: The trained model.
            data_loader: DataLoader containing the dataset to evaluate.
            scaler: Optional scaler used for inverse transforming predictions and targets.

        Returns:
            Tuple of (targets, predictions) as numpy arrays.
        """
        model = model.to(self.device)
        model.eval()  # Set model to evaluation mode

        all_predictions = []
        all_targets = []

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = model(inputs)

            # Move to CPU and convert to numpy
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # If a scaler is provided, inverse transform the predictions and targets
        if scaler is not None:
            all_predictions = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
            all_targets = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()

        return all_targets, all_predictions

    def calculate_metrics(self, targets, predictions):
        """
        Calculate evaluation metrics.

        Args:
            targets: Ground truth values.
            predictions: Predicted values.

        Returns:
            Dictionary containing RMSE, MAE, R2, and MAPE.
        """
        mse = mean_squared_error(targets, predictions)
        metrics = {
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(targets, predictions),
            "R2": r2_score(targets, predictions),
            "MAPE": mean_absolute_percentage_error(targets, predictions)
        }
        return metrics

    def compare_models(self, models_dict, data_loader, scaler=None):
        """
        Compare multiple models on the same dataset.

        Args:
            models_dict: Dictionary of models to compare. Keys are model names, values are model instances.
            data_loader: DataLoader containing the dataset to evaluate.
            scaler: Optional scaler used for inverse transforming predictions and targets.

        Returns:
            DataFrame containing evaluation metrics for each model.
        """
        results = []

        for model_name, model in models_dict.items():
            print(f"Evaluating model: {model_name}...")
            targets, predictions = self.get_predictions(model, data_loader, scaler)
            metrics = self.calculate_metrics(targets, predictions)
            metrics["Model"] = model_name
            results.append(metrics)

        return pd.DataFrame(results).set_index("Model")

    def build_performance_dataframe(self, nested_models_dict, data_loaders_dict, scalers_dict, target_metric="MSE"):
        """
        Build a performance DataFrame for multiple models across different datasets.

        Args:
            models_dict: Dictionary of models to evaluate. Keys are model names, values are model instances.
            data_loaders_dict: Dictionary of DataLoaders for different datasets. Keys are dataset names, values are DataLoader instances.
            scalers_dict: Dictionary of scalers for each dataset. Keys are dataset names, values are scaler instances.
            target_metric: The metric to use for ranking models (default is "MSE").

        Returns:
            DataFrame containing evaluation target metric for each model across datasets.
        """
        # Initialize an empty dictionary to hold the performance data
        dataset_names = list(data_loaders_dict.keys())
        first_country = dataset_names[0]
        model_names = list(nested_models_dict[first_country].keys())
        performance_data = {model_name: [] for model_name in model_names}

        # Loop through each dataset
        for dataset_name, data_loader in data_loaders_dict.items():
            print(f"Evaluating models for dataset: {dataset_name}...")

            trained_models_in_dataset = nested_models_dict.get(dataset_name, {})
            scaler = scalers_dict.get(dataset_name, None)

            # Evaluate all models
            for model_name, model in trained_models_in_dataset.items():
                targets, predictions = self.get_predictions(model, data_loader, scaler)
                metrics = self.calculate_metrics(targets, predictions)

                if target_metric not in metrics:
                    raise KeyError(f"Metric '{target_metric}' not found in calculate_metrics output.")
                
                performance_data[model_name].append(metrics[target_metric])

        performance_df = pd.DataFrame(performance_data, index=dataset_names)
        return performance_df

    def friedman_test(self, performance_df):
        """
        Perform the Friedman test to compare multiple models across different datasets.

        Args:
            performance_df: DataFrame containing evaluation metrics for each model across datasets.

        Returns:
            Tuple containing the Friedman statistic and p-value.
        """

        # Extract the values for the Friedman test
        values = [performance_df[model].values for model in performance_df.columns.tolist()]

        # Perform the Friedman test
        stat, p_value = friedmanchisquare(*values)
        return stat, p_value

    def holm_bonferroni_posthoc(self, performance_df, control_model, alpha=0.05, metric_is_loss=True):
        """
        Perform Holm-Bonferroni post-hoc test for pairwise comparisons after the Friedman test.

        Args:
            performance_df: DataFrame containing evaluation metrics for each model across datasets.
            control_model: The column name of the proposed model.
            alpha: The significance level for the post-hoc test.
            metric_is_loss: Whether the metric is a loss function (True) or a score (False).

        Returns:
            DataFrame containing the results of the post-hoc test.
        """

        # 1. Validation
        if control_model not in performance_df.columns:
            raise ValueError(f"Control model '{control_model}' not found in performance DataFrame columns.")

        models = performance_df.columns.tolist()
        baselines = [m for m in models if m!= control_model]
        num_baselines = len(baselines)

        # Determine the direction of the Wilcoxon test 
        # based on whether the metric is a loss or a score
        alternative = 'less' if metric_is_loss else 'greater'

        raw_results = []
        p_values = []

        # 2. Calculate pairwise unadjusted p-values using the Wilcoxon signed-rank test
        for baseline in baselines:
            stat, p_value = stats.wilcoxon(performance_df[control_model], 
                                           performance_df[baseline], 
                                           alternative=alternative)
            p_values.append(p_value)
            raw_results.append({'baseline model': baseline, 'unadjusted p-value': p_value})

        # 3. Apply the Holm correction via statsmodels
        reject, adjusted_p_vals, _, _ = multipletests(p_values, alpha=alpha, method='holm')

        for i, res in enumerate(raw_results):
            res['Holm adjusted p-value'] = adjusted_p_vals[i]
            res['reject null hypothesis'] = reject[i]

        hb_df = pd.DataFrame(raw_results)

        # 4. Format the DataFrame to reflect the physical step-down process
        # Sort by unadjusted p-values (smallest to largest)
        hb_df = hb_df.sort_values(by='unadjusted p-value').reset_index(drop=True)

        # Calculate the dynamic Holm thresholds: alpha / (k - i + 1) where k is the number of hypotheses and i is 1-indexed
        hb_df['Step (i)'] = range(1, num_baselines + 1)
        hb_df['Holm threshold'] = alpha / (num_baselines - hb_df['Step (i)'] + 1)
        # Reorder columns for clarity, logical reading flow
        h_df = hb_df[['baseline model', 
                      'unadjusted p-value', 
                      'Step (i)',
                      'Holm threshold',
                      'Holm adjusted p-value', 
                      'reject null hypothesis']]
        
        return h_df.set_index('baseline model')