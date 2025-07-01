# evaluate_model_performance.py
# Requires: numpy >= 1.21, pandas >= 1.0.0, scikit-learn >= 1.0, tensorflow >= 2.8 (optional), torch >= 1.10 (optional)

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Optional: TensorFlow/PyTorch imports only if models provided are those types
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
except ImportError:
    torch = None

# ---------------------- Logging Setup --------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============ Utility Functions ==============

def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads the data from the specified path. Supports .csv and .json formats.

    Args:
        data_path (str): Path to the data file.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported or file is corrupt.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File not found at {data_path}")
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            return df
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
            return df
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def make_predictions(model: object, X_test: np.ndarray) -> np.ndarray:
    """
    Generate predictions for the test set using a trained model, supporting TensorFlow (Keras) and PyTorch.

    Args:
        model: Trained TensorFlow/Keras `Model` or PyTorch `nn.Module`.
        X_test: Array (n_samples, seq_len, n_features).

    Returns:
        predictions_normalized: np.ndarray of shape (n_samples, 1) in normalized scale.
    """
    # TensorFlow/Keras
    if tf is not None and isinstance(model, (tf.keras.Model,)):
        preds = model.predict(X_test)
        # Keras may return (n,1) or (n,) depending on model architecture
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds
    # PyTorch
    elif torch is not None and isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            if not isinstance(X_test, torch.Tensor):
                X_test_tensor = torch.from_numpy(X_test).float()
            else:
                X_test_tensor = X_test
            preds = model(X_test_tensor)
            if hasattr(preds, "numpy"):
                preds = preds.numpy()
            else:
                preds = preds.detach().cpu().numpy()
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            return preds
    # Could be a custom or legacy object with .predict()
    elif hasattr(model, "predict"):
        preds = model.predict(X_test)
        if isinstance(preds, list):
            preds = np.array(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds
    else:
        raise TypeError("Unsupported model type for prediction. Must be TensorFlow/Keras, PyTorch, or an object with .predict().")

def inverse_transform_values(
    predictions_normalized: np.ndarray,
    y_test_normalized: np.ndarray,
    scaler: MinMaxScaler,
    target_feature_index: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse transform the predicted and actual values back to their original stock price scale.

    Args:
        predictions_normalized: np.ndarray, shape (n_samples, 1)
        y_test_normalized: np.ndarray, shape (n_samples, 1)
        scaler: A fitted sklearn MinMaxScaler object (on all OHLC columns).
        target_feature_index: int, column index of 'Close' in the original scaler.

    Returns:
        Tuple:
            predictions_actual_scale: np.ndarray, shape (n_samples,)
            y_test_actual_scale: np.ndarray, shape (n_samples,)
    """
    n_samples = predictions_normalized.shape[0]
    # Determine feature count from scaler
    n_features = scaler.n_features_in_
    # Fill dummy arrays with zeros except the 'Close' column
    dummy_array_pred = np.zeros((n_samples, n_features))
    dummy_array_pred[:, target_feature_index] = predictions_normalized.ravel()
    dummy_array_true = np.zeros((n_samples, n_features))
    dummy_array_true[:, target_feature_index] = y_test_normalized.ravel()
    # Inverse transform
    pred_actual = scaler.inverse_transform(dummy_array_pred)[:, target_feature_index]
    true_actual = scaler.inverse_transform(dummy_array_true)[:, target_feature_index]
    return pred_actual, true_actual

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate RMSE and MAE.

    Args:
        y_true: np.ndarray of true values.
        y_pred: np.ndarray of predicted values.

    Returns:
        metrics: {'RMSE': rmse, 'MAE': mae}
    """
    # Ensure 1D arrays
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": float(rmse), "MAE": float(mae)}

def evaluate_model_performance(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    close_index: int = -1
) -> Dict[str, Any]:
    """
    Evaluate a dict of trained models on the test data.

    Args:
        models: Dict where key is model name and value is the trained model instance.
        X_test: np.ndarray, test data features.
        y_test: np.ndarray, true test data targets (normalized).
        scaler: MinMaxScaler fitted on all OHLC columns.
        close_index: int (default=-1): index in scaler corresponding to 'Close' column.

    Returns:
        metrics_report: Dict of 'model_name' -> {'RMSE': float, 'MAE': float}
        predictions_report: Dict of 'model_name' -> np.ndarray (original scale)
    """
    metrics_report = {}
    predictions_report = {}

    for model_name, model in models.items():
        try:
            logger.info(f"Evaluating model: {model_name}")
            # 1. Generate predictions (normalized)
            preds_norm = make_predictions(model, X_test)

            # Shape fix
            if preds_norm.ndim == 1:
                preds_norm = preds_norm.reshape(-1, 1)
            y_test_norm = y_test
            if y_test_norm.ndim == 1:
                y_test_norm = y_test_norm.reshape(-1, 1)
            # 2. Inverse transform
            preds_actual, y_actual = inverse_transform_values(
                preds_norm, y_test_norm, scaler, close_index
            )
            # 3. Metrics
            metrics = calculate_metrics(y_actual, preds_actual)
            metrics_report[model_name] = metrics
            predictions_report[model_name] = preds_actual
            logger.info(f"{model_name} : RMSE = {metrics['RMSE']:.3f} | MAE = {metrics['MAE']:.3f}")
        except Exception as e:
            logger.error(f"Error evaluating model '{model_name}': {e}", exc_info=True)
            # Continue with other models

    return {
        'metrics_report': metrics_report,
        'predictions_report': predictions_report
    }


# ============== Main (for demo/testing) ==============
def main():
    # ---- Config/Paths ----
    DATA_PATH = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated\expected_input_for_training.csv'

    try:
        df = load_data(DATA_PATH)
        logger.info("Data loaded successfully. Head:")
        print(df.head())
    except FileNotFoundError:
        logger.error(f"Data file not found at path: {DATA_PATH}")
        # Optional: Use mock data to allow the rest of the test/demo to run
        logger.info("Using mock data for demonstration.")
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "Open": np.linspace(100, 200, 100),
            "High": np.linspace(102, 202, 100),
            "Low": np.linspace(98, 198, 100),
            "Close": np.linspace(101, 201, 100),
        })

    # ---- Demo/test for evaluate_model_performance ----
    # The following is MOCK demo logic for illustration/testing of the metric functions,
    # not for real stock modeling or segmentation

    # Simulate: create normalized sequences for 1 model as in use-case
    np.random.seed(42)
    N = 32
    SEQ_LEN = 60
    N_FEATURES = 4  # 'Open', 'High', 'Low', 'Close'
    scaler = MinMaxScaler()
    ohlc_fake = df[['Open', 'High', 'Low', 'Close']].values
    scaler.fit(ohlc_fake)
    ohlc_norm = scaler.transform(ohlc_fake)

    # Prepare dummy sequences (sliding window)
    # For illustration, create 32 sequences with past SEQ_LEN days
    X_test = np.zeros((N, SEQ_LEN, N_FEATURES))
    y_test = np.zeros((N, 1))
    for i in range(N):
        if i + SEQ_LEN >= len(ohlc_norm):
            break
        X_test[i] = ohlc_norm[i:i+SEQ_LEN]
        y_test[i, 0] = ohlc_norm[i+SEQ_LEN, -1]  # Close of the next day

    # --- Dummy Models ---
    class DummyKerasModel:
        # Mocks tf.keras.Model
        def predict(self, x):
            # Dummy: simply returns the mean of 'Close' for the last day in the sequence plus noise
            return x[:, -1, -1:1].flatten() + np.random.uniform(-0.01, 0.01, size=(x.shape[0],))  # shape (N,)

    class DummyTorchModel(torch.nn.Module):
        def forward(self, x):
            # x: (batch, seq_len, n_features)
            # Use mean of 'Open' price in last 3 days (normalized) as prediction + noise
            mean_open = torch.mean(x[:, -3:, 0], dim=1, keepdim=True)
            noise = torch.from_numpy(np.random.uniform(-0.01, 0.01, size=(x.shape[0], 1))).float()
            return mean_open + noise

    dummy_keras_model = DummyKerasModel()

    if torch is not None:
        dummy_torch_model = DummyTorchModel()
    else:
        dummy_torch_model = None

    models = {"LSTM": dummy_keras_model}
    if dummy_torch_model is not None:
        models["Transformer"] = dummy_torch_model

    # Determine index of 'Close' in scaler
    close_index = -1  # By default, -1 (last col)
    try:
        # If original col names are kept in scaler:
        # scaler.feature_names_in_ uses the trained feature order
        if hasattr(scaler, 'feature_names_in_'):
            names = scaler.feature_names_in_
            if 'Close' in names:
                close_index = int(np.where(names == 'Close')[0][0])
    except Exception:
        close_index = -1

    # ---- Call evaluation function
    results = evaluate_model_performance(
        models=models,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        close_index=close_index
    )

    # Print aggregated results
    print("Aggregated Evaluation Report:")
    print("  [metrics_report]:")
    for model_name, metrics in results['metrics_report'].items():
        print(f"    {model_name}: {metrics}")
    print("  [predictions_report]: (first 5 values per model)")
    for model_name, preds in results['predictions_report'].items():
        print(f"    {model_name}: {preds[:5]}")

# If run directly, execute main
if __name__ == "__main__":
    main()

# ------- END OF MODULE -------