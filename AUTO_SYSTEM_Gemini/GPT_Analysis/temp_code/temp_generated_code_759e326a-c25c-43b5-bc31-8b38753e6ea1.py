# evaluate_model_performance.py
# Requires: numpy >= 1.21, pandas >= 1.0.0, scikit-learn >= 1.0, tensorflow >= 2.8 (optional), torch >= 1.10 (optional)

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Union, List
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
    n_features = scaler.n_features_in_
    dummy_array_pred = np.zeros((n_samples, n_features))
    dummy_array_pred[:, target_feature_index] = predictions_normalized.ravel()
    dummy_array_true = np.zeros((n_samples, n_features))
    dummy_array_true[:, target_feature_index] = y_test_normalized.ravel()
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
            preds_norm = make_predictions(model, X_test)
            if preds_norm.ndim == 1:
                preds_norm = preds_norm.reshape(-1, 1)
            y_test_norm = y_test
            if y_test_norm.ndim == 1:
                y_test_norm = y_test_norm.reshape(-1, 1)
            preds_actual, y_actual = inverse_transform_values(
                preds_norm, y_test_norm, scaler, close_index
            )
            # Ensure shape compatibility before metrics
            assert preds_actual.shape == y_actual.shape, \
                f"Prediction/true shape mismatch: {preds_actual.shape} vs {y_actual.shape}"
            metrics = calculate_metrics(y_actual, preds_actual)
            metrics_report[model_name] = metrics
            predictions_report[model_name] = preds_actual
            logger.info(f"{model_name} : RMSE = {metrics['RMSE']:.3f} | MAE = {metrics['MAE']:.3f}")
        except Exception as e:
            logger.error(f"Error evaluating model '{model_name}': {e}", exc_info=True)
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
        logger.info("Using mock data for demonstration.")
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "Open": np.linspace(100, 200, 100),
            "High": np.linspace(102, 202, 100),
            "Low": np.linspace(98, 198, 100),
            "Close": np.linspace(101, 201, 100),
        })

    # ---------- PATCH: Identify windowed OHLC columns ------------
    ohlc_feats: List[str] = ['Open', 'High', 'Low', 'Close']
    open_cols = [c for c in df.columns if c.startswith('Open_t-')]
    # Decide windowing mode
    if open_cols:
        # Assume columns like Open_t-59 ... Open_t-0
        try:
            seq_nums = [int(col.split('_t-')[1]) for col in open_cols]
            seq_len = max(seq_nums) + 1 if seq_nums else 60
        except Exception:
            seq_len = 60
        # Build correct ordered list of all feature columns for each t
        feature_cols = []
        for t in range(seq_len-1, -1, -1):
            for feat in ohlc_feats:
                col_name = f"{feat}_t-{t}"
                if col_name in df.columns:
                    feature_cols.append(col_name)
        N_FEATURES = len(ohlc_feats)
        SEQ_LEN = seq_len
        # Shape
        N = len(df)
        if len(feature_cols) != N_FEATURES * SEQ_LEN:
            logger.warning(f"Detected {len(feature_cols)} feature columns, expected {N_FEATURES * SEQ_LEN}. Columns: {feature_cols[:8]} ...")
        X_flat = df[feature_cols].values
        try:
            X_test = X_flat.reshape(-1, SEQ_LEN, N_FEATURES)
        except Exception as reshape_e:
            logger.error(f"Unable to reshape features: {reshape_e}")
            raise
        # Target
        if 'target' in df.columns:
            y_test = df['target'].values.reshape(-1, 1)
        else:
            raise ValueError("Input data must contain a 'target' column for y_test.")
        # Setup scaler to match feature order, using t-0 columns
        scaler_fit_cols: List[str] = [f"{feat}_t-0" for feat in ohlc_feats if f"{feat}_t-0" in df.columns]
        if len(scaler_fit_cols) != 4:
            raise ValueError("One or more OHLC t-0 columns are missing for scaler fitting: found " +
                             f"{scaler_fit_cols}")
        scaler = MinMaxScaler()
        scaler.fit(df[scaler_fit_cols])
        # Determine 'Close' index
        close_index = scaler_fit_cols.index('Close_t-0') if 'Close_t-0' in scaler_fit_cols else -1
    elif all(feat in df.columns for feat in ohlc_feats):
        # Raw OHLC single value per row mode
        ohlc_fake = df[ohlc_feats].values
        N, N_FEATURES = ohlc_fake.shape
        SEQ_LEN = 60
        # For the demo: create sliding windows (may not match production. For simple test/demo only!)
        if N >= SEQ_LEN + 1:
            X_test = np.zeros((N - SEQ_LEN, SEQ_LEN, N_FEATURES))
            y_test = np.zeros((N - SEQ_LEN, 1))
            for i in range(N - SEQ_LEN):
                X_test[i] = ohlc_fake[i:i+SEQ_LEN]
                y_test[i, 0] = ohlc_fake[i+SEQ_LEN, -1]  # 'Close' of next day
            scaler = MinMaxScaler()
            scaler.fit(ohlc_fake)
            close_index = ohlc_feats.index("Close")
        else:
            logger.error("Not enough data for sliding window construction.")
            raise ValueError("Insufficient rows for sequence building.")
    else:
        missing = [f for f in ohlc_feats if f not in df.columns and not any(c.startswith(f+"_t-") for c in df.columns)]
        raise ValueError(f"Cannot identify OHLC columns in data. Missing: {missing}")

    # Shape check
    assert X_test.shape[0] == y_test.shape[0], \
        f"Feature-label sample count mismatch: X_test {X_test.shape}, y_test {y_test.shape}"

    # --- Dummy Models ---
    class DummyKerasModel:
        def predict(self, x):
            # Return mean of 'Close' price in last day in the sequence plus noise
            return x[:, -1, -1:1].flatten() + np.random.uniform(-0.01, 0.01, size=(x.shape[0],))  # shape (N,)
    if torch is not None:
        class DummyTorchModel(torch.nn.Module):
            def forward(self, x):
                mean_open = torch.mean(x[:, -3:, 0], dim=1, keepdim=True)
                noise = torch.from_numpy(np.random.uniform(-0.01, 0.01, size=(x.shape[0], 1))).float()
                return mean_open + noise
        dummy_torch_model = DummyTorchModel()
    else:
        dummy_torch_model = None
    dummy_keras_model = DummyKerasModel()
    models = {"LSTM": dummy_keras_model}
    if dummy_torch_model is not None:
        models["Transformer"] = dummy_torch_model

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

if __name__ == "__main__":
    main()