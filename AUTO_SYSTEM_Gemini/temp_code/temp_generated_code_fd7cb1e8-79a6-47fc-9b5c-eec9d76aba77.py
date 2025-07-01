#
# Module: evaluate_model_performance
#
# Description:
# This module provides functions to quantitatively evaluate trained time-series
# forecasting models (e.g., LSTM, Transformer). It calculates standard error
# metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)
# on a test set to determine the predictive accuracy of each model.
#

# --- Dependencies ---
# Requires:
#   - python >= 3.8
#   - numpy >= 1.21.0
#   - scikit-learn >= 1.0.0
#   - pandas >= 1.3.0
#   - tensorflow >= 2.8.0 or torch >= 1.10.0 (for model objects)

import os
import logging
from typing import Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Task 1: Generate Predictions ---
def make_predictions(
    model: Any,
    X_test: np.ndarray
) -> np.ndarray:
    """
    Generate predictions for the test set using a trained model.

    Handles both TensorFlow/Keras and PyTorch model interfaces.

    Args:
        model: A trained model object (TensorFlow/Keras or PyTorch).
        X_test: A numpy.ndarray of shape (n_samples, sequence_length, n_features)
                representing the test features.

    Returns:
        A numpy.ndarray of shape (n_samples, 1) containing the model's
        predictions in the normalized scale [0, 1].

    Raises:
        AttributeError: If the model object does not have a 'predict' method
                        (for Keras-like models) or is not a callable (for PyTorch models).
        Exception: For any other errors during prediction.
    """
    try:
        # Keras-like model
        if hasattr(model, 'predict'):
            predictions_normalized = model.predict(X_test)
        # PyTorch-like model
        elif callable(model):
            # PyTorch model evaluation requires more setup (e.g., torch.no_grad).
            # This is a simplified placeholder. A real implementation would check
            # for 'torch' and handle tensor conversions.
            # For this example, we assume it's been wrapped to have a predict-like interface.
            # If a raw PyTorch model is passed, it might require a wrapper.
            logger.info("Attempting prediction with a callable model (assumed PyTorch-like).")
            # The following line is a common pattern for PyTorch models, but it is
            # commented out because it requires the torch library and tensor inputs.
            # import torch
            # model.eval()
            # with torch.no_grad():
            #     predictions_normalized = model(torch.from_numpy(X_test).float()).numpy()
            raise NotImplementedError("Direct PyTorch model evaluation needs a specific wrapper. "
                                      "Please wrap it in a class with a .predict() method.")
        else:
            raise AttributeError("Model object is not a recognized type (Keras/PyTorch).")

        # Ensure predictions are 2D (n_samples, 1)
        if predictions_normalized.ndim == 1:
            predictions_normalized = predictions_normalized.reshape(-1, 1)

        return predictions_normalized

    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise


# --- Task 2: Inverse Transform Values ---
def inverse_transform_values(
    predictions_normalized: np.ndarray,
    y_test_normalized: np.ndarray,
    scaler: MinMaxScaler
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse transform the predicted and actual values back to their original scale.

    This function correctly handles scalers that were fit on multiple features.

    Args:
        predictions_normalized: A numpy.ndarray of shape (n_samples, 1)
                                containing normalized predictions.
        y_test_normalized: A numpy.ndarray of shape (n_samples, 1)
                           containing the true normalized values.
        scaler: A fitted scikit-learn MinMaxScaler object. It is assumed that the
                target variable ('Close' price) was the last column the scaler was
                fitted on.

    Returns:
        A tuple containing:
        - predictions_actual_scale: A numpy.ndarray of predictions in the original price scale.
        - y_test_actual_scale: A numpy.ndarray of true values in the original price scale.
    """
    if not hasattr(scaler, 'n_features_in_'):
        raise ValueError("Scaler must be a fitted scikit-learn MinMaxScaler.")
        
    num_features = scaler.n_features_in_

    # Create dummy arrays with the same number of features as the original data
    dummy_predictions = np.zeros((len(predictions_normalized), num_features))
    dummy_actuals = np.zeros((len(y_test_normalized), num_features))

    # Place the prediction and actual values in the last column
    # This assumes the target ('Close' price) was the last feature scaled
    dummy_predictions[:, -1] = predictions_normalized.ravel()
    dummy_actuals[:, -1] = y_test_normalized.ravel()

    # Perform the inverse transformation
    predictions_actual_scale = scaler.inverse_transform(dummy_predictions)[:, -1]
    y_test_actual_scale = scaler.inverse_transform(dummy_actuals)[:, -1]

    return predictions_actual_scale, y_test_actual_scale


# --- Task 3: Calculate Metrics ---
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

    Args:
        y_true: A numpy.ndarray of actual target values.
        y_pred: A numpy.ndarray of predicted values.

    Returns:
        A dictionary containing the calculated 'RMSE' and 'MAE' as float values.
        Example: {'RMSE': 10.5, 'MAE': 8.2}
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': float(rmse), 'MAE': float(mae)}


# --- Main Evaluation Orchestrator ---
def evaluate_all_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test_normalized: np.ndarray,
    scaler: MinMaxScaler
) -> Dict[str, Dict]:
    """
    Evaluates a dictionary of trained models on the test data and aggregates results.

    Args:
        models: A dictionary where keys are model names (e.g., 'LSTM') and values
                are the trained model objects.
        X_test: The test data features (normalized).
        y_test_normalized: The true test data targets (normalized).
        scaler: The scaler used to normalize the data.

    Returns:
        A dictionary containing two keys:
        - 'metrics_report': A dictionary of performance metrics for each model.
        - 'predictions_report': A dictionary of predictions for each model.
    """
    metrics_report = {}
    predictions_report = {}

    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        try:
            # Step 1: Generate predictions
            predictions_normalized = make_predictions(model, X_test)

            # Step 2: Inverse transform values
            predictions_actual, y_test_actual = inverse_transform_values(
                predictions_normalized,
                y_test_normalized,
                scaler
            )

            # Step 3: Calculate metrics
            metrics = calculate_metrics(y_test_actual, predictions_actual)
            logger.info(f"Metrics for {model_name}: {metrics}")

            # Store results
            metrics_report[model_name] = metrics
            predictions_report[model_name] = predictions_actual

        except Exception as e:
            # As per requirements, log the error and continue with the next model
            logger.error(f"Could not evaluate model '{model_name}'. Reason: {e}", exc_info=True)
            metrics_report[model_name] = {'error': str(e)}
            predictions_report[model_name] = None

    return {
        'metrics_report': metrics_report,
        'predictions_report': predictions_report
    }


# --- Data Loading Utility (as per Implementation Guidelines) ---
def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from the specified file path.

    Args:
        data_path: The path to the data file (CSV or JSON).

    Returns:
        A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or data loading fails.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File not found at {data_path}")
    try:
        if data_path.lower().endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.lower().endswith('.json'):
            return pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
    except Exception as e:
        raise ValueError(f"Error loading data from {data_path}: {e}")


# --- Main Entry Point for Demonstration (as per Implementation Guidelines) ---
def main():
    """
    Main function to demonstrate the usage of the evaluation module.
    This function uses mock data and models for stand-alone execution.
    """
    logger.info("--- Starting Model Evaluation Demonstration ---")

    # Use default data path to automatically load data
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\generated\expected_input.csv'

    try:
        df = load_data(data_path)
        logger.info("Data loaded successfully from specified path. Head of the data:")
        print(df.head().to_string())
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"{e}")
        logger.info("Using mock data as a fallback.")
        # Create mock data if the file is not found
        mock_data = {
            'Date': pd.to_datetime(pd.date_range(start='2022-01-01', periods=200)),
            'Open': np.random.uniform(100, 500, 200),
            'High': np.random.uniform(100, 500, 200),
            'Low': np.random.uniform(100, 500, 200),
            'Close': np.linspace(200, 400, 200) + np.random.normal(0, 10, 200)
        }
        df = pd.DataFrame(mock_data)
        logger.info("Mock data created. Head of the data:")
        print(df.head().to_string())

    # --- Create Mock Objects for Demonstration ---
    # In a real pipeline, these would be the actual outputs from previous modules.
    
    # 1. Mock Scaler (fitted on 4 features: O, H, L, C)
    scaler = MinMaxScaler()
    # Let's assume the 'Close' prices range from 100 to 500 for scaling purposes
    mock_training_data = np.random.rand(100, 4) * 400 + 100
    scaler.fit(mock_training_data)

    # 2. Mock Test Data (100 samples, 60-day sequence, 4 features)
    n_test_samples = 50
    sequence_length = 60
    n_features = 4
    X_test = np.random.rand(n_test_samples, sequence_length, n_features).astype(np.float32)
    
    # Mock true values (normalized)
    y_test_normalized = np.random.rand(n_test_samples, 1).astype(np.float32)

    # 3. Mock Models (with a .predict() method)
    class MockModel:
        def __init__(self, noise_level: float = 0.05):
            self._noise = noise_level
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Simulates prediction by adding noise to a synthetic sine wave."""
            # This is just a placeholder to return data of the correct shape and type.
            # A real model's output would depend on X.
            num_predictions = X.shape[0]
            base_prediction = (np.sin(np.arange(num_predictions) * 0.1) + 1) / 2 # Normalize to [0,1]
            noise = np.random.normal(0, self._noise, num_predictions)
            return (base_prediction + noise).reshape(-1, 1).astype(np.float32)

    mock_lstm_model = MockModel(noise_level=0.05)
    mock_transformer_model = MockModel(noise_level=0.03)

    models_to_evaluate = {
        'LSTM': mock_lstm_model,
        'Transformer': mock_transformer_model
    }

    # --- Run Evaluation ---
    logger.info("\n--- Running evaluation on mock models ---")
    evaluation_results = evaluate_all_models(
        models=models_to_evaluate,
        X_test=X_test,
        y_test_normalized=y_test_normalized,
        scaler=scaler
    )

    # --- Print Results ---
    print("\n" + "="*50)
    print("           MODEL EVALUATION REPORT")
    print("="*50)
    
    print("\n--- Metrics Report ---")
    metrics_report = evaluation_results.get('metrics_report', {})
    if metrics_report:
        print(pd.DataFrame.from_dict(metrics_report, orient='index').round(4))
    else:
        print("No metrics were generated.")
        
    print("\n--- Predictions Report ---")
    predictions_report = evaluation_results.get('predictions_report', {})
    if predictions_report:
        # Since actuals are not part of the report, we calculate them once for comparison.
        # The first argument to `inverse_transform_values` is a placeholder as it's not used for actuals.
        _, y_test_actual = inverse_transform_values(
            y_test_normalized, y_test_normalized, scaler
        )

        for model_name, predictions in predictions_report.items():
            if predictions is not None:
                print(f"\nFirst 5 predictions for '{model_name}':")
                pred_df = pd.DataFrame({
                    'Actual Price': y_test_actual[:5],
                    'Predicted Price': predictions[:5]
                })
                print(pred_df.to_string(index=False))
            else:
                print(f"\nPredictions for '{model_name}' are not available due to an error.")
    else:
        print("No predictions were generated.")
        
    print("\n" + "="*50)
    logger.info("--- Model Evaluation Demonstration Finished ---")


if __name__ == '__main__':
    main()