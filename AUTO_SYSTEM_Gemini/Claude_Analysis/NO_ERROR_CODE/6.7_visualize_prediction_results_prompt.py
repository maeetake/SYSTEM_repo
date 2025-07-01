# Requires: scikit-learn >= 1.0, numpy >= 1.21, tensorflow >= 2.8 or torch >= 1.10
import logging
from typing import Dict, Any, Union

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def make_predictions(
    model: Any,
    X_test: np.ndarray
) -> np.ndarray:
    """
    Generate predictions for the test set using a trained model.

    This function takes a trained model (TensorFlow/Keras or PyTorch) and
    the test feature set to produce predictions.

    Args:
        model (Any): A trained model object (TensorFlow/Keras or PyTorch).
                     The model should have a `predict` method (for Keras) or be
                     callable and set to evaluation mode (for PyTorch).
        X_test (np.ndarray): A numpy.ndarray of shape
                             (n_samples, sequence_length, n_features)
                             representing the test features.

    Returns:
        np.ndarray: A numpy.ndarray of shape (n_samples, 1) containing
                    the model's predictions in the normalized scale [0, 1].

    Raises:
        ValueError: If X_test is not a valid numpy array.
        Exception: For any errors during model prediction.
    """
    if not isinstance(X_test, np.ndarray) or X_test.ndim != 3:
        msg = "X_test must be a 3D numpy array."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Generating predictions on the test set.")
    try:
        # The logic handles a standard Keras model.predict() call.
        # For a PyTorch model, this would need adaptation (e.g., model.eval(), torch.no_grad()).
        # Given the context of the skeleton, a Keras model is assumed.
        predictions_normalized = model.predict(X_test)
        
        # Ensure output is 2D (n_samples, 1)
        if predictions_normalized.ndim == 1:
            predictions_normalized = predictions_normalized.reshape(-1, 1)

        logger.info("Successfully generated predictions.")
        return predictions_normalized
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise


def inverse_transform_values(
    predictions_normalized: np.ndarray,
    y_test_normalized: np.ndarray,
    scaler: MinMaxScaler
) -> Dict[str, np.ndarray]:
    """
    Inverse transform the predicted and actual values back to their original scale.

    Since the scaler was fitted on multiple features (e.g., OHLC), this function
    reconstructs the data structure to correctly apply `inverse_transform` on the
    target variable ('Close' price), which is assumed to be the last feature.

    Args:
        predictions_normalized (np.ndarray): A numpy.ndarray of shape (n_samples, 1)
                                             containing the normalized predictions.
        y_test_normalized (np.ndarray): A numpy.ndarray of shape (n_samples, 1)
                                        containing the true normalized values.
        scaler (MinMaxScaler): The fitted scikit-learn MinMaxScaler object used
                               for the original data normalization.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the predictions and true
                               values in their original price scale.
                               Example: {'predictions_actual_scale': array([...]),
                                         'y_test_actual_scale': array([...])}
    """
    logger.info("Inverse transforming predicted and actual values.")
    try:
        if not hasattr(scaler, 'n_features_in_'):
            msg = "Scaler object appears to be invalid or not fitted."
            logger.error(msg)
            raise ValueError(msg)
        
        n_features = scaler.n_features_in_

        # Create dummy arrays with the original number of features
        dummy_predictions = np.zeros((len(predictions_normalized), n_features))
        dummy_y_test = np.zeros((len(y_test_normalized), n_features))

        # Place the normalized values in the last column, assuming 'Close' was the last feature
        dummy_predictions[:, -1] = predictions_normalized.ravel()
        dummy_y_test[:, -1] = y_test_normalized.ravel()

        # Perform the inverse transformation
        predictions_actual_scale = scaler.inverse_transform(dummy_predictions)[:, -1]
        y_test_actual_scale = scaler.inverse_transform(dummy_y_test)[:, -1]

        logger.info("Successfully inverse transformed the values.")
        return {
            'predictions_actual_scale': predictions_actual_scale,
            'y_test_actual_scale': y_test_actual_scale
        }
    except Exception as e:
        logger.error(f"Failed to inverse transform values: {e}")
        raise


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

    Args:
        y_true (np.ndarray): A numpy.ndarray of actual target values.
        y_pred (np.ndarray): A numpy.ndarray of predicted values.

    Returns:
        Dict[str, float]: A dictionary containing the calculated 'RMSE' and 'MAE'
                          as float values. Example: {'RMSE': 10.5, 'MAE': 8.2}
    """
    logger.info("Calculating performance metrics (RMSE, MAE).")
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics = {'RMSE': float(rmse), 'MAE': float(mae)}
        logger.info(f"Calculated metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        raise