# train_models.py

# Requires:
#   pandas >= 1.0.0
#   numpy >= 1.18
#   scikit-learn >= 1.0
#   tensorflow >= 2.10  (see below for PyTorch compatibility comment)
#
# Note: This module implements the 'train_models' phase for time-series forecasting with OHLC data only.
#       It is designed primarily for TensorFlow. PyTorch is not directly implemented here,
#       but the training loop structure can be adapted similarly for torch.nn.Module if required.

import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, Union

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

# Set up logging to file and standard output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads a dataset from the specified path.
    Args:
        data_path (str): Path to the data file (CSV or JSON).
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If load fails due to format or other error.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File not found at {data_path}")
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")
    return df


def preprocess_data(
    df: pd.DataFrame,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Cleans data, normalizes features, and generates sequential (X, y) pairs.

    Args:
        df (pd.DataFrame): Raw OHLC data with columns ['Date', 'Open', 'High', 'Low', 'Close']
        sequence_length (int): Number of days to use as input sequence.

    Returns:
        X (np.ndarray): 3D array (num_samples, sequence_length, num_features)
        y (np.ndarray): 1D array (num_samples,)
        scaler (MinMaxScaler): Fitted scaler for inverse_transform later.
    """
    # Only use OHLC columns (ignore Date and others if present)
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("Data must contain 'Open', 'High', 'Low', 'Close' columns.")
    ohlc_data = df[['Open', 'High', 'Low', 'Close']].copy()
    ohlc_data = ohlc_data.astype(float)

    # Handle missing values: forward fill, then (if still missing) backward fill
    ohlc_data.ffill(inplace=True)
    ohlc_data.bfill(inplace=True)

    # Normalize features using MinMaxScaler to range [0, 1]
    scaler = MinMaxScaler()
    ohlc_data_scaled = scaler.fit_transform(ohlc_data.values)

    # Generate sliding window sequences
    X, y = [], []
    num_samples = len(ohlc_data_scaled)
    for idx in range(sequence_length, num_samples):
        X.append(ohlc_data_scaled[idx-sequence_length:idx])
        y.append(ohlc_data_scaled[idx, 3])  # only Close as target
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, scaler


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_params: Dict[str, Any]
) -> Tuple[tf.keras.Model, Dict[str, list]]:
    """
    Trains a given Keras model on the provided training and validation datasets.

    Args:
        model (tf.keras.Model): The compiled Keras model to be trained.
        X_train (np.ndarray): Training data features, shape (n_samples, seq_len, n_features)
        y_train (np.ndarray): Training data targets, shape (n_samples,)
        X_val (np.ndarray): Validation data features.
        y_val (np.ndarray): Validation data targets.
        training_params (dict): Dictionary of training parameters like 'epochs', 'batch_size'

    Returns:
        (tf.keras.Model, Dict): The trained model and a dict containing lists of training/validation loss
    """

    # Set up EarlyStopping
    patience = training_params.get('early_stopping_patience', 10)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # Model checkpoint: best model by val_loss
    checkpoint_path = training_params.get('checkpoint_path', f'best_{model.name}.h5')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    logging.info(f"--- Starting Training for model: {model.name} ---")

    # Ensure data are float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)

    history = model.fit(
        X_train, y_train,
        epochs=training_params.get('epochs', 50),
        batch_size=training_params.get('batch_size', 32),
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1  # Print loss every epoch
    )

    logging.info(f"--- Finished Training for model: {model.name} ---")

    # Reload best weights (defensive: in case best is not last)
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    # Prepare loss history as dict for visualization
    loss_history = {
        'loss': history.history.get('loss', []),
        'val_loss': history.history.get('val_loss', [])
    }

    return model, loss_history


def split_data_chronologically(
    X: np.ndarray, y: np.ndarray,
    train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data chronologically into train/val/test using ratios.
    Args:
        X (np.ndarray): Feature sequences.
        y (np.ndarray): Targets.
        train_ratio (float): Ratio of data to use for training.
        val_ratio (float): Ratio of data to use for validation.
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    n = len(X)
    train_end = int(np.floor(train_ratio * n))
    val_end = int(np.floor((train_ratio + val_ratio) * n))
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_lstm_model(
    sequence_length: int,
    num_features: int
) -> tf.keras.Model:
    """
    Builds and compiles an LSTM model for regression.
    Args:
        sequence_length (int): Input sequence length.
        num_features (int): Number of input features per timestep (should be 4).
    Returns:
        tf.keras.Model: Compiled model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, num_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)  # Linear output for regression
    ])
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[]
    )
    model.name = "LSTM"
    return model


def build_transformer_model(
    sequence_length: int,
    num_features: int
) -> tf.keras.Model:
    """
    Builds and compiles a simple Transformer-based model for regression.
    Args:
        sequence_length (int): Input sequence length.
        num_features (int): Number of input features.
    Returns:
        tf.keras.Model: Compiled model.
    """
    # Simple Transformer encoder architecture suitable for time series
    inputs = tf.keras.layers.Input(shape=(sequence_length, num_features))
    # Positional encoding for time series: Add a learned embedding or simple sin/cos
    pos_encoding = tf.keras.layers.Embedding(
        input_dim=sequence_length, output_dim=num_features
    )(tf.range(start=0, limit=sequence_length, delta=1))
    x = inputs + pos_encoding
    # Transformer Encoder layers
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=num_features)(x, x)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Transformer")
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[]
    )
    return model


def main():
    # Hardcoded data path as per spec
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated\expected_input_for_training.csv'
    sequence_length = 60

    # Load the data
    try:
        df = load_data(data_path)
        logging.info("Data loaded successfully. Head of the data:")
        print(df.head())
    except FileNotFoundError as fnf_err:
        logging.error(str(fnf_err))
        # As fallback, provide small mock data for further code testing
        logging.warning("Using mock data!")
        # 64 days of OHLC so 4 samples with seq_len=60
        mock_dates = pd.date_range("2020-01-01", periods=64)
        mock_data = pd.DataFrame({
            'Date': mock_dates,
            'Open': np.linspace(100, 130, 64),
            'High': np.linspace(102, 133, 64),
            'Low': np.linspace(99, 129, 64),
            'Close': np.linspace(101, 131, 64),
        })
        df = mock_data
        print(df.head())
    except Exception as e:
        logging.error(f"An error occurred during data loading: {e}")
        return

    # Preprocess data into sequences
    try:
        X, y, scaler = preprocess_data(df, sequence_length=sequence_length)
        logging.info(f"Generated {X.shape[0]} input sequences of shape ({sequence_length}, {X.shape[2]}).")
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        return

    # Split the data: 80% train, 10% val, 10% test (chronologically)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_chronologically(X, y)

    logging.info(f"Split data: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test sequences.")

    # Platform: Use GPU if available
    if tf.config.list_physical_devices('GPU'):
        logging.info("GPU found. Training will use GPU.")
    else:
        logging.warning("No GPU found. Training will be performed on CPU.")

    # -------- Train LSTM Model --------
    lstm_model = build_lstm_model(sequence_length, X.shape[2])
    lstm_params = {
        'epochs': 50,
        'batch_size': 32,
        'early_stopping_patience': 10,  # Early stop if no improvement in 10 epochs
        'checkpoint_path': 'best_LSTM_model.h5'
    }
    lstm_trained_model, lstm_history = train_model(
        lstm_model, X_train, y_train, X_val, y_val, lstm_params
    )

    logging.info(f"LSTM training history (last 5 epochs): { {k: v[-5:] for k, v in lstm_history.items()} }")

    # -------- Train Transformer Model --------
    transformer_model = build_transformer_model(sequence_length, X.shape[2])
    trans_params = {
        'epochs': 50,
        'batch_size': 32,
        'early_stopping_patience': 10,
        'checkpoint_path': 'best_Transformer_model.h5'
    }
    transformer_trained_model, transformer_history = train_model(
        transformer_model, X_train, y_train, X_val, y_val, trans_params
    )

    logging.info(f"Transformer training history (last 5 epochs): { {k: v[-5:] for k, v in transformer_history.items()} }")

    # Save models (artifacts)
    try:
        lstm_trained_model.save('LSTM_stock_predictor.h5')
        transformer_trained_model.save('Transformer_stock_predictor.h5')
        logging.info("Saved trained models as 'LSTM_stock_predictor.h5' and 'Transformer_stock_predictor.h5'")
    except Exception as e:
        logging.warning(f"Could not save models to disk: {e}")

    # For future: return, yield, or write history to file as required by downstream modules
    return {
        "lstm_model": lstm_trained_model,
        "transformer_model": transformer_trained_model,
        "lstm_history": lstm_history,
        "transformer_history": transformer_history,
        "scaler": scaler,
        "test_set": (X_test, y_test)
    }


if __name__ == "__main__":
    # Run main only if executed directly; do NOT require user input for any parameters
    main()