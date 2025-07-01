# Revised Integrated Script
# The following import statements are for illustrative purposes,
# representing the modular structure of a complete project.
# The integrated script below is self-contained.
from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model
from PACKAGE.e_train_models_prompt import train_model

# ==============================================================================
# Integrated Source Code
# ==============================================================================

# main.py
# This script serves as the executable entry point for the full data preparation, model building,
# training, evaluation, and visualization pipeline. It integrates all necessary components
# into a single, functional script.

# ==============================================================================
# Dependencies
# ==============================================================================
# Requires: pandas, numpy, scikit-learn, tensorflow, matplotlib
import pandas as pd
import numpy as np
import logging
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any
import os
import matplotlib.pyplot as plt

# ==============================================================================
# Module: a_load_user_provided_data_prompt
# ==============================================================================
def load_data_from_csv(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): The full path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    
    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        ValueError: If the file is empty or cannot be parsed.
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("CSV file is empty.")
        logging.info(f"Successfully loaded data from: {filepath}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at path: {filepath}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while reading the CSV file: {e}")
        raise ValueError(f"Could not parse CSV file: {e}")

# ==============================================================================
# Module: b_preprocess_data_prompt
# ==============================================================================
class DataPreprocessor:
    """
    A class to handle preprocessing of time-series financial data.
    This includes cleaning, feature selection, scaling, sequence creation, and splitting.
    """
    def __init__(self, sequence_length: int = 60, train_split_ratio: float = 0.8, val_split_ratio: float = 0.1):
        """
        Initializes the DataPreprocessor.

        Args:
            sequence_length (int): The number of time steps in each input sequence.
            train_split_ratio (float): The proportion of the dataset to reserve for the training set.
            val_split_ratio (float): The proportion of the dataset to reserve for the validation set.
                                     The test set size is inferred from 1 - train_split_ratio - val_split_ratio.
        """
        if not (0 < train_split_ratio < 1 and 0 < val_split_ratio < 1 and train_split_ratio + val_split_ratio < 1):
            raise ValueError("Split ratios must be between 0 and 1, and their sum must be less than 1.")
        self.sequence_length = sequence_length
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.original_dates = None

    def _clean_and_prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and prepares the DataFrame."""
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        
        feature_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in feature_columns):
             raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns.")

        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df[feature_columns] = df[feature_columns].ffill()
        df = df.dropna(subset=feature_columns)
        
        self.original_dates = df.index
        return df[feature_columns]

    def _create_sequences(self, data: np.ndarray):
        """Creates sequences and corresponding labels from the data."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 3])  # Target is the 'Close' price (index 3)
        return np.array(X), np.array(y)

    def process(self, df: pd.DataFrame) -> dict:
        """
        Executes the full preprocessing pipeline.

        Args:
            df (pd.DataFrame): The raw input DataFrame.

        Returns:
            dict: A dictionary containing split datasets, the scaler, and test dates.
        """
        prepared_df = self._clean_and_prepare_df(df)
        scaled_data = self.scaler.fit_transform(prepared_df)
        X, y = self._create_sequences(scaled_data)
        
        if len(X) == 0:
            raise ValueError(f"Not enough data to create sequences with length {self.sequence_length}. "
                             f"Need at least {self.sequence_length + 1} data points, but found {len(prepared_df)}.")

        # Correctly implement chronological split using cumulative indices
        total_sequences = len(X)
        train_split_idx = int(total_sequences * self.train_split_ratio)
        val_split_idx = int(total_sequences * (self.train_split_ratio + self.val_split_ratio))
        
        X_train, y_train = X[:train_split_idx], y[:train_split_idx]
        X_val, y_val = X[train_split_idx:val_split_idx], y[train_split_idx:val_split_idx]
        X_test, y_test = X[val_split_idx:], y[val_split_idx:]
        
        # Dates for the test set correspond to the 'y' values
        test_dates_start_idx = self.sequence_length + val_split_idx
        test_dates = self.original_dates[test_dates_start_idx:]

        return {
            'X_train': X_train.astype(np.float32), 'y_train': y_train.astype(np.float32),
            'X_val': X_val.astype(np.float32), 'y_val': y_val.astype(np.float32),
            'X_test': X_test.astype(np.float32), 'y_test': y_test.astype(np.float32),
            'scaler': self.scaler, 'test_dates': test_dates
        }

# ==============================================================================
# Module: d_build_lstm_and_transformer_models_prompt
# ==============================================================================

def build_lstm_model(input_shape: tuple) -> keras.Model:
    """Builds an LSTM model architecture."""
    model = keras.Sequential([
        layers.Input(shape=input_shape, name='input_layer'),
        layers.LSTM(50, return_sequences=True, name='lstm_1'),
        layers.Dropout(0.2, name='dropout_1'),
        layers.LSTM(50, return_sequences=False, name='lstm_2'),
        layers.Dropout(0.2, name='dropout_2'),
        layers.Dense(25, activation='relu', name='dense_1'),
        layers.Dense(1, name='output_layer')
    ], name="LSTM_Model")
    return model

class PositionalEncoding(layers.Layer):
    def __init__(self, position: int, d_model: int, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position, self.d_model = position, d_model
        self.pos_encoding = self._build_positional_encoding(position, d_model)
    def get_config(self):
        config = super().get_config(); config.update({'position': self.position, 'd_model': self.d_model}); return config
    def _get_angles(self, pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)); return pos * angle_rates
    def _build_positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]); angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm1, self.layernorm2 = layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)
        self.dropout1, self.dropout2 = layers.Dropout(rate), layers.Dropout(rate)
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs); attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output); ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training); return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.att.key_dim, 'num_heads': self.att.num_heads, 'ff_dim': self.ffn.layers[0].units, 'rate': self.dropout1.rate})
        return config

def build_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0, mlp_dropout=0):
    """Builds a Transformer model architecture."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(head_size)(inputs)
    x = PositionalEncoding(position=input_shape[0], d_model=head_size)(x)
    for i in range(num_transformer_blocks):
        x = TransformerEncoderBlock(embed_dim=head_size, num_heads=num_heads, ff_dim=ff_dim, rate=dropout, name=f'transformer_block_{i+1}')(x)
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x); x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name="Transformer_Model")
    return model

# ==============================================================================
# Module: e_train_models_prompt
# ==============================================================================
def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, training_params: Dict[str, Any]) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Trains a given Keras model."""
    if not all(k in training_params for k in ['epochs', 'batch_size']):
        raise ValueError("`training_params` must contain 'epochs' and 'batch_size'.")
    patience = training_params.get('patience', 10)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    logging.info(f"--- Starting training for {model.name} ---")
    history = model.fit(X_train, y_train, epochs=training_params['epochs'], batch_size=training_params['batch_size'], validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    logging.info(f"--- Finished training for {model.name} ---")
    return model, history

# ==============================================================================
# Utility and Execution Functions
# ==============================================================================
def create_dummy_csv(filepath: str = "NVDA_dummy.csv", num_rows: int = 500):
    if os.path.exists(filepath): return
    logging.info(f"Creating dummy data file at: {filepath}")
    dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=num_rows, freq='B'))
    data = {'Date': dates.strftime('%Y-%m-%d')}
    base_price = np.linspace(100, 800, num=num_rows) + np.random.randn(num_rows).cumsum() * 5
    data['Open'] = (base_price + np.random.uniform(-5, 5, size=num_rows)).round(2)
    data['Close'] = (base_price + np.random.uniform(-5, 5, size=num_rows)).round(2)
    data['High'] = np.maximum(data['Open'], data['Close']) + np.random.uniform(1, 5, size=num_rows)
    data['Low'] = np.minimum(data['Open'], data['Close']) - np.random.uniform(1, 5, size=num_rows)
    pd.DataFrame(data).to_csv(filepath, index=False)

def evaluate_and_visualize(model: tf.keras.Model, history: tf.keras.callbacks.History, datasets: dict):
    """Evaluates the model and generates plots for loss and predictions."""
    model_name = model.name
    X_test, y_test, scaler, test_dates = datasets['X_test'], datasets['y_test'], datasets['scaler'], datasets['test_dates']
    
    # --- Evaluation ---
    logging.info(f"--- Evaluating {model_name} on Test Set ---")
    predictions_scaled = model.predict(X_test)
    num_features = scaler.n_features_in_
    
    # Inverse transform predictions
    dummy_pred = np.zeros((len(predictions_scaled), num_features))
    dummy_pred[:, -1] = predictions_scaled.ravel()
    predictions_actual = scaler.inverse_transform(dummy_pred)[:, -1]
    
    # Inverse transform actuals
    dummy_y = np.zeros((len(y_test), num_features))
    dummy_y[:, -1] = y_test.ravel()
    y_test_actual = scaler.inverse_transform(dummy_y)[:, -1]
    
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    print(f"{model_name} Test RMSE: {rmse:.4f}")
    print(f"{model_name} Test MAE:  {mae:.4f}")

    # --- Loss Visualization ---
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training & Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.legend(); plt.grid(True)
    plt.show()

    # --- Prediction Visualization ---
    if len(test_dates) != len(y_test_actual):
        logging.warning("Test dates and test values length mismatch. Adjusting for plot.")
        test_dates = test_dates[-len(y_test_actual):]
        
    plt.figure(figsize=(15, 7))
    plt.plot(test_dates, y_test_actual, color='blue', label='Actual Price')
    plt.plot(test_dates, predictions_actual, color='red', linestyle='--', label='Predicted Price')
    plt.title(f'NVIDIA Stock Price Prediction ({model_name})'); plt.xlabel('Date'); plt.ylabel('Stock Price (USD)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Time-series forecasting pipeline.")
    dummy_csv_path = "NVDA_dummy.csv"
    create_dummy_csv(filepath=dummy_csv_path)
    parser.add_argument("--filepath", type=str, default=dummy_csv_path, help=f"Path to the input CSV. Defaults to '{dummy_csv_path}'.")
    args = parser.parse_args()

    try:
        logging.info("--- Step 1: Data Loading ---")
        df = load_data_from_csv(args.filepath)

        logging.info("\n--- Step 2: Data Preprocessing ---")
        preprocessor = DataPreprocessor(sequence_length=60, train_split_ratio=0.8, val_split_ratio=0.1)
        datasets = preprocessor.process(df)
        input_shape = datasets['X_train'].shape[1:]
        logging.info(f"Data preprocessed successfully. Input shape for models: {input_shape}")

        logging.info("\n--- Step 3: Model Building and Compilation ---")
        lstm_model = build_lstm_model(input_shape)
        lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        logging.info("LSTM Model built and compiled.")
        lstm_model.summary()

        transformer_model = build_transformer_model(input_shape)
        transformer_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        logging.info("Transformer Model built and compiled.")
        transformer_model.summary()
        
        training_params = {'epochs': 10, 'batch_size': 32, 'patience': 5}

        logging.info("\n--- Step 4: Training LSTM Model ---")
        trained_lstm, lstm_history = train_model(
            lstm_model, datasets['X_train'], datasets['y_train'], 
            datasets['X_val'], datasets['y_val'], training_params
        )

        logging.info("\n--- Step 5: Training Transformer Model ---")
        trained_transformer, transformer_history = train_model(
            transformer_model, datasets['X_train'], datasets['y_train'], 
            datasets['X_val'], datasets['y_val'], training_params
        )

        logging.info("\n--- Step 6: Evaluation and Visualization ---")
        evaluate_and_visualize(trained_lstm, lstm_history, datasets)
        evaluate_and_visualize(trained_transformer, transformer_history, datasets)

    except (FileNotFoundError, ValueError, Exception) as e:
        logging.error(f"An error occurred in the pipeline: {e}", exc_info=True)
        print("\nExecution halted due to an error. Please check the logs.")

if __name__ == '__main__':
    main()