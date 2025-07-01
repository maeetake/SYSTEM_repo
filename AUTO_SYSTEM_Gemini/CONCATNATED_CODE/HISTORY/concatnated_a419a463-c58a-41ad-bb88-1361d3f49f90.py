from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model
from PACKAGE.e_train_models_prompt import train_model
from PACKAGE.f_evaluate_model_performance_prompt import evaluate_all_models

# ==============================================================================
# Integrated Source Code
# ==============================================================================

# main.py
# This script serves as the executable entry point for the full data preparation and model building pipeline.
# It integrates data loading, preprocessing, dataset splitting, model architecture definition, training, and evaluation.

# ==============================================================================
# Dependencies
# ==============================================================================
# Requires: pandas, numpy, scikit-learn, tensorflow
import pandas as pd
import numpy as np
import logging
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any, Union
import os

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
        print(f"Successfully loaded data from: {filepath}")
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
    def __init__(self, sequence_length: int = 60, test_split_size: float = 0.1, val_split_size: float = 0.1):
        """
        Initializes the DataPreprocessor.

        Args:
            sequence_length (int): The number of time steps in each input sequence.
            test_split_size (float): The proportion of the dataset to reserve for the test set.
            val_split_size (float): The proportion of the remaining data to reserve for the validation set.
        """
        self.sequence_length = sequence_length
        self.test_split_size = test_split_size
        self.val_split_size = val_split_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _clean_and_prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and prepares the DataFrame."""
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Select features based on common financial data and the log output
        feature_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in feature_columns):
             raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns.")

        # Ensure numeric types, coercing errors
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=feature_columns)
        
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
            dict: A dictionary containing split datasets ('X_train', 'y_train', etc.) and the scaler.
        """
        prepared_df = self._clean_and_prepare_df(df)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(prepared_df)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        if len(X) == 0:
            raise ValueError(f"Not enough data to create sequences with the given sequence length of {self.sequence_length}. "
                             f"Need at least {self.sequence_length + 1} data points after cleaning, but found {len(scaled_data)}.")

        # Split data chronologically into training (80%), validation (10%), and test (10%)
        train_size = int(len(X) * (1 - self.test_split_size - self.val_split_size))
        val_size = int(len(X) * self.val_split_size)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

        return {
            'X_train': X_train.astype(np.float32),
            'y_train': y_train.astype(np.float32),
            'X_val': X_val.astype(np.float32),
            'y_val': y_val.astype(np.float32),
            'X_test': X_test.astype(np.float32),
            'y_test': y_test.astype(np.float32),
            'scaler': self.scaler
        }

# ==============================================================================
# Module: d_build_lstm_and_transformer_models_prompt
# ==============================================================================

# --- LSTM Model Builder ---
def build_lstm_model(input_shape: tuple) -> keras.Model:
    """
    Builds and compiles an LSTM model based on the provided input shape.

    Args:
        input_shape (tuple): The shape of the input data (sequence_length, num_features).

    Returns:
        keras.Model: The compiled Keras LSTM model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape, name='input_layer'),
        layers.LSTM(50, return_sequences=True, name='lstm_1'),
        layers.Dropout(0.2, name='dropout_1'),
        layers.LSTM(50, return_sequences=False, name='lstm_2'),
        layers.Dropout(0.2, name='dropout_2'),
        layers.Dense(25, activation='relu', name='dense_1'),
        layers.Dense(1, name='output_layer')
    ], name="LSTM_Model")
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Transformer Model Components and Builder ---
class TransformerEncoderBlock(layers.Layer):
    """
    Custom Keras layer for a Transformer Encoder block.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.att.key_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate,
        })
        return config

def build_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0, mlp_dropout=0):
    """
    Builds and compiles a Transformer-based model for time series forecasting.

    Args:
        input_shape (tuple): The shape of the input data (sequence_length, num_features).
        head_size (int): The embedding dimension for the transformer blocks.
        num_heads (int): The number of attention heads.
        ff_dim (int): The dimension of the feed-forward network.
        num_transformer_blocks (int): The number of transformer blocks to stack.
        mlp_units (list): A list of integers for the dimensions of the final MLP layers.
        dropout (float): Dropout rate for the transformer blocks.
        mlp_dropout (float): Dropout rate for the final MLP layers.

    Returns:
        keras.Model: The compiled Keras Transformer model.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Project input features to the transformer's embedding dimension
    x = layers.Dense(head_size)(x)
    
    for i in range(num_transformer_blocks):
        x = TransformerEncoderBlock(
            embed_dim=head_size, num_heads=num_heads, ff_dim=ff_dim, rate=dropout, name=f'transformer_block_{i+1}'
        )(x)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs, name="Transformer_Model")
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# =======================================================================================
# Module: e_train_models_prompt
# =======================================================================================
def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_params: Dict[str, Any]
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains a given Keras model on the provided training and validation datasets.

    Args:
        model (tf.keras.Model): The compiled, un-trained Keras model.
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data targets.
        X_val (np.ndarray): Validation data features.
        y_val (np.ndarray): Validation data targets.
        training_params (Dict[str, Any]): Dictionary with 'epochs', 'batch_size', 'patience'.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: A tuple containing the
            trained model and its training history.
    """
    required_params = ['epochs', 'batch_size']
    for param in required_params:
        if param not in training_params:
            raise ValueError(f"Missing required key in `training_params`: '{param}'")

    patience = training_params.get('patience', 10)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    model_name = getattr(model, 'name', 'Unnamed Model')
    logging.info(f"--- Starting training for {model_name} ---")
    logging.info(f"Hyperparameters: Epochs={training_params['epochs']}, "
                 f"Batch Size={training_params['batch_size']}, "
                 f"EarlyStopping Patience={patience}")

    history = model.fit(
        X_train,
        y_train,
        epochs=training_params['epochs'],
        batch_size=training_params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    logging.info(f"--- Finished training for {model_name} ---")
    
    return model, history

# =======================================================================================
# Module: f_evaluate_model_performance_prompt
# =======================================================================================

def make_predictions(
    model: Any,
    X_test: np.ndarray
) -> np.ndarray:
    """
    Generate predictions for the test set using a trained model.

    Args:
        model: A trained model object (TensorFlow/Keras or PyTorch).
        X_test: A numpy.ndarray of shape (n_samples, sequence_length, n_features)
                representing the test features.

    Returns:
        A numpy.ndarray of shape (n_samples, 1) containing the model's
        predictions in the normalized scale [0, 1].
    """
    try:
        if hasattr(model, 'predict'):
            predictions_normalized = model.predict(X_test)
        else:
            raise AttributeError("Model object does not have a 'predict' method.")

        if predictions_normalized.ndim == 1:
            predictions_normalized = predictions_normalized.reshape(-1, 1)
        return predictions_normalized
    except Exception as e:
        logging.error(f"Failed to generate predictions: {e}")
        raise

def inverse_transform_values(
    predictions_normalized: np.ndarray,
    y_test_normalized: np.ndarray,
    scaler: MinMaxScaler
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse transform the predicted and actual values back to their original scale.

    Args:
        predictions_normalized: A numpy.ndarray of normalized predictions.
        y_test_normalized: A numpy.ndarray of the true normalized values.
        scaler: A fitted scikit-learn MinMaxScaler object.
    """
    if not hasattr(scaler, 'n_features_in_'):
        raise ValueError("Scaler must be a fitted scikit-learn MinMaxScaler.")
    
    num_features = scaler.n_features_in_
    dummy_predictions = np.zeros((len(predictions_normalized), num_features))
    dummy_actuals = np.zeros((len(y_test_normalized), num_features))

    dummy_predictions[:, -1] = predictions_normalized.ravel()
    dummy_actuals[:, -1] = y_test_normalized.ravel()

    predictions_actual_scale = scaler.inverse_transform(dummy_predictions)[:, -1]
    y_test_actual_scale = scaler.inverse_transform(dummy_actuals)[:, -1]

    return predictions_actual_scale, y_test_actual_scale

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': float(rmse), 'MAE': float(mae)}

def evaluate_all_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test_normalized: np.ndarray,
    scaler: MinMaxScaler
) -> Dict[str, Dict]:
    """
    Evaluates a dictionary of trained models on the test data and aggregates results.
    """
    metrics_report = {}
    predictions_report = {}

    for model_name, model in models.items():
        logging.info(f"Evaluating model: {model_name}")
        try:
            predictions_normalized = make_predictions(model, X_test)
            predictions_actual, y_test_actual = inverse_transform_values(
                predictions_normalized,
                y_test_normalized,
                scaler
            )
            metrics = calculate_metrics(y_test_actual, predictions_actual)
            logging.info(f"Metrics for {model_name}: {metrics}")

            metrics_report[model_name] = metrics
            predictions_report[model_name] = {
                'predictions': predictions_actual,
                'actuals': y_test_actual
            }
        except Exception as e:
            logging.error(f"Could not evaluate model '{model_name}'. Reason: {e}", exc_info=True)
            metrics_report[model_name] = {'error': str(e)}
            predictions_report[model_name] = None

    return {
        'metrics_report': metrics_report,
        'predictions_report': predictions_report
    }


# =======================================================================================
# Helper Functions and Main Execution
# =======================================================================================
def create_dummy_csv(filepath: str = "NVDA_dummy.csv", num_rows: int = 1380):
    """Creates a dummy CSV file for demonstration if it doesn't exist."""
    if os.path.exists(filepath):
        print(f"Dummy data file '{filepath}' already exists. Skipping creation.")
        return

    print(f"Creating dummy data file at: {filepath} for ~5.5 years of business days.")
    dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=num_rows, freq='B'))
    base_price = 200 + np.cumsum(np.random.normal(0, 2, size=num_rows))
    data = {
        'Date': dates.strftime('%Y-%m-%d'),
        'Open': base_price + np.random.normal(0, 5, size=num_rows),
        'Close': base_price + np.random.normal(0, 5, size=num_rows),
        'High': np.maximum(base_price + 10, base_price + np.random.normal(10, 5, size=num_rows)),
        'Low': np.minimum(base_price - 10, base_price + np.random.normal(-10, 5, size=num_rows)),
    }
    df = pd.DataFrame(data)
    # Ensure High is highest and Low is lowest
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) + np.random.uniform(0, 2)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) - np.random.uniform(0, 2)
    df = df.round(2)
    df.to_csv(filepath, index=False)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the data loading, preprocessing, model building, training, and evaluation pipeline.
    """
    dummy_csv_path = "NVDA_5_5_years_daily.csv"
    create_dummy_csv(filepath=dummy_csv_path, num_rows=int(252*5.5)) # Approx 5.5 years of trading days

    parser = argparse.ArgumentParser(description="Run the full model training and evaluation pipeline.")
    parser.add_argument("--filepath", type=str, default=dummy_csv_path, help=f"Path to the input CSV data. Defaults to '{dummy_csv_path}'.")
    args = parser.parse_args()

    # --- Step 1: Data Loading ---
    print("\n--- Step 1: Data Loading ---")
    try:
        df = load_data_from_csv(args.filepath)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError loading data: {e}. Execution halted.")
        return

    # --- Step 2: Data Preprocessing ---
    print("\n--- Step 2: Data Preprocessing and Splitting ---")
    try:
        preprocessor = DataPreprocessor(sequence_length=60, test_split_size=0.1, val_split_size=0.1)
        final_datasets = preprocessor.process(df)
        print("Preprocessing and splitting complete.")
        for key, value in final_datasets.items():
            if isinstance(value, np.ndarray):
                print(f"Key: {key:<10} | Shape: {str(value.shape):<20} | DType: {value.dtype}")
    except (ValueError, Exception) as e:
        print(f"\nError during data preprocessing: {e}. Pipeline halted.")
        return

    # --- Step 3: Model Building ---
    print("\n--- Step 3: Model Building ---")
    try:
        input_shape = final_datasets['X_train'].shape[1:]
        print(f"Determined model input shape: {input_shape}")
        
        print("\nBuilding LSTM Model...")
        lstm_model = build_lstm_model(input_shape)
        lstm_model.summary()

        print("\nBuilding Transformer Model...")
        transformer_model = build_transformer_model(input_shape)
        transformer_model.summary()
    except (ValueError, KeyError, Exception) as e:
        print(f"\nError during model building: {e}. Pipeline halted.")
        return

    # --- Step 4: Model Training ---
    print("\n--- Step 4: Model Training ---")
    trained_models = {}
    try:
        training_params = {'epochs': 10, 'batch_size': 32, 'patience': 5}
        
        lstm_model, lstm_history = train_model(
            model=lstm_model,
            X_train=final_datasets['X_train'], y_train=final_datasets['y_train'],
            X_val=final_datasets['X_val'], y_val=final_datasets['y_val'],
            training_params=training_params
        )
        trained_models['LSTM'] = lstm_model
        
        transformer_model, transformer_history = train_model(
            model=transformer_model,
            X_train=final_datasets['X_train'], y_train=final_datasets['y_train'],
            X_val=final_datasets['X_val'], y_val=final_datasets['y_val'],
            training_params=training_params
        )
        trained_models['Transformer'] = transformer_model
        
        print("\nBoth models have been trained successfully.")
    except (ValueError, Exception) as e:
        print(f"\nError during model training: {e}. Pipeline halted.")
        return

    # --- Step 5: Model Evaluation ---
    print("\n--- Step 5: Model Evaluation ---")
    try:
        evaluation_results = evaluate_all_models(
            models=trained_models,
            X_test=final_datasets['X_test'],
            y_test_normalized=final_datasets['y_test'],
            scaler=final_datasets['scaler']
        )
        
        print("\n" + "="*50)
        print("           MODEL EVALUATION REPORT")
        print("="*50)
        
        metrics_report = evaluation_results.get('metrics_report', {})
        if metrics_report:
            print("\n--- Performance Metrics (RMSE & MAE) ---")
            metrics_df = pd.DataFrame.from_dict(metrics_report, orient='index')
            print(metrics_df.round(4))
        else:
            print("No metrics were generated.")
            
        print("\n" + "="*50)

    except Exception as e:
        print(f"\nAn error occurred during model evaluation: {e}")
    
    print("\nPipeline execution finished.")


if __name__ == '__main__':
    main()