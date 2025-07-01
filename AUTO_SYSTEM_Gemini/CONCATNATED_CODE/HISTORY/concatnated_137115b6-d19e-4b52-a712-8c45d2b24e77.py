from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model
from PACKAGE.e_train_models_prompt import train_model
from PACKAGE.f_evaluate_model_performance_prompt import evaluate_all_models
from PACKAGE.g_visualize_prediction_results_prompt import plot_predictions
from PACKAGE.h_visualize_training_history_prompt import plot_and_save_history
# ==============================================================================
# Integrated Source Code
# ==============================================================================

# main.py
# This script serves as the executable entry point for the full data preparation and model building pipeline.
# It integrates data loading, preprocessing, dataset splitting, model architecture definition, training, evaluation, and visualization.

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
from typing import Tuple, Dict, Any, List
import os
import matplotlib.pyplot as plt

# Import from the conceptual package structure as per specifications
# The function `plot_and_save_history` will be used to replace the local implementation.
# In a real package, the function's code would reside in the specified module file.
def plot_and_save_history(
    model_history: Dict[str, List[float]], 
    model_name: str, 
    output_path: str
) -> str:
    """
    Generates and saves a plot of the training and validation loss for a given
    model's training history. (Functionality integrated from h_visualize_training_history_prompt.py)

    Args:
        model_history (Dict[str, List[float]]): 
            A dictionary containing 'loss' and 'val_loss' lists.
        model_name (str): 
            The name of the model for the plot title.
        output_path (str): 
            The file path where the plot image will be saved.

    Returns:
        str: 
            The absolute path of the saved image file.

    Raises:
        KeyError: If 'loss' or 'val_loss' keys are not present.
        ValueError: If model_name or output_path are empty.
        IOError: If the file cannot be saved.
    """
    if not isinstance(model_history, dict) or 'loss' not in model_history or 'val_loss' not in model_history:
        raise KeyError("The 'model_history' dictionary must contain 'loss' and 'val_loss' keys.")
    if not model_name or not isinstance(model_name, str):
        raise ValueError("The 'model_name' must be a non-empty string.")
    if not output_path or not isinstance(output_path, str):
        raise ValueError("The 'output_path' must be a non-empty string.")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs = range(1, len(model_history['loss']) + 1)
    
    ax.plot(epochs, model_history['loss'], 'o-', label='Training Loss', color='royalblue')
    ax.plot(epochs, model_history['val_loss'], 's-', label='Validation Loss', color='darkorange')
    
    ax.set_title(f'{model_name} Model - Training & Validation Loss', fontsize=16, weight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xticks(epochs) if len(epochs) < 20 else ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='x', rotation=0)
    fig.tight_layout()

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except IOError as e:
        error_msg = f"Failed to save plot to '{output_path}'. Reason: {e}"
        logging.error(error_msg)
        raise IOError(error_msg) from e
    finally:
        plt.close(fig)

    saved_path = os.path.abspath(output_path)
    logging.info(f"Successfully saved training history plot to: {saved_path}")
    return saved_path


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
        # This scaler will be used to scale 'Close' price for accurate inverse transform
        self.target_scaler = MinMaxScaler(feature_range=(0,1))

    def _clean_and_prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and prepares the DataFrame."""
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        feature_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in feature_columns):
             raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns.")

        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=feature_columns)
        
        return df

    def _create_sequences(self, data: np.ndarray, target_data: np.ndarray):
        """Creates sequences and corresponding labels from the data."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target_data[i]) # Target is the scaled 'Close' price
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
        feature_df = prepared_df[['Open', 'High', 'Low', 'Close']]
        
        # Scale all features
        scaled_features = self.scaler.fit_transform(feature_df)
        
        # Scale target variable ('Close') separately for inverse transformation
        scaled_target = self.target_scaler.fit_transform(prepared_df[['Close']])

        X, y = self._create_sequences(scaled_features, scaled_target)
        
        if len(X) == 0:
            raise ValueError(f"Not enough data to create sequences with the given sequence length of {self.sequence_length}. "
                             f"Need at least {self.sequence_length + 1} data points after cleaning, but found {len(scaled_features)}.")

        # Calculate split indices based on the length of sequences (X)
        total_size = len(X)
        test_size = int(total_size * self.test_split_size)
        val_size = int(total_size * self.val_split_size)
        train_size = total_size - test_size - val_size

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
            'scaler': self.scaler, # The multi-feature scaler
            'target_scaler': self.target_scaler, # The single-feature target scaler
            'original_df': prepared_df # Keep the cleaned df for date indexing
        }

# ==============================================================================
# Module: d_build_lstm_and_transformer_models_prompt
# ==============================================================================

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

class TransformerEncoderBlock(layers.Layer):
    """
    Custom Keras layer for a Transformer Encoder block.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
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
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config

def build_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0.1, mlp_dropout=0.1):
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
    
    # Project the input features into the embedding dimension
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
    """
    try:
        predictions_normalized = model.predict(X_test)
        if predictions_normalized.ndim == 1:
            predictions_normalized = predictions_normalized.reshape(-1, 1)
        return predictions_normalized
    except Exception as e:
        logging.error(f"Failed to generate predictions: {e}")
        raise

def inverse_transform_values(
    predictions_normalized: np.ndarray,
    y_test_normalized: np.ndarray,
    target_scaler: MinMaxScaler
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse transform the predicted and actual values back to their original scale
    using the dedicated target scaler.
    """
    if not hasattr(target_scaler, 'n_features_in_') or target_scaler.n_features_in_ != 1:
        raise ValueError("Scaler must be a fitted scikit-learn MinMaxScaler trained on a single feature.")
    
    predictions_actual_scale = target_scaler.inverse_transform(predictions_normalized).ravel()
    y_test_actual_scale = target_scaler.inverse_transform(y_test_normalized).ravel()

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
    target_scaler: MinMaxScaler
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
                target_scaler # Use the target-specific scaler
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
# Module: g_visualize_prediction_results_prompt
# =======================================================================================
def plot_predictions(
    actual_prices: pd.Series,
    predicted_prices_lstm: np.ndarray,
    predicted_prices_transformer: np.ndarray,
    dates: pd.Series,
    output_path: str
) -> str:
    """
    Creates and saves a time-series plot comparing actual test set prices
    with the predicted prices from both the LSTM and Transformer models.

    Args:
        actual_prices (pd.Series): A pandas Series containing the true closing prices.
        predicted_prices_lstm (np.ndarray): An array of predicted prices from the LSTM model.
        predicted_prices_transformer (np.ndarray): An array of predicted prices from the Transformer model.
        dates (pd.Series): A pandas Series of datetime objects for the x-axis.
        output_path (str): The file path where the plot will be saved.

    Returns:
        str: The absolute path to the saved image file.
    """
    if not (len(actual_prices) == len(predicted_prices_lstm) ==
            len(predicted_prices_transformer) == len(dates)):
        msg = (f"Input arrays have mismatched lengths: "
               f"Actuals={len(actual_prices)}, LSTM={len(predicted_prices_lstm)}, "
               f"Transformer={len(predicted_prices_transformer)}, Dates={len(dates)}")
        logging.error(msg)
        raise ValueError(msg)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))

        ax.plot(dates, actual_prices, color='royalblue', label='Actual Price', linewidth=2)
        ax.plot(dates, predicted_prices_lstm, color='darkorange', linestyle='--', label='LSTM Prediction')
        ax.plot(dates, predicted_prices_transformer, color='forestgreen', linestyle='-.', label='Transformer Prediction')

        ax.set_title('Stock Price Prediction: Actual vs. Predicted', fontsize=18, weight='bold')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Closing Price', fontsize=14)
        fig.autofmt_xdate()
        ax.legend(fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Prediction plot successfully saved to: {output_path}")

    except Exception as e:
        err_msg = f"Failed to generate or save plot. Reason: {e}"
        logging.error(err_msg, exc_info=True)
        raise IOError(err_msg) from e
    finally:
        plt.close(fig)

    return os.path.abspath(output_path)

# =======================================================================================
# Helper Functions and Main Execution
# =======================================================================================
def create_dummy_csv(filepath: str = "NVDA_dummy.csv", num_rows: int = 1380):
    """Creates a dummy CSV file for demonstration if it doesn't exist."""
    if os.path.exists(filepath):
        logging.info(f"Dummy data file '{filepath}' already exists. Skipping creation.")
        return

    logging.info(f"Creating dummy data file at: {filepath} for ~5.5 years of business days.")
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
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) + np.random.uniform(0, 2)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) - np.random.uniform(0, 2)
    df = df.round(2)
    df.to_csv(filepath, index=False)

def get_test_dates(original_cleaned_df: pd.DataFrame, final_datasets: dict) -> pd.Series:
    """
    Extracts the correct dates for the test set predictions.
    """
    num_test_samples = len(final_datasets['y_test'])
    
    # The dates for the test set are the last `num_test_samples` dates from the cleaned dataframe
    test_dates = original_cleaned_df['Date'].iloc[-num_test_samples:].reset_index(drop=True)

    # Ensure the length of dates matches the number of test predictions
    if len(test_dates) != num_test_samples:
        logging.warning("Mismatch between number of test dates and test labels. Visualization may be inaccurate.")
    
    return test_dates

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the full pipeline.
    """
    dummy_csv_path = "NVDA_5_5_years_daily.csv"
    create_dummy_csv(filepath=dummy_csv_path, num_rows=int(252*5.5))

    parser = argparse.ArgumentParser(description="Run the full model training and evaluation pipeline.")
    parser.add_argument("--filepath", type=str, default=dummy_csv_path, help=f"Path to input CSV data. Defaults to '{dummy_csv_path}'.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save plots and results.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Data Loading ---
    logging.info("\n--- Step 1: Data Loading ---")
    try:
        df = load_data_from_csv(args.filepath)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error loading data: {e}. Execution halted.")
        return

    # --- Step 2: Data Preprocessing ---
    logging.info("\n--- Step 2: Data Preprocessing and Splitting ---")
    try:
        preprocessor = DataPreprocessor(sequence_length=60, test_split_size=0.1, val_split_size=0.1)
        final_datasets = preprocessor.process(df)
        logging.info("Preprocessing and splitting complete.")
    except (ValueError, Exception) as e:
        logging.error(f"Error during data preprocessing: {e}. Pipeline halted.")
        return

    # --- Step 3: Model Building ---
    logging.info("\n--- Step 3: Model Building ---")
    try:
        input_shape = final_datasets['X_train'].shape[1:]
        logging.info(f"Determined model input shape: {input_shape}")
        
        logging.info("Building LSTM Model...")
        lstm_model = build_lstm_model(input_shape)
        lstm_model.summary(print_fn=logging.info)

        logging.info("Building Transformer Model...")
        transformer_model = build_transformer_model(input_shape)
        transformer_model.summary(print_fn=logging.info)
    except (ValueError, KeyError, Exception) as e:
        logging.error(f"Error during model building: {e}. Pipeline halted.")
        return

    # --- Step 4: Model Training ---
    logging.info("\n--- Step 4: Model Training ---")
    trained_models = {}
    training_histories = {}
    try:
        training_params = {'epochs': 10, 'batch_size': 32, 'patience': 5}
        
        lstm_model, lstm_history = train_model(
            model=lstm_model,
            X_train=final_datasets['X_train'], y_train=final_datasets['y_train'],
            X_val=final_datasets['X_val'], y_val=final_datasets['y_val'],
            training_params=training_params)
        trained_models['LSTM'] = lstm_model
        training_histories['LSTM'] = lstm_history
        
        transformer_model, transformer_history = train_model(
            model=transformer_model,
            X_train=final_datasets['X_train'], y_train=final_datasets['y_train'],
            X_val=final_datasets['X_val'], y_val=final_datasets['y_val'],
            training_params=training_params)
        trained_models['Transformer'] = transformer_model
        training_histories['Transformer'] = transformer_history
        
        logging.info("Both models have been trained successfully.")
    except (ValueError, Exception) as e:
        logging.error(f"Error during model training: {e}. Pipeline halted.")
        return

    # --- Step 5: Model Evaluation ---
    logging.info("\n--- Step 5: Model Evaluation ---")
    evaluation_results = {}
    try:
        evaluation_results = evaluate_all_models(
            models=trained_models,
            X_test=final_datasets['X_test'],
            y_test_normalized=final_datasets['y_test'],
            target_scaler=final_datasets['target_scaler'])
        
        metrics_report = evaluation_results.get('metrics_report', {})
        if metrics_report:
            logging.info("\n--- Performance Metrics (RMSE & MAE) ---\n" + pd.DataFrame.from_dict(metrics_report, orient='index').round(4).to_string())
    except Exception as e:
        logging.error(f"An error occurred during model evaluation: {e}")

    # --- Step 6: Visualization ---
    logging.info("\n--- Step 6: Generating Visualizations ---")
    try:
        # Plot training histories using the integrated function
        for model_name, history in training_histories.items():
            history_plot_path = os.path.join(args.output_dir, f"{model_name}_loss_history.png")
            plot_and_save_history(
                model_history=history.history,
                model_name=model_name,
                output_path=history_plot_path
            )

        # Plot prediction results
        predictions_report = evaluation_results.get('predictions_report', {})
        if predictions_report and 'LSTM' in predictions_report and predictions_report['LSTM'] and 'Transformer' in predictions_report and predictions_report['Transformer']:
            test_dates = get_test_dates(final_datasets['original_df'], final_datasets)
            plot_predictions(
                actual_prices=pd.Series(predictions_report['LSTM']['actuals']),
                predicted_prices_lstm=predictions_report['LSTM']['predictions'],
                predicted_prices_transformer=predictions_report['Transformer']['predictions'],
                dates=test_dates,
                output_path=os.path.join(args.output_dir, "prediction_comparison.png")
            )
        else:
            logging.warning("Could not generate prediction plot. Not all model predictions are available.")
    except Exception as e:
        logging.error(f"An error occurred during visualization: {e}")
    
    logging.info("\nPipeline execution finished.")

if __name__ == '__main__':
    main()