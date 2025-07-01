from typing import Tuple, Dict, Any
# The following imports are simulated based on the project structure.
# In a real package, these would be actual import statements.
# from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
# from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
# from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model
# from PACKAGE.e_train_models_prompt import train_model

# ==============================================================================
# Dependencies
# ==============================================================================
# Requires: pandas, numpy, scikit-learn, tensorflow
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
            val_split_size (float): The proportion of the training data (after test split) to reserve for validation.
        """
        self.sequence_length = sequence_length
        # Correct calculation for chronological split
        self.train_split_size = 1.0 - test_split_size - val_split_size 
        self.val_split_size = val_split_size
        self.test_split_size = test_split_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _clean_and_prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and prepares the DataFrame."""
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        feature_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in feature_columns):
             raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns.")

        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values using forward fill
        df = df.ffill()
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
        
        scaled_data = self.scaler.fit_transform(prepared_df)
        
        X, y = self._create_sequences(scaled_data)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences with the given sequence length.")

        # Chronological split
        train_len = int(len(X) * (1 - self.val_split_size - self.test_split_size))
        val_len = int(len(X) * self.val_split_size)
        
        X_train, y_train = X[:train_len], y[:train_len]
        X_val, y_val = X[train_len:train_len + val_len], y[train_len:train_len + val_len]
        X_test, y_test = X[train_len + val_len:], y[train_len + val_len:]
        
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
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
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
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    return model

# ==============================================================================
# Module: e_train_models_prompt
# ==============================================================================
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
        training_params (Dict[str, Any]): Dictionary of training hyperparameters.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: A tuple containing the
        trained model and the training history.
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


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the full pipeline: data loading, preprocessing,
    model building, and model training.
    """
    data_path = 'NVDA.csv' # Use a relative path for portability
    df = None

    print("--- Step 1: Data Loading ---")
    try:
        df = load_data_from_csv(data_path)
        print("\nData loaded successfully. Raw DataFrame head:")
        print(df.head())
    except (FileNotFoundError, ValueError) as e:
        print(f"\nAn error occurred while loading the data: {e}")
        print("---")
        print("Proceeding with mock data for demonstration purposes.")
        mock_data = {
            'Date': pd.to_datetime([f'2020-01-{i:02d}' for i in range(1, 31)] * 4),
            'Open': np.linspace(150, 250, 120),
            'High': np.linspace(152, 255, 120),
            'Low': np.linspace(148, 248, 120),
            'Close': np.linspace(151, 252, 120),
        }
        df = pd.DataFrame(mock_data)
        print("\nMock data generated. DataFrame head:")
        print(df.head())
    except Exception as e:
        print(f"\nAn unexpected critical error occurred during data loading: {e}")
        print("Execution halted.")
        return

    final_datasets = None
    if df is not None:
        print("\n--- Step 2: Data Preprocessing and Splitting ---")
        try:
            # Using 80% train, 10% validation, 10% test split
            preprocessor = DataPreprocessor(sequence_length=60, val_split_size=0.1, test_split_size=0.1)
            final_datasets = preprocessor.process(df)
            print("\nPreprocessing and splitting complete. Verifying output...")
            print("\n" + "="*20 + " Final Datasets Output " + "="*20)
            for key, value in final_datasets.items():
                if isinstance(value, np.ndarray):
                    print(f"Key: {key:<10} | Shape: {str(value.shape):<20} | DType: {value.dtype}")
            print("="*64)
        except (ValueError, Exception) as e:
            print(f"\nAn error occurred during data preprocessing: {e}")
            print("Pipeline halted.")

    lstm_model = None
    transformer_model = None
    if final_datasets:
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
            print(f"\nAn error occurred during model building: {str(e)}")
            print("Model building halted.")
    else:
        print("\nSkipping model building due to preprocessing failure.")

    if lstm_model and transformer_model:
        print("\n--- Step 4: Model Training ---")
        try:
            training_params = {
                'epochs': 50,
                'batch_size': 32,
                'patience': 10
            }

            # Train LSTM Model
            trained_lstm_model, lstm_history = train_model(
                model=lstm_model,
                X_train=final_datasets['X_train'],
                y_train=final_datasets['y_train'],
                X_val=final_datasets['X_val'],
                y_val=final_datasets['y_val'],
                training_params=training_params
            )
            print("LSTM Model training complete.")
            print(f"Final validation loss for LSTM: {lstm_history.history['val_loss'][-1]:.4f}")


            # Train Transformer Model
            trained_transformer_model, transformer_history = train_model(
                model=transformer_model,
                X_train=final_datasets['X_train'],
                y_train=final_datasets['y_train'],
                X_val=final_datasets['X_val'],
                y_val=final_datasets['y_val'],
                training_params=training_params
            )
            print("Transformer Model training complete.")
            print(f"Final validation loss for Transformer: {transformer_history.history['val_loss'][-1]:.4f}")

        except (ValueError, Exception) as e:
            print(f"\nAn error occurred during model training: {str(e)}")
            print("Model training halted.")
    else:
        print("\nSkipping model training due to previous errors.")
        
    print("\n--- Pipeline Finished ---")


if __name__ == '__main__':
    main()