# Revised Integrated Script
# main.py
# This script serves as the executable entry point for the full data preparation and model building pipeline.
# It orchestrates the pipeline by importing and executing modules for data loading, preprocessing, and model building.

# ==============================================================================
# Dependencies
# ==============================================================================
# Standard library imports
import logging

# Third-party imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# Local application/library specific imports
# The following functions and classes are assumed to be in their respective
# modules within a 'PACKAGE' directory, as per the modular design.
# The code for these modules is based on the original integrated script and feedback.

# from PACKAGE.a_load_user_provided_data_prompt import load_data_from_csv
# from PACKAGE.c_split_dataset_prompt import split_sequential_data
# from PACKAGE.b_preprocess_data_prompt import DataPreprocessor
# from PACKAGE.d_build_lstm_and_transformer_models_prompt import build_lstm_model, build_transformer_model

# For standalone execution, the original definitions are included below.
# In a real package structure, these would be in separate files and imported.

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
# Module: c_split_dataset_prompt
# ==============================================================================
def split_sequential_data(X: np.ndarray, y: np.ndarray, test_size: float, val_size: float) -> tuple:
    """
    Splits sequential data chronologically into training, validation, and test sets.
    The split sizes are based on the proportions of the original total dataset.

    Args:
        X (np.ndarray): The feature data.
        y (np.ndarray): The target data.
        test_size (float): The proportion of the dataset to reserve for testing (e.g., 0.1 for 10%).
        val_size (float): The proportion of the dataset to reserve for validation (e.g., 0.1 for 10%).

    Returns:
        tuple: A tuple containing (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    if not (0 < test_size < 1 and 0 < val_size < 1 and (test_size + val_size) < 1):
        raise ValueError("test_size and val_size must be floats between 0 and 1, and their sum must be less than 1.")
    
    total_samples = len(X)
    test_split_index = int(total_samples * (1 - test_size))
    val_split_index = int(total_samples * (1 - test_size - val_size))

    X_train, y_train = X[:val_split_index], y[:val_split_index]
    X_val, y_val = X[val_split_index:test_split_index], y[val_split_index:test_split_index]
    X_test, y_test = X[test_split_index:], y[test_split_index:]

    return X_train, y_train, X_val, y_val, X_test, y_test

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
            val_split_size (float): The proportion of the dataset to reserve for the validation set.
        """
        self.sequence_length = sequence_length
        self.test_split_size = test_split_size
        self.val_split_size = val_split_size
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
        
        # Apply forward fill for missing values
        df[feature_columns] = df[feature_columns].ffill()
        
        # Drop any remaining NaN rows (likely at the beginning of the dataset)
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

        # Use the dedicated splitting function for improved modularity
        X_train, y_train, X_val, y_val, X_test, y_test = split_sequential_data(
            X, y, test_size=self.test_split_size, val_size=self.val_split_size
        )
        
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
class PositionalEncoding(layers.Layer):
    """
    Custom Keras layer for Positional Encoding.
    This layer injects information about the relative or absolute position of the
    tokens in the sequence. The positional encodings have the same dimension as
    the embeddings so that the two can be summed.
    """
    def __init__(self, position: int, d_model: int, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
        })
        return config

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerEncoderBlock(layers.Layer):
    """Custom Keras layer for a Transformer Encoder block."""
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
    
    # Project features to the embedding dimension
    x = layers.Dense(head_size)(x)
    
    # Add positional encoding
    x = PositionalEncoding(position=input_shape[0], d_model=head_size)(x)
    
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


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the data loading, preprocessing, splitting,
    and model building pipeline.
    """
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\NVIDIA.csv'
    df = None

    print("--- Step 1: Data Loading ---")
    try:
        df = load_data_from_csv(data_path)
        print("\nData loaded successfully. Raw DataFrame head:")
        print(df.head())

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred while loading the primary data: {e}")
        print("---")
        print("Proceeding with mock data for demonstration purposes.")

        mock_data = {
            'Date': pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 91)]),
            'Open': np.linspace(150, 200, 90),
            'High': np.linspace(152, 205, 90),
            'Low': np.linspace(148, 198, 90),
            'Close': np.linspace(151, 202, 90),
            'Volume': np.linspace(1000000, 1500000, 90, dtype=int)
        }
        df = pd.DataFrame(mock_data)

        print("\nMock data generated. DataFrame head:")
        print(df.head())

    except Exception as e:
        print(f"\nAn unexpected critical error occurred during data loading: {e}")
        print("Execution halted.")
        return

    if df is not None:
        final_datasets = None
        print("\n--- Step 2: Data Preprocessing and Splitting ---")
        try:
            # Using 80/10/10 split
            preprocessor = DataPreprocessor(sequence_length=60, test_split_size=0.1, val_split_size=0.1)
            final_datasets = preprocessor.process(df)
            
            print("\nPreprocessing and splitting complete. Verifying output...")
            
            print("\n" + "="*20 + " Final Datasets Output " + "="*20)
            for key, value in final_datasets.items():
                if isinstance(value, np.ndarray):
                    print(f"Key: {key:<10} | Shape: {str(value.shape):<20} | DType: {value.dtype}")
                else:
                    print(f"Key: {key:<10} | Type: {type(value)}")
            print("="*64)

        except (ValueError, Exception) as e:
            print(f"\nAn error occurred during data preprocessing and splitting: {e}")
            print("Pipeline halted.")
            final_datasets = None

        if final_datasets:
            print("\n--- Step 3: Model Building ---")
            try:
                input_shape = final_datasets['X_train'].shape[1:]
                print(f"Determined model input shape: {input_shape}")

                print("\nBuilding LSTM Model...")
                lstm_model = build_lstm_model(input_shape)
                print("LSTM Model Summary:")
                lstm_model.summary()

                print("\nBuilding Transformer Model...")
                transformer_model = build_transformer_model(input_shape)
                print("Transformer Model Summary:")
                transformer_model.summary()

            except (ValueError, KeyError, Exception) as e:
                print(f"\nAn error occurred during model building: {str(e)}")
                print("Model building halted.")
        else:
            print("\nSkipping model building due to preprocessing failure.")
    
    else:
        print("\nData loading failed. Cannot proceed with preprocessing and model building.")


if __name__ == '__main__':
    main()