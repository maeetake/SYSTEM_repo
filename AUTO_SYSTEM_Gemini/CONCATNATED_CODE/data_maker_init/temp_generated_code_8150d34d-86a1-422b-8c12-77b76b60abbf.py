# main.py
# This script serves as the executable entry point for the full data preparation and model building pipeline.
# It integrates data loading, preprocessing, dataset splitting, and model architecture definition.

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
import os # Added for file and directory operations

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
            raise ValueError("Not enough data to create sequences with the given sequence length.")

        # Split data into training, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_split_size, shuffle=False
        )
        
        # Adjust validation split size relative to the remaining data
        val_size_adjusted = self.val_split_size / (1 - self.test_split_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, shuffle=False
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
    The architecture is based on the summary provided in the problem description.

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
        
        # FIX: Corrected the variable name from 'fn_output' to 'ffn_output'
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


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the data loading, preprocessing, splitting, and model building pipeline.

    This entry point first loads data, then preprocesses it into sequences,
    splits the sequences into training, validation, and test sets, and finally
    builds the LSTM and Transformer models based on the data shape. It includes
    error handling for all stages and uses a fallback to mock data for the
    loading stage to ensure the script can demonstrate its flow even if the
    primary data source is unavailable.
    """
    # Per "Implementation Guidelines", use a predefined data path.
    # Using a raw string (r'...') to handle backslashes in Windows paths correctly.
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\NVIDIA.csv'
    df = None  # Initialize df to ensure it's in scope

    print("--- Step 1: Data Loading ---")
    try:
        # Attempt to load the primary dataset by calling the imported function
        df = load_data_from_csv(data_path)
        print("\nData loaded successfully. Raw DataFrame head:")
        print(df.head())

    except (FileNotFoundError, ValueError, KeyError) as e:
        # Per "main Function Instructions", handle errors with informative messages
        # and provide mock data as a fallback.
        print(f"\nAn error occurred while loading the primary data: {e}")
        print("---")
        print("Proceeding with mock data for demonstration purposes.")

        # Create a mock DataFrame that mimics the expected structure.
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
        return # Halt if loading fails critically

    # This section proceeds if the 'df' DataFrame was successfully created.
    if df is not None:
        # ==============================================================================
        # START: Added feature to save expected input data
        # ==============================================================================
        try:
            # Determine required columns based on the specification's 'data_acquisition' section.
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
            
            # Verify that the loaded/mocked DataFrame contains the necessary columns.
            if all(col in df.columns for col in required_columns):
                # Hardcode the save directory and filename as per requirements.
                save_dir = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_Gemini\UNITTEST_DATA\generated'
                file_path = os.path.join(save_dir, 'expected_input.csv')
                
                # Create the target directory if it doesn't exist.
                os.makedirs(save_dir, exist_ok=True)
                
                # Select only the columns specified as required for the next module.
                df_to_save = df[required_columns]
                
                # Save the DataFrame to a CSV file, overwriting any existing file.
                # index=False prevents pandas from writing row indices into the CSV.
                df_to_save.to_csv(file_path, index=False)
                print(f"\nSuccessfully saved the expected input data for the next module to: {file_path}")
            else:
                # If the data format is incorrect, do not save the file and print a message.
                missing_cols = [col for col in required_columns if col not in df.columns]
                print(f"\nSkipping save of expected input: The data is missing required columns: {missing_cols}")

        except Exception as e:
            # Handle potential errors during file I/O or directory creation.
            print(f"\nAn error occurred while saving the expected input data: {e}")
        # ==============================================================================
        # END: Added feature
        # ==============================================================================

        final_datasets = None  # Initialize to ensure it's in scope
        # The DataPreprocessor module handles both preprocessing and splitting.
        print("\n--- Step 2: Data Preprocessing and Splitting ---")
        try:
            # Instantiate the preprocessor from the imported module
            preprocessor = DataPreprocessor(sequence_length=60)
            
            # Run the full preprocessing pipeline.
            # The preprocessor returns a dictionary with already-split datasets.
            final_datasets = preprocessor.process(df)
            
            print("\nPreprocessing and splitting complete. Verifying output...")
            
            # Display the shapes and types of the final dataset arrays
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
            final_datasets = None # Ensure it is None on failure

        # --- Step 3: Model Building ---
        # This step proceeds only if preprocessing was successful.
        if final_datasets:
            print("\n--- Step 3: Model Building ---")
            try:
                # Determine the input shape from the preprocessed training data.
                # Shape is (sequence_length, num_features).
                input_shape = final_datasets['X_train'].shape[1:]
                print(f"Determined model input shape: {input_shape}")

                # Build the LSTM model
                print("\nBuilding LSTM Model...")
                lstm_model = build_lstm_model(input_shape)
                print("LSTM Model Summary:")
                lstm_model.summary()

                # Build the Transformer model
                print("\nBuilding Transformer Model...")
                transformer_model = build_transformer_model(input_shape)
                print("Transformer Model Summary:")
                transformer_model.summary()

            except (ValueError, KeyError, Exception) as e:
                # FIX: Convert exception 'e' to a string to prevent UnicodeEncodeError on some terminals.
                print(f"\nAn error occurred during model building: {str(e)}")
                print("Model building halted.")
        else:
            print("\nSkipping model building due to preprocessing failure.")
    
    else:
        # This case occurs if data loading (Step 1) failed and the mock data
        # fallback also failed.
        print("\nData loading failed. Cannot proceed with preprocessing and model building.")


if __name__ == '__main__':
    # This block allows the script to be run directly to execute the main pipeline logic.
    main()