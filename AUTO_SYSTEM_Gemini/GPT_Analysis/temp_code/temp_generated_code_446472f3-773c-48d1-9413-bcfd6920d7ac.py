# build_lstm_and_transformer_models.py

"""
Module: build_lstm_and_transformer_models

This module defines two deep learning architectures for next-day NVIDIA stock closing price prediction:
    1. LSTM model via Keras Sequential API
    2. Transformer model via Keras Functional API with custom encoder and positional encoding

No model is compiled in this module. Use the provided architecture building functions before training.

Dependencies:
    - Python >= 3.8
    - tensorflow >= 2.10
    - numpy >= 1.21
    - pandas >= 1.0.0

Specification and rationale:
    - Input features: Only OHLC, shape (sequence_length, 4)
    - Both models output a single real-valued prediction (the next-day closing price, scaled)
    - Error handling: Invalid shapes cause critical halting errors (ValueError)
    - This module does not access the original CSV: it builds and returns models only
    - Data loading utilities and example main(logically present, but commented for clarity)

Author: AUTO_SYSTEM_GPT (2024)
"""

# == External Dependencies ==
# Requires: tensorflow >= 2.10, numpy >= 1.21, pandas >= 1.0.0

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import logging

# ================ Custom Transformer Components ================

class TransformerEncoderBlock(layers.Layer):
    """
    A single Transformer Encoder block using MultiHeadAttention and position-wise feedforward.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        """
        Args:
            embed_dim: output dimension of the dense projection, model dimension
            num_heads: number of attention heads
            ff_dim: Dimensionality of feed-forward network
            rate: Dropout rate
        """
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    """
    Sine & Cosine positional encoding layer (non-trainable).
    """
    def __init__(self, sequence_length: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.embed_dim))
        return pos * angle_rates

    def call(self, x):
        # x: (batch, seq_len, num_features)
        seq_len = self.sequence_length
        d_model = self.embed_dim

        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rads = self.get_angles(pos, i)

        # apply sin to even indices, cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]  # (1, seq_len, d_model)
        pos_encoding = tf.cast(pos_encoding, dtype=x.dtype)
        return x + pos_encoding # shape broadcast

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "embed_dim": self.embed_dim})
        return config

# ================ Model Builder Functions ================

def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """
    Builds an uncompiled LSTM model for time series regression.

    Args:
        input_shape: Tuple (sequence_length, num_features); e.g. (60, 4)

    Returns:
        tf.keras.Model (uncompiled)
    """
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2
            and all(isinstance(dim, int) and dim > 0 for dim in input_shape)):
        logging.critical(f"Invalid input_shape for LSTM: {input_shape!.r}. Must be a tuple of two positive ints.")
        raise ValueError(f"Invalid input_shape: {input_shape}. Expected tuple like (60,4), positive ints.")

    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units=50, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(units=50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(25, activation='relu'),
        layers.Dense(1)  # Output: 1 value (next-day closing price, normalized)
    ])
    return model

def build_transformer_model(
    input_shape: tuple,
    head_size: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_transformer_blocks: int = 2,
    mlp_units: list = [64],
    dropout: float = 0.1,
    mlp_dropout: float = 0.1
    ) -> tf.keras.Model:
    """
    Builds an uncompiled Transformer model for time series regression.

    Args:
        input_shape: Tuple (sequence_length, num_features)
        head_size: output feature size of Dense before MHA and key_dim for attention
        num_heads: Number of attention heads in each encoder block
        ff_dim: Dimension of feedforward hidden in encoder block
        num_transformer_blocks: Number of encoder blocks stacked
        mlp_units: List of int, units in each final Dense MLP layer (excluding output)
        dropout: Dropout rate in encoder block
        mlp_dropout: Dropout rate in MLP layers before output

    Returns:
        tf.keras.Model (uncompiled)
    """
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2
            and all(isinstance(dim, int) and dim > 0 for dim in input_shape)):
        logging.critical(f"Invalid input_shape for Transformer: {input_shape!r}. Must be a tuple of two positive ints.")
        raise ValueError(f"Invalid input_shape: {input_shape}. Expected tuple like (60,4), positive ints.")

    # Model input: (sequence_length, num_features)
    sequence_length, num_features = input_shape
    inputs = tf.keras.Input(shape=(sequence_length, num_features))

    # Project to model dimension
    x = layers.Dense(head_size)(inputs)
    # Add positional encoding
    x = PositionalEncoding(sequence_length, head_size)(x)

    # Stack Transformer Encoder blocks
    for _ in range(num_transformer_blocks):
        x = TransformerEncoderBlock(head_size, num_heads, ff_dim, dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)  # summarize sequence to vector

    for units in mlp_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(1)(x)  # Output: 1 value (next-day closing price)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ================ Optional Data Loader Example =================
# Not required by this module's interface, but included for completeness per guidelines

import pandas as pd
import os

def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads CSV or JSON data from disk using pandas. Used for main() demo.

    Args:
        data_path: Path to file

    Returns:
        pd.DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File not found at {data_path}")
    try:
        if data_path.lower().endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.lower().endswith('.json'):
            return pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or JSON.")
    except Exception as e:
        raise ValueError(f"Error loading data from {data_path}: {e}")

# ================ Main Function Demo ===========================
# Only for demonstration; the actual model builders do not load any data

def main():
    """
    Demo/utility main function for loading data and reporting head of DataFrame.

    This does NOT build the models (as these are to be called directly/externally).
    """
    data_path = r'C:\Users\T25ma\OneDrive\Desktop\AUTO_SYSTEM_GPT\UNITTEST_DATA\generated\expected_input_for_training.csv'
    try:
        df = load_data(data_path)
        print("Data loaded successfully. Head of the data:")
        print(df.head())
    except FileNotFoundError as e:
        print(str(e))
        # Optionally provide mock data
        mock_data = pd.DataFrame({"OHLC1": [0.1, 0.2, 0.3], "OHLC2": [0.2, 0.3, 0.4]})
        print("Using mock data:")
        print(mock_data.head())
    except Exception as e:
        print(f"An error occurred: {e}")

# If run as a script, demo data loading only
if __name__ == "__main__":
    main()

# ================ End of Module ================================