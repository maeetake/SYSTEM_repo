# build_lstm_and_transformer_models.py

# ==============================================================================
# Dependency Specifications
# ==============================================================================
# Requires:
#   - python: 3.8+
#   - tensorflow: 2.10+
#   - numpy: 1.21+
# ==============================================================================

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Transformer Model Helper Classes ---

class PositionalEncoding(layers.Layer):
    """
    Injects positional information into the input tensor.

    This layer adds a positional encoding vector to the input embeddings.
    The encodings are static and not learned. They are calculated using sine and
    cosine functions of different frequencies.

    Formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, position: int, d_model: int, **kwargs):
        """
        Initializes the PositionalEncoding layer.

        Args:
            position (int): The maximum sequence length.
            d_model (int): The dimensionality of the embeddings (and the model).
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        # The calculation of positional encoding is done once and stored.
        self.pos_encoding = self.calculate_positional_encoding(position, d_model)

    def get_config(self):
        """Returns the layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model,
        })
        return config

    def calculate_positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        """
        Calculates the positional encoding matrix.

        Args:
            position (int): Maximum sequence length.
            d_model (int): Embedding dimension.

        Returns:
            tf.Tensor: A tensor of shape (1, position, d_model) with positional encodings.
        """
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        """Calculates the angle rates for the positional encoding formula."""
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            inputs (tf.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            tf.Tensor: The input tensor with added positional information.
        """
        seq_len = tf.shape(inputs)[1]
        # The positional encoding is added to the input embeddings.
        return inputs + self.pos_encoding[:, :seq_len, :]


class TransformerEncoderBlock(layers.Layer):
    """
    A single Transformer Encoder block.

    This block consists of a Multi-Head Self-Attention mechanism followed by a
    Position-wise Feed-Forward Network. Layer normalization and dropout are
    applied after each sub-layer.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1, **kwargs):
        """
        Initializes the TransformerEncoderBlock.

        Args:
            embed_dim (int): The dimensionality of the input and output (d_model).
            num_heads (int): The number of attention heads.
            ff_dim (int): The number of units in the hidden layer of the feed-forward network.
            rate (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        """Returns the layer configuration for serialization."""
        config = super().get_config()
        # Note: embed_dim can be inferred from key_dim in self.att
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "rate": self.dropout1.rate,
        })
        return config

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass for the Transformer Encoder block.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool, optional): Flag indicating if the model is in training mode.
                                     Passed to Dropout layers.

        Returns:
            tf.Tensor: The output tensor from the block.
        """
        # Multi-Head Attention sub-layer
        attn_output = self.att(query=inputs, value=inputs, key=inputs)
        attn_output = self.dropout1(attn_output, training=training)
        # Residual connection and layer normalization
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-Forward Network sub-layer
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection and layer normalization
        return self.layernorm2(out1 + fn_output)

# --- Model Building Functions ---

def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """
    Defines and constructs a sequential model using LSTM layers.

    The architecture consists of two LSTM layers with Dropout for regularization,
    followed by a Dense layer for feature extraction and a final Dense output layer.

    Args:
        input_shape (tuple): A tuple of integers, e.g., (60, 4), representing
                             (sequence_length, num_features).

    Returns:
        tf.keras.Model: An uncompiled tf.keras.Model object representing the
                        LSTM architecture.

    Raises:
        ValueError: If `input_shape` is not a tuple of two positive integers.
    """
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2 and
            all(isinstance(dim, int) and dim > 0 for dim in input_shape)):
        msg = f"Invalid input_shape: {input_shape}. Expected a tuple of two positive integers."
        logging.critical(msg)
        raise ValueError(msg)

    model = tf.keras.Sequential([
        layers.Input(shape=input_shape, name="input_layer"),
        layers.LSTM(units=50, return_sequences=True, name="lstm_1"),
        layers.Dropout(0.2, name="dropout_1"),
        layers.LSTM(units=50, return_sequences=False, name="lstm_2"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(units=25, activation='relu', name="dense_1"),
        layers.Dense(units=1, name="output_layer")  # Single neuron for next-day price prediction
    ], name="LSTM_Model")

    logging.info(f"LSTM model built successfully with input shape {input_shape}.")
    return model

def build_transformer_model(
    input_shape: tuple,
    head_size: int = 256,
    num_heads: int = 4,
    ff_dim: int = 4,
    num_transformer_blocks: int = 4,
    mlp_units: list = [128],
    dropout: float = 0.25,
    mlp_dropout: float = 0.4
) -> tf.keras.Model:
    """
    Defines and constructs a model using a Transformer encoder architecture.

    The model first projects the input features into an embedding space (`head_size`),
    adds positional encoding, and then passes the result through a stack of
    Transformer Encoder Blocks. Finally, it uses Global Average Pooling and a
    Multi-Layer Perceptron (MLP) head to produce the final prediction.

    Args:
        input_shape (tuple): A tuple of integers, e.g., (60, 4), representing
                             (sequence_length, num_features).
        head_size (int): The dimensionality of the model's embeddings.
        num_heads (int): The number of attention heads in the Multi-Head Attention layer.
        ff_dim (int): The dimensionality of the hidden layer in the feed-forward network.
        num_transformer_blocks (int): The number of Transformer Encoder Blocks to stack.
        mlp_units (list): A list of integers specifying the number of units in each
                          Dense layer of the final MLP head.
        dropout (float): The dropout rate for the Transformer Encoder Blocks.
        mlp_dropout (float): The dropout rate for the final MLP layers.

    Returns:
        tf.keras.Model: An uncompiled tf.keras.Model object representing the
                        Transformer architecture.
                        
    Raises:
        ValueError: If `input_shape` is not a tuple of two positive integers.
    """
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2 and
            all(isinstance(dim, int) and dim > 0 for dim in input_shape)):
        msg = f"Invalid input_shape: {input_shape}. Expected a tuple of two positive integers."
        logging.critical(msg)
        raise ValueError(msg)

    sequence_length, num_features = input_shape
    inputs = tf.keras.Input(shape=input_shape, name="input_layer")

    # Project input features to the model's embedding dimension
    x = layers.Dense(head_size, name="feature_projection")(inputs)

    # Add positional information.
    # Note: This is a standard implementation detail. Positional encoding is
    # added to the embeddings to give the model information about sequence order.
    x = PositionalEncoding(position=sequence_length, d_model=head_size)(x)

    # Stack of Transformer Encoder Blocks
    for i in range(num_transformer_blocks):
        x = TransformerEncoderBlock(
            embed_dim=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            rate=dropout,
            name=f"transformer_block_{i+1}"
        )(x)

    # Global Average Pooling to reduce sequence dimension
    x = layers.GlobalAveragePooling1D(data_format="channels_last", name="global_avg_pooling")(x)

    # MLP Head for final prediction
    for i, dim in enumerate(mlp_units):
        x = layers.Dense(dim, activation="relu", name=f"mlp_dense_{i+1}")(x)
        x = layers.Dropout(mlp_dropout, name=f"mlp_dropout_{i+1}")(x)

    # Final output layer
    outputs = layers.Dense(1, name="output_layer")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Transformer_Model")
    
    logging.info(f"Transformer model built successfully with input shape {input_shape}.")
    return model

# Note on main() and load_data():
# As per the module's specified role ("build_lstm_and_transformer_models"),
# this file is responsible only for defining the model architectures.
# Data loading, preprocessing, and model training are handled by other modules
# in the system. Including a main() or data loading functions here would
# violate the Single Responsibility Principle and make the system less modular.
# The functions defined above are intended to be imported and used by a
# separate training script.