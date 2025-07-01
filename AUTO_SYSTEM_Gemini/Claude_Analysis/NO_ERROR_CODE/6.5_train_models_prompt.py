# Requires: tensorflow >= 2.10, numpy >= 1.21
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransformerEncoderBlock(layers.Layer):
    """
    Implements a single block of the Transformer encoder.

    This block consists of a multi-head self-attention mechanism followed by a
    position-wise feed-forward network. Layer normalization and dropout are
    applied after each sub-layer.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        """
        Initializes the TransformerEncoderBlock.

        Args:
            embed_dim (int): The dimensionality of the input and output.
            num_heads (int): The number of attention heads.
            ff_dim (int): The dimensionality of the hidden layer in the feed-forward network.
            rate (float): The dropout rate.
        """
        super(TransformerEncoderBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        """
        Forward pass for the TransformerEncoderBlock.

        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool): Flag indicating whether the model is in training mode.

        Returns:
            tf.Tensor: The output tensor from the block.
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(layers.Layer):
    """
    Injects positional information into the input tensor.

    This layer adds a positional encoding vector to the input embeddings.
    The encodings are fixed and not learned.
    """
    def __init__(self, position, d_model):
        """
        Initializes the PositionalEncoding layer.

        Args:
            position (int): The maximum sequence length.
            d_model (int): The dimensionality of the embeddings (and the model).
        """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        """Calculates the angle rates for the positional encoding formula."""
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        """Creates the positional encoding matrix."""
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
        """Adds positional encoding to the input tensor."""
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """
    Builds an LSTM model architecture.

    This function defines a sequential model using LSTM layers for time-series prediction.

    Args:
        input_shape (tuple): A tuple specifying the input shape (sequence_length, num_features).

    Returns:
        tf.keras.Model: An uncompiled tf.keras.Model for the LSTM.

    Raises:
        ValueError: If `input_shape` is not a tuple of two positive integers.
    """
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2 and all(isinstance(dim, int) and dim > 0 for dim in input_shape)):
        logger.critical(f"Invalid input_shape: {input_shape}. Expected a tuple of two positive integers.")
        raise ValueError(f"Invalid input_shape: {input_shape}. Expected a tuple of two positive integers.")

    model = tf.keras.Sequential([
        layers.Input(shape=input_shape, name="input_layer"),
        layers.LSTM(units=50, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(units=50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(units=25, activation='relu'),
        layers.Dense(units=1, name="output_layer")
    ], name="LSTM_Model")
    logger.info("Successfully built LSTM model.")
    return model


def build_transformer_model(input_shape: tuple, head_size: int = 256, num_heads: int = 4, ff_dim: int = 4,
                            num_transformer_blocks: int = 4, mlp_units: list = [128], dropout: float = 0.25,
                            mlp_dropout: float = 0.4) -> tf.keras.Model:
    """
    Builds a Transformer model architecture.

    This function defines a model using a stack of Transformer encoder blocks.

    Args:
        input_shape (tuple): A tuple specifying the input shape (sequence_length, num_features).
        head_size (int): Dimensionality of the model's attention heads and embeddings.
        num_heads (int): Number of attention heads.
        ff_dim (int): Hidden layer size in the feed-forward network inside the transformer.
        num_transformer_blocks (int): Number of encoder blocks to stack.
        mlp_units (list): A list of integers for the final MLP layer sizes.
        dropout (float): Dropout rate for the transformer blocks.
        mlp_dropout (float): Dropout rate for the final MLP layers.

    Returns:
        tf.keras.Model: An uncompiled tf.keras.Model for the Transformer.

    Raises:
        ValueError: If `input_shape` is not a tuple of two positive integers.
    """
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2 and all(isinstance(dim, int) and dim > 0 for dim in input_shape)):
        logger.critical(f"Invalid input_shape: {input_shape}. Expected a tuple of two positive integers.")
        raise ValueError(f"Invalid input_shape: {input_shape}. Expected a tuple of two positive integers.")

    inputs = tf.keras.Input(shape=input_shape)
    
    # Project input features to the model's embedding dimension
    x = layers.Dense(head_size)(inputs)
    
    # Add positional encoding
    x = PositionalEncoding(input_shape[0], head_size)(x)
    x = layers.Dropout(dropout)(x)

    # Stack Transformer Encoder blocks
    for _ in range(num_transformer_blocks):
        x = TransformerEncoderBlock(head_size, num_heads, ff_dim, dropout)(x)

    # Global Average Pooling to reduce sequence to a single vector
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    
    # Final MLP head
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Transformer_Model")
    logger.info("Successfully built Transformer model.")
    return model