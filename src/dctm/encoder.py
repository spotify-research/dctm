#
# Copyright 2020 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Module for encoding class and functions, using keras.Sequential and Dense layers."""
from typing import List
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as tfkl

tfd = tfp.distributions
tfpl = tfp.layers


def variational_net(
    layer_sizes: List[Union[str, int]] = (300, 300, 300),
    activation: str = "relu",
    dtype: str = "float64",
):
    """Variational network. A sequential dense network.

    Args:
        layer_sizes (List[Union[str, int]], optional): Size of layers. Defaults to (300, 300, 300).
        activation (str, optional): Activation function. Defaults to "relu".
        dtype (str, optional): Data type. Defaults to "float64".

    Returns:
        tf.keras.Model: Sequential dense network.
    """
    layers = [
        tfkl.Dense(
            units,
            activation=activation,
            kernel_initializer="glorot_normal",
            dtype=dtype,
        )
        for units in layer_sizes
    ]
    return tf.keras.Sequential(layers)


class MeanScaleEncoder(tf.keras.Model):
    def __init__(
        self,
        output_dim: int,
        layer_sizes: List[Union[str, int]] = (300, 300, 300),
        output_bias_scale: int = -3,
        jitter: float = 1e-8,
        kernel_initializer: str = "glorot_normal",
        activation: str = "relu",
        dtype: str = "float64",
        name: str = "MeanScaleEncoder",
    ):
        """Variational encoder for mean and variance of a normal distribution.

        They share the hidden part of the network. 
        Then, the mean and variance have a different Dense layer.

        Args:
            output_dim (int): Output dimensionality.
            layer_sizes (List[Union[str, int]], optional): Size of layers. Defaults to (300, 300, 300).
            output_bias_scale (int, optional): Bias initializer of the scale. Defaults to -3.
            jitter (float, optional): Jitter for scale. Defaults to 1e-8.
            kernel_initializer (str, optional): Kernel initializer. Defaults to "glorot_normal".
            activation (str, optional): Activation function. Defaults to "relu".
            dtype (str, optional): Data type. Defaults to "float64".
            name (str, optional): Name of this class. Defaults to "MeanScaleEncoder".

        """
        super(MeanScaleEncoder, self).__init__(name, dtype=dtype)

        self.jitter = jitter
        self.hidden_layers = variational_net(
            layer_sizes=layer_sizes, activation=activation, dtype=dtype
        )
        self.dense_mean = tf.keras.layers.Dense(
            output_dim,
            activation=None,
            kernel_initializer=kernel_initializer,
            dtype=dtype,
            name="dense_mean",
        )
        self.dense_scale = tf.keras.layers.Dense(
            output_dim,
            # the bias initializer for scale ensures that the variational scale
            # starts from a small value.
            bias_initializer=tf.keras.initializers.Constant(output_bias_scale),
            activation="softplus",
            kernel_initializer=kernel_initializer,
            dtype=dtype,
            name="dense_scale",
        )

    def call(self, inputs):
        h = self.hidden_layers(inputs)
        return (self.dense_mean(h), self.dense_scale(h) + self.jitter)


class VariationalEncoderDistribution(tf.keras.Model):
    def __init__(
        self,
        encoder,
        make_distribution_fn,
        dtype="float64",
        name="VariationalEncoderDistribution",
    ):
        super(VariationalEncoderDistribution, self).__init__(name, dtype=dtype)
        self.encoder = encoder
        self.distribution_lambda = tfpl.DistributionLambda(
            make_distribution_fn=make_distribution_fn, dtype=dtype
        )

    def call(self, inputs):
        outputs = self.encoder(inputs)
        return self.distribution_lambda(outputs)
