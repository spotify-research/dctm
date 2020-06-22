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
"""
Latent Dirichlet Allocation with relaxation.

The distribution of topic proportion, instead of being a Dirichlet,
is a Logistic Normal distribution (with diagonal covariance matrix).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import metrics as skm

import encoder

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers


class RelaxedLDA(tf.Module):
    """Latent Dirichlet Allocation with relaxation.
    
    Eta (topics distribution) is drawn from a Normal distribution with diagonal covariance matrix.
    """

    def __init__(
        self,
        n_words: int,
        n_topics: int = 3,
        prior_eta_loc: Union[None, tf.Variable, tf.Tensor] = None,
        prior_eta_scale_diag: Union[None, tf.Variable, tf.Tensor] = None,
        topics_words_logits: Union[None, tf.Variable, tf.Tensor] = None,
        layer_sizes: List[Union[int, str]] = (300, 300, 300),
        activation: str = "relu",
        dtype: Union[np.dtype, tf.DType, str] = np.float64,
        jitter: float = 1e-5,
        validate_args: bool = False,
        allow_nan_stats: bool = False,
        name: str = "RelaxedLDA",
    ):
        """LDA model with relaxation. Topic distribution follows a logistic normal distribution with diagonal covariance matrix.

        Args:
            n_words (int): Dimensionality of the dataset.
            n_topics (int, optional): Number of topics. Defaults to 3.
            prior_eta_loc (Union[None, tf.Variable, tf.Tensor], optional): Mean prior for eta. Defaults to None.
            prior_eta_scale_diag (Union[None, tf.Variable, tf.Tensor], optional): Scale prior for eta. Defaults to None.
            topics_words_logits (Union[None, tf.Variable, tf.Tensor], optional): Topic words logits. Defaults to None.
            layer_sizes (List[Union[int, str]], optional): Layer sizes for the creation of
                the neural network parametrising mu and Sigma
                for eta posterior. Defaults to (300, 300, 300).
            activation (str, optional): Neural net activation. Defaults to "relu".
            dtype (Union[np.dtype, tf.DType, str], optional): Data type of the variables. Defaults to np.float64.
            jitter (float, optional): Jitter for encoder. Defaults to 1e-5.
            validate_args (bool, optional): Validate the arguments of distribution classes. Defaults to False.
            allow_nan_stats (bool, optional): Allow nan stats for the arguments of distribution classes. Defaults to False.
            name (str, optional): Class name. Defaults to "RelaxedLDA".

        """
        super(RelaxedLDA, self).__init__(name=name)
        self.dtype = dtype
        self.validate_args = validate_args

        if topics_words_logits is None:
            # topics_words_logits = tf.ones(
            #     [n_topics, n_words], dtype=dtype) / n_words
            topics_words_logits = tf.math.softmax(
                np.random.standard_normal([n_topics, n_words]).astype(dtype=dtype)
            )
        self.topics_words_logits = tfp.util.TransformedVariable(
            topics_words_logits,
            bijector=tfb.SoftmaxCentered(),
            name="topics_words_logits",
        )

        if prior_eta_loc is None:
            prior_eta_loc = tf.zeros([1, n_topics], dtype=dtype)

        if prior_eta_scale_diag is None:
            prior_eta_scale_diag = tf.zeros([1, n_topics], dtype=dtype) + 1e-4

        self.prior_eta = tfd.MultivariateNormalDiag(
            loc=prior_eta_loc,
            scale_diag=prior_eta_scale_diag,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="topics_prior",
        )

        variational_posterior_distribution_fn = lambda t: tfd.MultivariateNormalDiag(  # noqa
            loc=t[0],
            scale_diag=t[1],
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="topics_posterior",
        )
        # posterior_eta = encoder.VariationalEncoderDistribution(
        #     encoder=encoder.MeanScaleEncoder(
        #         output_dim=n_topics, layer_sizes=layer_sizes
        #     ),
        #     make_distribution_fn=variational_posterior_distribution_fn,
        #     dtype=dtype,
        # )
        # posterior_eta.build((1, n_words))

        inputs = tf.keras.Input((n_words,))
        outputs = encoder.MeanScaleEncoder(
            output_dim=n_topics, layer_sizes=layer_sizes, jitter=jitter, dtype=dtype
        )(inputs)
        outputs = tfpl.DistributionLambda(
            make_distribution_fn=variational_posterior_distribution_fn, dtype="float64",
        )(outputs)
        posterior_eta = tf.keras.Model(inputs, outputs)

        self.surrogate_posterior_eta = posterior_eta

    @property
    def _kl_guard(self):
        """Ensure KL divergence is always positive (with a small slack)."""
        return tf.cast(-1e-3, self.dtype)

    def predict(self, X):
        """Return topics associated to documents."""
        posterior_eta = self.surrogate_posterior_eta(X)
        return tf.math.softmax(posterior_eta.loc)

    def score(self, X, topics):
        """Score function using V measure (takes into account shuffling)."""
        return skm.v_measure_score(
            np.argmax(topics, axis=1), np.argmax(self.predict(X), axis=1)
        )

    def expectation(self, X, eta_samples, beta_samples):
        r"""log p(X | eta, beta), where (eta, beta) \sim q(eta, beta)."""
        word_probs = tf.matmul(eta_samples, beta_samples)
        return tfd.OneHotCategorical(probs=word_probs, name="bag_of_words").log_prob(X)

    @tf.Module.with_name_scope
    @tf.function
    def elbo(
        self,
        X: tf.Tensor,
        observation_index_points: tf.Tensor = None,
        kl_weight: float = 1.0,
        sample_size: int = 1,
    ) -> tf.Tensor:
        """ELBO, computed as E_q[p(X|eta)] - KL[q(eta)||p(eta|mu,scale)].

        Args:
            X (tf.Tensor): Data.
            observation_index_points (tf.Tensor, optional): Unused.
            kl_weight (float, optional): Unused.

        Returns:
            tf.Tensor: elbo for each sample.
        """
        # del observation_index_points, kl_weight  # unused

        posterior_eta = self.surrogate_posterior_eta(X)
        eta_samples = posterior_eta.sample(sample_size)
        eta_samples = tf.math.softmax(eta_samples)

        reconstruction = tf.reduce_mean(
            self.expectation(
                X=X, eta_samples=eta_samples, beta_samples=self.topics_words_logits
            ),
            0,
        )
        # reconstruction = tfp.monte_carlo.expectation(
        #     f=lambda eta: self.expectation(X, eta, self.topics_words_logits),
        #     samples=eta_samples,
        #     log_prob=posterior_eta.log_prob,
        #     axis=0,
        #     use_reparameterization=True,
        # )

        kl = tfd.kl_divergence(posterior_eta, self.prior_eta)
        # with tf.control_dependencies(
        #     [tf1.assert_greater(kl, self._kl_guard, message="kl")]
        # ):
        #     kl = tf.identity(kl)
        elbo = reconstruction - kl

        # actual ELBO would be the sum of this.
        # We return the single parts not merged together.
        return elbo

    def loss(
        self,
        X: tf.Tensor,
        observation_index_points: tf.Tensor = None,
        kl_weight: float = 1.0,
        sample_size: int = 1,
    ) -> float:
        """Loss function. Computed as the negative average ELBO.

        Args:
            X (tf.Tensor): Input data.
            observation_index_points (tf.Tensor, optional): Location of the input.
                Defaults to None if not used.
            kl_weight (float, optional): Weight for KL divergence.
                Useful for minibatch training. Defaults to 1.

        Returns:
            float: negative average ELBO.
        """
        elbo = self.elbo(
            X,
            observation_index_points=observation_index_points,
            kl_weight=kl_weight,
            sample_size=sample_size,
        )
        avg_elbo = tf.reduce_mean(input_tensor=elbo)
        return -avg_elbo

    def perplexity(self, X: tf.Tensor, elbo: tf.Tensor) -> float:
        """Average perplexity of the samples in input.

        Args:
            X (tf.Tensor): Input data.
            elbo: ELBO for each sample.

        Returns:
            float: average perplexity of the input data.
        """
        # The perplexity is an exponent of the average negative ELBO per word.
        words_per_document = tf.reduce_sum(X, axis=-1)
        log_perplexity = -elbo / words_per_document
        perplexity = tf.exp(tf.reduce_mean(log_perplexity))
        return perplexity

    def loss_perplexity(
        self, X, observation_index_points=None, kl_weight=1.0, sample_size=1
    ):
        elbo = self.elbo(
            X,
            observation_index_points=observation_index_points,
            kl_weight=kl_weight,
            sample_size=sample_size,
        )
        perplexity = self.perplexity(X, elbo=elbo)
        return -tf.reduce_mean(elbo, 0), perplexity

    @tf.function
    def batch_optimize(
        self,
        x_batch,
        optimizer,
        observation_index_points=None,
        trainable_variables=None,
        kl_weight: float = 1.0,
        sample_size: int = 1,
    ):
        with tf.GradientTape(
            watch_accessed_variables=trainable_variables is None
        ) as tape:
            for v in trainable_variables or []:
                tape.watch(v)
            loss_value, perpl = self.loss_perplexity(
                x_batch,
                observation_index_points=observation_index_points,
                kl_weight=kl_weight,
                sample_size=sample_size,
            )

        watched_variables = tape.watched_variables()
        grads = tape.gradient(loss_value, watched_variables)
        optimizer.apply_gradients(zip(grads, watched_variables))
        return loss_value, perpl

    def optimize(
        self,
        observations,
        observation_index_points=None,
        n_iter=200,
        learning_rate=0.1,
        batch_size=None,
        n_burnin=50,
        sample_size: int = 1,
    ):
        from tqdm.notebook import tqdm

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # optimizer.iterations = tf1.train.get_or_create_global_step()
        n_training_points = observations.shape[-2]
        if batch_size is None:
            # assuming to have times in first dimension
            batch_size = n_training_points

        # using tf utils for batching
        dataset = tf.data.Dataset.from_tensor_slices(observations)
        dataset = dataset.batch(batch_size)

        pbar = tqdm(range(n_iter), disable=False)
        losses = []
        perplexities = []
        trainable_variables = None  # self.trainable_variables
        for i in pbar:
            loss_value = 0
            perplexity_value = 0
            for x_batch in dataset:
                loss, perpl = self.batch_optimize(
                    x_batch,
                    optimizer=optimizer,
                    observation_index_points=observation_index_points,
                    trainable_variables=trainable_variables,
                    kl_weight=tf.constant(float(batch_size) / float(n_training_points)),
                    sample_size=tf.constant(sample_size),
                )
                loss_value += loss
                perplexity_value += perpl

            losses.append(loss_value)
            perplexities.append(perplexity_value)
            pbar.set_description(
                "loss {:.3e}, perpl {:.3e}".format(loss_value, perplexity_value)
            )

        self.loss_values_ = losses
        self.perplexities_ = perplexities

