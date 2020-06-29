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
"""Dynamic Correlated Topic Model.

References
[1] Heaukulani, Creighton, and Mark van der Wilk.
"Scalable Bayesian dynamic covariance modeling with variational Wishart
and inverse Wishart processes."
Advances in Neural Information Processing Systems. 2019.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from dctm import correlated_topic_model
from dctm import variational_wishart_process as vwp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


class DCTMBeta(correlated_topic_model.CTM):
    """A correlated topic model, where the topic-word probability evolves over time.
    
    In other words, it is a simpler dynamic correlated topic model, where the
    topic distribution is static, and the topic-word probability is dynamic.
    """

    def __init__(
        self,
        kernel_beta: tfk.PositiveSemidefiniteKernel,
        index_points_beta: tf.Tensor,
        inducing_index_points_beta: tf.Tensor,
        n_words: int,
        n_topics: int = 3,
        observation_noise_variance_beta: Union[float, tf.Variable] = 1e-2,
        m_beta: Union[None, tf.Variable] = None,
        sigma_beta: Union[None, tf.Variable] = None,
        prior_mu_loc: Union[None, tf.Variable, tf.Tensor] = None,
        prior_mu_scale_diag: Union[None, tf.Variable, tf.Tensor] = None,
        m_mu: Union[None, tf.Variable] = None,
        sigma_mu: Union[None, tf.Variable] = None,
        prior_ell_loc: Union[None, tf.Variable, tf.Tensor] = None,
        prior_ell_scale_diag: Union[None, tf.Variable, tf.Tensor] = None,
        m_ell: Union[None, tf.Variable] = None,
        sigma_ell: Union[None, tf.Variable] = None,
        layer_sizes: List[Union[int, str]] = (300, 300, 300),
        activation: str = "relu",
        dtype: Union[np.dtype, tf.DType, str] = np.float64,
        jitter: float = 1e-5,
        encoder_jitter: float = 1e-8,
        validate_args: bool = False,
        allow_nan_stats: bool = False,
        name: str = "DCTMBeta",
    ):
        """Dynamic correlated topic model, where beta ~ GP().

        mu and ell are not dynamic.

        Args:
            kernel_beta (tfk.PositiveSemidefiniteKernel): Kernel for beta.
            index_points_beta (tf.Tensor): Index points of beta.
            inducing_index_points_beta (tf.Tensor): Index points of the
                inducing variables for beta.
            n_words (int): Dimensionality of the dataset.
            n_topics (int, optional): Number of topics to consider. Defaults to 3.
            observation_noise_variance_beta (Union[float, tf.Variable], optional): Observation noise variance. Defaults to 0.1.
            m_beta (Union[None, tf.Variable], optional): Mean posterior for beta. Defaults to None.
            sigma_beta (Union[None, tf.Variable], optional): Scale posterior for beta. Defaults to None.
            prior_mu_loc (Union[None, tf.Variable], optional): Mean prior for mu. Defaults to None.
            prior_mu_scale_diag (Union[None, tf.Variable], optional): Scale prior for mu. Defaults to None.
            m_mu (Union[None, tf.Variable], optional): Mean posterior for mu. Defaults to None.
            sigma_mu (Union[None, tf.Variable], optional): Scale posterior for beta. Defaults to None.
            prior_ell_loc (Union[None, tf.Variable], optional): Mean prior for ell. Defaults to None.
            prior_ell_scale_diag (Union[None, tf.Variable], optional): Scale prior for ell. Defaults to None.
            m_ell (Union[None, tf.Variable], optional): Mean posterior for ell. Defaults to None.
            sigma_ell (Union[None, tf.Variable], optional): Scale posterior for ell. Defaults to None.
            layer_sizes (List[Union[int, str]], optional): Layer sizes for the creation of
                the neural network parametrising mu and Sigma
                for eta posterior. Defaults to (300, 300, 300).
            activation (str, optional): Neural net activation. Defaults to "relu".
            dtype (Union[np.dtype, tf.DType, str], optional): Data type of the variables. Defaults to np.float64.
            jitter (float, optional): Jitter for Cholesky decompositions. Defaults to 1e-5.
            encoder_jitter (float, optional): Jitter for encoder. Defaults to 1e-8.
            validate_args (bool, optional): Validate the arguments of distribution classes. Defaults to False.
            allow_nan_stats (bool, optional): Allow nan stats of distribution classes. Defaults to False.
            name (str, optional): Class name. Defaults to "DCTMBeta".
        """
        super(DCTMBeta, self).__init__(
            n_topics=n_topics,
            n_words=n_words,
            prior_mu_loc=prior_mu_loc,
            prior_mu_scale_diag=prior_mu_scale_diag,
            m_mu=m_mu,
            sigma_mu=sigma_mu,
            prior_ell_loc=prior_ell_loc,
            prior_ell_scale_diag=prior_ell_scale_diag,
            m_ell=m_ell,
            sigma_ell=sigma_ell,
            layer_sizes=layer_sizes,
            activation=activation,
            jitter=encoder_jitter,
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )
        # These are replaced by the tfd.VariationalGaussianProcess
        # We don't need explicit prior for beta
        del self.prior_beta, self.surrogate_posterior_beta

        n_inducing_points_beta = inducing_index_points_beta.shape[-2]

        if m_beta is None:
            # m_beta = np.random.standard_normal(
            #     [n_words, n_topics] + [n_inducing_points_beta]).astype(dtype=dtype)
            m_beta = (
                np.zeros([n_words, n_topics] + [n_inducing_points_beta], dtype=dtype)
                + np.random.standard_normal([n_words, n_topics]).astype(dtype=dtype)[
                    :, :, None
                ]
            )
        m_beta = tf.Variable(m_beta, name="m_beta")

        if sigma_beta is None:
            # sigma_beta = (
            #     tf.zeros(
            #         # comment to share sigma_beta
            #         # [n_topics] +
            #         [n_inducing_points_beta],
            #         dtype=dtype,
            #     )
            #     + 1e-4
            # )
            # shared across topics
            sigma_beta = np.eye(n_inducing_points_beta, dtype=dtype) * 1e-4
        sigma_beta = tfp.util.TransformedVariable(
            sigma_beta,
            # bijector=tfb.Chain([tfb.Shift(tf.cast(1e-6, dtype)), tfb.Softplus()]),
            bijector=tfb.FillScaleTriL(diag_shift=tf.cast(1e-6, dtype)),
            name="sigma_beta",
        )

        observation_noise_variance_beta = tfp.util.TransformedVariable(
            observation_noise_variance_beta,
            bijector=tfb.Softplus(),
            dtype=dtype,
            name="observation_noise_variance_beta",
        )

        self.surrogate_posterior_beta = tfd.VariationalGaussianProcess(
            kernel=kernel_beta,
            index_points=index_points_beta,
            inducing_index_points=inducing_index_points_beta,
            variational_inducing_observations_loc=m_beta,
            variational_inducing_observations_scale=sigma_beta,
            observation_noise_variance=observation_noise_variance_beta,
            jitter=jitter,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="posterior_beta",
        )

    @tf.Module.with_name_scope
    # @tf.function
    def elbo(
        self,
        X: tf.Tensor,
        observation_index_points: tf.Tensor = None,
        kl_weight: float = 1.0,
        sample_size: int = 1,
    ) -> tf.Tensor:
        """Elbo for each sample.

        Args:
            X (tf.Tensor): Input data. Should be at least 3-dimensional,
                with shape [b1,...,bN, 1, f].
            index_points (tf.Tensor): Location corresponding to samples,
                with shape [b1,...,bN, 1].
            kl_weight (float, optional): Weight for KL divergence.
                Useful for minibatch training. Defaults to 1.

        Returns:
            tf.Tensor: ELBO of each sample.
        """
        posterior_mu = self.surrogate_posterior_mu
        mu_sample = posterior_mu.sample(sample_size)

        posterior_ell = self.surrogate_posterior_ell
        ell_sample = posterior_ell.sample(sample_size)
        scale_tril = self.bijector_ell(ell_sample)

        prior_eta = tfd.MultivariateNormalTriL(
            loc=tf.expand_dims(mu_sample, -2),  # this is to ease kl_divergence code
            scale_tril=tf.expand_dims(scale_tril, -3),
            validate_args=self.validate_args,
            allow_nan_stats=False,
            name="prior_eta",
        )

        posterior_eta = self.surrogate_posterior_eta(X)
        eta_samples = posterior_eta.sample(sample_size)
        eta_samples = tf.nn.softmax(eta_samples)

        beta_samples = self.surrogate_posterior_beta.get_marginal_distribution(
            index_points=observation_index_points
        ).sample(sample_size)

        beta_samples = tfb.Transpose(rightmost_transposed_ndims=3)(beta_samples)
        beta_samples = tf.nn.softmax(beta_samples)

        reconstruction = tf.reduce_mean(
            self.expectation(X=X, eta_samples=eta_samples, beta_samples=beta_samples), 0
        )

        # this is for each sample
        kl = tf.reduce_mean(tfd.kl_divergence(posterior_eta, prior_eta), 0)

        # here we sum everything
        # reconstruction = tf.reduce_sum(reconstruction)
        # kl = tf.reduce_sum(kl, axis=[-1, -2])

        # checks = [tf.assert_greater(kl, self._kl_guard, message="kl")]
        # with tf.control_dependencies(checks):
        #     kl = tf.identity(kl)

        kl_mu = tfd.kl_divergence(posterior_mu, self.prior_mu)
        kl_ell = tfd.kl_divergence(posterior_ell, self.prior_ell)

        # this is equivalent to flattening the parameters and then computing kl
        kl_beta = tf.reduce_sum(
            self.surrogate_posterior_beta.surrogate_posterior_kl_divergence_prior(),
            axis=[-1, -2],
        )

        # if we sum everything this needs to be 1.
        norm_factor = np.prod(X.shape[:-1])
        kl_global = (kl_mu + kl_ell + kl_beta) / norm_factor

        # for the minibatch case, we need to reweight the prior
        # for the total number of samples in the dataset
        elbo = reconstruction - kl - kl_weight * kl_global
        return elbo


class DCTM(DCTMBeta):
    r"""Dynamic correlated topic model.

    In this model, beta ~ GP(0, k_beta), mu ~ GP(0, k_mu),
    ell ~ WP(nu, k_ell).
    """

    def __init__(
        self,
        kernel_beta: tfk.PositiveSemidefiniteKernel,
        kernel_mu: tfk.PositiveSemidefiniteKernel,
        kernel_ell: tfk.PositiveSemidefiniteKernel,
        index_points_beta: tf.Tensor,
        index_points_mu: tf.Tensor,
        index_points_ell: tf.Tensor,
        inducing_index_points_beta: tf.Tensor,
        inducing_index_points_mu: tf.Tensor,
        inducing_index_points_ell: tf.Tensor,
        n_words: int,
        n_topics: int = 3,
        observation_noise_variance_beta: Union[float, tf.Variable] = 1e-2,
        m_beta: Union[None, tf.Variable] = None,
        sigma_beta: Union[None, tf.Variable] = None,
        observation_noise_variance_mu: Union[float, tf.Variable] = 1e-2,
        m_mu: Union[None, tf.Variable] = None,
        sigma_mu: Union[None, tf.Variable] = None,
        observation_noise_variance_ell: Union[float, tf.Variable] = 1e-2,
        m_ell: Union[None, tf.Variable] = None,
        sigma_ell: Union[None, tf.Variable] = None,
        nu: Union[int, None] = None,
        prior_lower_cholesky_loc: Union[None, tf.Variable, tf.Tensor] = None,
        prior_lower_cholesky_scale_diag: Union[None, tf.Variable, tf.Tensor] = None,
        m_lower_wishart: Union[None, tf.Variable] = None,
        sigma_lower_wishart: Union[None, tf.Variable] = None,
        layer_sizes: List[Union[int, str]] = (300, 300, 300),
        activation: str = "relu",
        jitter_beta: float = 1e-5,
        jitter_mu: float = 1e-5,
        jitter_ell: float = 1e-5,
        encoder_jitter: float = 1e-8,
        white_noise_jitter: float = 1e-5,
        dtype: Union[np.dtype, tf.DType, str] = np.float64,
        validate_args: bool = False,
        allow_nan_stats: bool = False,
        name: str = "DCTM",
    ):
        """Dynamic correlated topic model.

        Args:
            kernel_beta (tfk.PositiveSemidefiniteKernel): Kernel for beta.
            kernel_mu (tfk.PositiveSemidefiniteKernel): Kernel for mu.
            kernel_ell (tfk.PositiveSemidefiniteKernel): Kernel for ell.
            index_points_beta (tf.Tensor): Index points of beta.
            index_points_mu (tf.Tensor): Index points of mu.
            index_points_ell (tf.Tensor): Index points of ell.
            inducing_index_points_beta (tf.Tensor): Index points of the
                inducing variables for beta.
            inducing_index_points_mu (tf.Tensor): Index points of the
                inducing variables for mu.
            inducing_index_points_ell (tf.Tensor): Index points of the
                inducing variables for ell.
            n_words (int): Dimensionality of the dataset.
            n_topics (int, optional): Number of topics to consider. Defaults to 3.
            observation_noise_variance_beta (Union[float, tf.Variable], optional): Observation noise variance. Defaults to 1e-2.
            m_beta (tf.Tensor, optional): Mean posterior for beta. Defaults to None.
            sigma_beta (tf.Tensor, optional): Scale posterior for beta. Defaults to None.
            observation_noise_variance_mu (Union[float, tf.Variable], optional): Observation noise variance. Defaults to 1e-2.
            m_mu (tf.Tensor, optional): Mean posterior for mu. Defaults to None.
            sigma_mu (tf.Tensor, optional): Scale posterior for mu. Defaults to None.
            observation_noise_variance_ell (Union[float, tf.Variable], optional): Observation noise variance. Defaults to 1e-2.
            m_ell (tf.Tensor, optional): Mean posterior for ell. Defaults to None.
            sigma_ell (tf.Tensor, optional): Scale posterior for ell. Defaults to None.
            nu (int, optional): Degrees of freedom for Wishart lower Cholesky matrix. Defaults to None (evaluated to n_topics + 1).
            prior_lower_cholesky_loc (Union[None, tf.Variable], optional): Mean prior for lower Cholesky. Defaults to None.
            prior_lower_cholesky_scale_diag (Union[None, tf.Variable], optional): Scale prior for lower Cholesky. Defaults to None.
            m_lower_wishart (Union[None, tf.Variable], optional): Mean posterior for lower Cholesky. Defaults to None.
            sigma_lower_wishart (Union[None, tf.Variable], optional): ScaleMean posterior for lower Cholesky. Defaults to None.
            layer_sizes (tuple, optional): Layer sizes for the creation of
                the neural network parametrising mu and Sigma
                for eta posterior. Defaults to (300, 300, 300).
            activation (str, optional): Neural net activation. Defaults to "relu".
            jitter (float, optional): Jitter for Cholesky decompositions. Defaults to 1e-5.
            encoder_jitter (float, optional): Jitter for encoder. Defaults to 1e-8.
            white_noise_jitter (float, optional): White noise jitter for Wishart process (see [1]). Defaults to 1e-5.
            dtype (Union[np.dtype, tf.dtype, str], optional): Data type of the variables. Defaults to np.float64.
            validate_args (bool, optional): Validate the arguments of distribution classes. Defaults to False.
            allow_nan_stats (bool, optional): Allow nan stats in the arguments of distribution classes. Defaults to False.
            name (str, optional): Class name. Defaults to "DCTM".
        
        """
        super(DCTM, self).__init__(
            kernel_beta=kernel_beta,
            index_points_beta=index_points_beta,
            inducing_index_points_beta=inducing_index_points_beta,
            n_topics=n_topics,
            n_words=n_words,
            observation_noise_variance_beta=observation_noise_variance_beta,
            m_beta=m_beta,
            sigma_beta=sigma_beta,
            layer_sizes=layer_sizes,
            activation=activation,
            jitter=jitter_beta,
            encoder_jitter=encoder_jitter,
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )
        # These are replaced by tfd.VariationalGaussianProcess
        # We don't need explicit prior for mu and ell
        del self.prior_mu, self.surrogate_posterior_mu
        del self.prior_ell, self.surrogate_posterior_ell

        if nu is None:
            nu = n_topics + 1

        n_inducing_points_mu = inducing_index_points_mu.shape[-2]
        if m_mu is None:
            m_mu = np.zeros([n_topics] + [n_inducing_points_mu], dtype=dtype)
        m_mu = tf.Variable(m_mu, name="m_mu")

        if sigma_mu is None:
            # sigma_mu = (
            #     tf.zeros(
            #         # comment to share sigma_mu
            #         # [n_topics] +
            #         [n_inducing_points_mu], dtype=dtype) + 1e-4 )
            # shared across topics
            sigma_mu = np.eye(n_inducing_points_mu, dtype=dtype) * 1e-4
        sigma_mu = tfp.util.TransformedVariable(
            sigma_mu,
            bijector=tfb.FillScaleTriL(diag_shift=tf.cast(1e-6, dtype)),
            name="sigma_mu",
        )

        observation_noise_variance_mu = tfp.util.TransformedVariable(
            observation_noise_variance_mu,
            bijector=tfb.Softplus(),
            dtype=dtype,
            name="observation_noise_variance_mu",
        )

        # Here `ell` is the Sigma of the paper.
        # TODO: To better align to the paper, we can probably change the name
        # to Sigma (but right now we indicated with `Sigma` the covariance
        # of the posterior distributions).
        n_inducing_points_ell = inducing_index_points_ell.shape[-2]
        if m_ell is None:
            # m_ell = np.random.standard_normal(
            #     [nu, n_topics] + [n_inducing_points]).astype(dtype=dtype)
            m_ell = np.zeros((nu, n_topics, n_inducing_points_ell), dtype=dtype)
            m_ell[:n_topics] = np.expand_dims(np.eye(n_topics, dtype=dtype), -1)
        m_ell = tf.Variable(m_ell, name="m_ell")

        if sigma_ell is None:
            sigma_ell = np.eye(n_inducing_points_ell, dtype=dtype) * 1e-4
        sigma_ell = tfp.util.TransformedVariable(
            sigma_ell,
            bijector=tfb.FillScaleTriL(diag_shift=tf.cast(1e-6, dtype)),
            name="sigma_ell",
        )

        observation_noise_variance_ell = tfp.util.TransformedVariable(
            observation_noise_variance_ell,
            bijector=tfb.Softplus(),
            dtype=dtype,
            name="observation_noise_variance_ell",
        )

        self.surrogate_posterior_mu = tfd.VariationalGaussianProcess(
            kernel=kernel_mu,
            index_points=index_points_mu,
            inducing_index_points=inducing_index_points_mu,
            variational_inducing_observations_loc=m_mu,
            variational_inducing_observations_scale=sigma_mu,
            observation_noise_variance=observation_noise_variance_mu,
            jitter=jitter_mu,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="posterior_mu",
        )

        self.surrogate_posterior_ell = vwp.VariationalWishartProcessFullBayesian(
            kernel=kernel_ell,
            index_points=index_points_ell,
            inducing_index_points=inducing_index_points_ell,
            variational_inducing_observations_loc=m_ell,
            variational_inducing_observations_scale=sigma_ell,
            observation_noise_variance=observation_noise_variance_ell,
            jitter=jitter_ell,
            white_noise_jitter=white_noise_jitter,
            prior_lower_cholesky_loc=prior_lower_cholesky_loc,
            prior_lower_cholesky_scale_diag=prior_lower_cholesky_scale_diag,
            m_lower_wishart=m_lower_wishart,
            sigma_lower_wishart=sigma_lower_wishart,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="posterior_ell",
        )

    @tf.Module.with_name_scope
    @tf.function
    def elbo(
        self,
        X: tf.Tensor,
        observation_index_points: tf.Tensor,
        kl_weight: float = 1.0,
        sample_size: int = 1,
    ) -> tf.Tensor:
        """Elbo for each sample.

        Args:
            X (tf.Tensor): Input data. Should be at least 3-dimensional.
            kl_weight (float, optional): Weight for KL divergence.
                Useful for minibatch training. Defaults to 1.

        Returns:
            tf.Tensor: ELBO of each sample.
        """
        # Need to put n_topics as last dimension.
        mu_sample = tfb.Transpose(rightmost_transposed_ndims=2).forward(
            self.surrogate_posterior_mu.get_marginal_distribution(
                index_points=observation_index_points
            ).sample(sample_size)
        )
        scale_tril = self.surrogate_posterior_ell.sample(
            sample_size, index_points=observation_index_points
        )
        prior_eta = tfd.MultivariateNormalTriL(
            loc=tf.expand_dims(mu_sample, -2),  # this is to ease kl_divergence code
            scale_tril=tf.expand_dims(scale_tril, -3),
            validate_args=self.validate_args,
            allow_nan_stats=False,
            name="prior_eta",
        )

        posterior_eta = self.surrogate_posterior_eta(X)
        eta_samples = posterior_eta.sample(sample_size)
        eta_samples = tf.math.softmax(eta_samples, axis=-1)

        beta_samples = self.surrogate_posterior_beta.get_marginal_distribution(
            index_points=observation_index_points
        ).sample(sample_size)

        beta_samples = tfb.Transpose(rightmost_transposed_ndims=3)(beta_samples)
        # Words are now in the last dimension, so the softmax is
        # correctly normalizing the last dim by default.
        beta_samples = tf.nn.softmax(beta_samples, axis=-1)

        reconstruction = tf.reduce_mean(
            self.expectation(X=X, eta_samples=eta_samples, beta_samples=beta_samples), 0
        )
        kl = tf.reduce_mean(tfd.kl_divergence(posterior_eta, prior_eta), 0)

        # here we sum everything
        # kl = tf.reduce_sum(kl, [0, 1])
        # reconstruction = tf.reduce_sum(reconstruction)
        # checks = [tf.assert_greater(kl, self._kl_guard, message="kl")]
        # with tf.control_dependencies(checks):
        #     kl = tf.identity(kl)

        kl_mu = tf.reduce_sum(
            self.surrogate_posterior_mu.surrogate_posterior_kl_divergence_prior(),
            axis=[-1],
        )
        kl_ell = self.surrogate_posterior_ell.surrogate_posterior_kl_divergence_prior()

        # this is equivalent to flattening the parameters and then computing kl
        kl_beta = tf.reduce_sum(
            self.surrogate_posterior_beta.surrogate_posterior_kl_divergence_prior(),
            axis=[-1, -2],
        )

        # if we sum everything this needs to be 1.
        norm_factor = np.prod(X.shape[:-1])
        kl_global = (kl_mu + kl_ell + kl_beta) / norm_factor

        # for the minibatch case, we need to reweight the prior
        # for the total number of samples in the dataset
        elbo = reconstruction - kl - kl_weight * kl_global
        return elbo


def top_words(beta, vocab, top_n=10):
    """Top words in the vocabulary.

    Args:
        beta (np.array): Posterior of beta parameter for a single topic.
        vocab (np.array): List of words in the vocabulary.
        top_n (int, optional): Number of top words to return. Defaults to 10.

    Returns:
        list: Top words.
    """
    # account for multiple times -> in this case returns
    # the most common (unique) words across time
    # beta is for a single topic
    dd = tf.reshape(tf.tile(tf.expand_dims(vocab, -1), [1, beta.shape[-1]]), [-1])
    idx = tf.argsort(tf.reshape(beta, [-1]))[::-1].numpy()

    dd = iter(dd.numpy()[idx])
    twords = []
    while len(twords) < top_n:
        x = next(dd).decode("utf8")
        if x not in twords:
            twords.append(x)
    return twords


def print_topics(
    mdl,
    index_points,
    vocabulary,
    top_n_topic=10,
    top_n_time=5,
    inverse_transform_fn=None,
):
    """Print topics extracted by mdl.

    Args:
        mdl (tf.Module): Topic model.
        index_points (tf.Tensor): Index points.
        vocabulary (list or pd.Series): Vocabulary.
        top_n_topic (int, optional): Top n topics to show across times. Defaults to 10.
        top_n_time (int, optional): Top n topics to show at each time. Defaults to 5.
        inverse_transform_fn (func, optional): Function to map index_points
            to human readable. Defaults to None.

    Returns:
        list: Topics as found by the model.
    """
    # words associated with topics
    # scaler needed to print readable index points
    times_display = times = np.unique(index_points)
    if inverse_transform_fn is not None:
        times_display = inverse_transform_fn(times[:, None])
    topics = []
    words_topic = mdl.surrogate_posterior_beta.get_marginal_distribution(
        index_points=times[:, None]
    ).mean()
    words_topic = tf.nn.softmax(words_topic, axis=1)

    for topic_num in range(words_topic.shape[1]):
        wt = words_topic[:, topic_num, :]
        topics.append(
            " ".join(
                top_words(
                    tf.reduce_mean(wt, -1)[..., None], vocabulary, top_n=top_n_topic
                )
            )
        )
        print("Topic {}: {}".format(topic_num, topics[-1]))
        for i, time in enumerate(times_display):
            topics_t = top_words(wt[:, i, None], vocabulary, top_n=top_n_time)
            print("- at t={}: {}".format(time, " ".join(topics_t)))
    return topics


def get_correlation(scale_tril: tf.Tensor):
    """Get correlation and Sigma matrix from scale_tril.

    Args:
        scale_tril (tf.Tensor): Lower Cholesky of Sigma matrix.

    Returns:
        tuple: Correlation matrix and Sigma
    """
    Sigma = tf.matmul(scale_tril, scale_tril, transpose_b=True)
    diag = np.sqrt(tf.linalg.diag_part(Sigma))  # standard deviations of each variable
    corr = Sigma / tf.matmul(tf.expand_dims(diag, -1), tf.expand_dims(diag, -2))
    return corr, Sigma


def _perplexity_test(mdl, X, index_points, batch_size):
    from tqdm import tqdm

    kl_weight = 0

    ntot = X.shape[0]
    dataset = tf.data.Dataset.zip(
        tuple(map(tf.data.Dataset.from_tensor_slices, (X, index_points)))
    )
    data_ts = dataset.batch(batch_size)

    perplexity_value = 0
    n_topics = mdl.surrogate_posterior_beta.batch_shape[1]
    for x_batch, index_points_batch in tqdm(data_ts):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        emm_init = tf.nn.softmax(
            np.random.standard_normal([x_batch.shape[0], 1, n_topics])
        )
        ess_init = tf.nn.softmax(np.zeros([x_batch.shape[0], 1, n_topics]) + 0.001)
        emm = tf.Variable(emm_init, name="emm")
        ess = tfp.util.TransformedVariable(
            ess_init,
            bijector=tfb.Chain([tfb.Shift(tf.cast(1e-3, mdl.dtype)), tfb.Softplus()]),
            name="ess",
        )
        X = x_batch
        index_points = index_points_batch

        def f(x):
            return emm, ess

        trainable_variables = [emm, ess.trainable_variables[0]]

        pbar = tqdm(range(100), disable=False)
        for i in pbar:
            with tf.GradientTape(
                watch_accessed_variables=trainable_variables is None
            ) as tape:
                for v in trainable_variables or []:
                    tape.watch(v)
                posterior_mu = mdl.surrogate_posterior_mu.get_marginal_distribution(
                    index_points
                )

                # needing to put n_topics as last dimension
                mu_sample = tfb.Transpose(rightmost_transposed_ndims=2).forward(
                    posterior_mu.sample(sample_size)
                )

                scale_tril = mdl.surrogate_posterior_ell.sample(
                    sample_size, index_points=observation_index_points
                )

                prior_eta = tfd.MultivariateNormalTriL(
                    loc=tf.expand_dims(
                        mu_sample, -2
                    ),  # this is to ease kl_divergence code
                    scale_tril=tf.expand_dims(scale_tril, -3),
                    validate_args=mdl.validate_args,
                    allow_nan_stats=False,
                    name="prior_eta",
                )

                posterior_eta = tfd.MultivariateNormalDiag(
                    loc=emm, scale_diag=ess, name="topics_posterior"
                )
                eta_samples = posterior_eta.sample(mdl.draw_samples)
                eta_samples = tf.math.softmax(eta_samples)

                posterior_beta = mdl.surrogate_posterior_beta(index_points)
                beta_samples = posterior_beta.sample(mdl.draw_samples)
                beta_samples = tfb.Transpose(rightmost_transposed_ndims=3)(beta_samples)
                beta_samples = tf.nn.softmax(beta_samples)
                reconstruction = tf.reduce_mean(
                    mdl.expectation(
                        X=X, eta_samples=eta_samples, beta_samples=beta_samples
                    ),
                    0,
                )

                post_eta_reshaped = tfd.MultivariateNormalDiag(
                    loc=tf.transpose(posterior_eta.loc, [1, 0, 2]),
                    scale_diag=tf.transpose(posterior_eta.stddev(), [1, 0, 2]),
                )
                kl = tf.transpose(
                    tf.reduce_mean(
                        [
                            tfd.kl_divergence(post_eta_reshaped, prior_eta[i])
                            for i in range(mdl.draw_samples)
                        ],
                        axis=0,
                    )
                )

                elbo = reconstruction - kl
                loss = -tf.reduce_mean(elbo, 0)

            watched_variables = tape.watched_variables()
            grads = tape.gradient(loss, watched_variables)
            optimizer.apply_gradients(zip(grads, watched_variables))
            perpl = mdl.perplexity(x_batch, elbo=elbo)

        perplexity_value += perpl
        pbar.set_description("perpl {:.3e}".format(perplexity_value))
    return perplexity_value
