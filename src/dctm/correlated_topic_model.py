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
"""Correlated Topic Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.compat import v1 as tf1

from dctm.relaxed_latent_dirichlet_allocation import RelaxedLDA

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers


class CTMBeta(RelaxedLDA):
    """Correlated Topic Model.

    beta is treated as a variational parameter.
    """

    def __init__(
        self,
        n_words: int,
        n_topics: int = 3,
        prior_eta_loc: Union[None, tf.Variable, tf.Tensor] = None,
        prior_eta_scale_diag: Union[None, tf.Variable, tf.Tensor] = None,
        prior_beta_loc: Union[None, tf.Variable, tf.Tensor] = None,
        prior_beta_scale_diag: Union[None, tf.Variable, tf.Tensor] = None,
        m_beta: Union[None, tf.Variable] = None,
        sigma_beta: Union[None, tf.Variable] = None,
        layer_sizes: List[Union[int, str]] = (300, 300, 300),
        activation: str = "relu",
        dtype: Union[np.dtype, tf.DType, str] = np.float64,
        jitter: float = 1e-5,
        validate_args: bool = False,
        allow_nan_stats: bool = False,
        name: str = "CTMBeta",
    ):
        """Correlated topic model. beta is the only variational parameter.

        Args:
            n_words (int): Dimensionality of the dataset.
            n_topics (int, optional): Number of topics. Defaults to 3.
            prior_eta_loc (Union[None, tf.Variable, tf.Tensor], optional): Mean prior for eta. Defaults to None.
            prior_eta_scale_diag (Union[None, tf.Variable, tf.Tensor], optional): Scale prior for eta. Defaults to None.
            prior_beta_loc (Union[None, tf.Variable, tf.Tensor], optional): Mean prior for beta. Defaults to None.
            prior_beta_scale_diag (Union[None, tf.Variable, tf.Tensor], optional): Scale prior for beta. Defaults to None.
            m_beta (Union[None, tf.Variable], optional): Posterior mean for beta. Defaults to None.
            sigma_beta (Union[None, tf.Variable], optional): Posterior scale for beta. Defaults to None.
            layer_sizes (List[Union[int, str]], optional): Layer sizes for the creation of
                the neural network parametrising mu and Sigma
                for eta posterior. Defaults to (300, 300, 300).
            activation (str, optional): Neural net activation. Defaults to "relu".
            dtype (Union[np.dtype, tf.DType, str], optional): Data type of the variables. Defaults to np.float64.
            jitter (float, optional): Jitter for encoder. Defaults to 1e-5.
            validate_args (bool, optional): Validate the arguments of distribution classes. Defaults to False.
            allow_nan_stats (bool, optional): Allow nan stats for the arguments of distribution classes. Defaults to False.
            name (str, optional): Class name. Defaults to "CTMBeta".
        """
        RelaxedLDA.__init__(
            self,
            n_topics=n_topics,
            n_words=n_words,
            prior_eta_loc=prior_eta_loc,
            prior_eta_scale_diag=prior_eta_scale_diag,
            layer_sizes=layer_sizes,
            activation=activation,
            jitter=jitter,
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

        del self.topics_words_logits  # unused

        beta_shape = (n_topics, n_words)
        if prior_beta_loc is None:
            prior_beta_loc = np.random.standard_normal(beta_shape).astype(dtype=dtype)
        if prior_beta_scale_diag is None:
            prior_beta_scale_diag = tf.zeros(beta_shape, dtype=dtype) + 1e-2

        self.prior_beta = tfd.MultivariateNormalDiag(
            prior_beta_loc,
            scale_diag=prior_beta_scale_diag,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="prior_beta",
        )

        if m_beta is None:
            m_beta = np.random.standard_normal(beta_shape).astype(dtype=dtype)
        m_beta = tf.Variable(m_beta, name="m_beta")

        if sigma_beta is None:
            sigma_beta = tf.zeros(beta_shape, dtype=dtype) + 1e-5
        sigma_beta = tfp.util.TransformedVariable(
            sigma_beta,
            bijector=tfb.Chain([tfb.Shift(tf.cast(1e-8, dtype)), tfb.Softplus()]),
            name="sigma_beta",
        )
        self.surrogate_posterior_beta = tfd.MultivariateNormalDiag(
            m_beta,
            scale_diag=sigma_beta,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="posterior_beta",
        )

    @tf.Module.with_name_scope
    @tf.function
    def elbo(
        self,
        X: tf.Tensor,
        observation_index_points: tf.Tensor = None,
        kl_weight: float = 1.0,
        sample_size: int = 1,
    ) -> tf.Tensor:
        """ELBO computation.

        Args:
            X (tf.Tensor): Input data. Should be at least 2-dimensional.
            observation_index_points (tf.Tensor, optional): Unused.
            kl_weight (float, optional): Weight for KL divergence.
                Useful for minibatch training. Defaults to 1.

        Returns:
            tf.Tensor: elbo for each sample.
        """
        # del observation_index_points  # unused

        posterior_eta = self.surrogate_posterior_eta(X)
        eta_samples = tf.nn.softmax(posterior_eta.sample(sample_size))

        posterior_beta = self.surrogate_posterior_beta
        beta_samples = tf.nn.softmax(posterior_beta.sample(sample_size))

        reconstruction = tf.reduce_mean(
            self.expectation(X=X, eta_samples=eta_samples, beta_samples=beta_samples), 0
        )
        kl = tfd.kl_divergence(posterior_eta, self.prior_eta)

        # this is equivalent to flattening the parameters and then computing kl
        kl_beta = tf.reduce_sum(
            tfd.kl_divergence(posterior_beta, self.prior_beta), axis=0
        )

        # with tf.control_dependencies(
        #     [tf1.assert_greater(kl, self._kl_guard, message="kl")]
        # ):
        #     kl = tf.identity(kl)

        # if we sum everything this needs to be 1.
        norm_factor = np.prod(X.shape[:-1])
        kl_global = kl_beta / norm_factor

        # for the minibatch case, we need to reweight the prior
        # for the total number of samples in the dataset
        elbo = reconstruction - kl - kl_weight * kl_global
        return elbo


class CTMMuSigma(RelaxedLDA):
    """In this version, only mu and Sigma are treated as VI params."""

    def __init__(
        self,
        n_words: int,
        n_topics: int = 3,
        prior_mu_loc: Union[None, tf.Variable, tf.Tensor] = None,
        prior_mu_scale_diag: Union[None, tf.Variable, tf.Tensor] = None,
        m_mu: Union[None, tf.Variable] = None,
        sigma_mu: Union[None, tf.Variable] = None,
        prior_ell_loc: Union[None, tf.Variable, tf.Tensor] = None,
        prior_ell_scale_diag: Union[None, tf.Variable, tf.Tensor] = None,
        m_ell: Union[None, tf.Variable] = None,
        sigma_ell: Union[None, tf.Variable] = None,
        topics_words_logits: Union[None, tf.Variable, tf.Tensor] = None,
        layer_sizes: List[Union[int, str]] = (300, 300, 300),
        activation: str = "relu",
        dtype: Union[np.dtype, tf.DType, str] = np.float64,
        jitter: float = 1e-5,
        validate_args: bool = False,
        allow_nan_stats: bool = False,
        name: str = "CTMMuSigma",
    ):
        """Correlated topic model. Only mu and sigma are variationa parameters.

        Args:
            n_words (int): Dimensionality of the dataset.
            n_topics (int, optional): Number of topics. Defaults to 3.
            prior_mu_loc (Union[None, tf.Variable, tf.Tensor], optional): Prior mean for mu. Defaults to None.
            prior_mu_scale_diag (Union[None, tf.Variable, tf.Tensor], optional): Prior scale for mu. Defaults to None.
            m_mu (Union[None, tf.Variable], optional): Posterior mean for mu. Defaults to None.
            sigma_mu (Union[None, tf.Variable], optional): Posterior scale for mu. Defaults to None.
            prior_ell_loc (Union[None, tf.Variable, tf.Tensor], optional): Prior mean for ell. Defaults to None.
            prior_ell_scale_diag (Union[None, tf.Variable, tf.Tensor], optional): Prior scale for ell. Defaults to None.
            m_ell (Union[None, tf.Variable], optional): Posterior mean for ell. Defaults to None.
            sigma_ell (Union[None, tf.Variable], optional): Posterior scale for ell. Defaults to None.
            topics_words_logits (Union[None, tf.Variable, tf.Tensor], optional): Topic words logits. Defaults to None.
            layer_sizes (List[Union[int, str]], optional): Layer sizes for the creation of
                the neural network parametrising mu and Sigma
                for eta posterior. Defaults to (300, 300, 300).
            activation (str, optional): Neural net activation. Defaults to "relu".
            dtype (Union[np.dtype, tf.DType, str], optional): Data type of the variables. Defaults to np.float64.
            jitter (float, optional): Jitter for encoder. Defaults to 1e-5.
            validate_args (bool, optional): Validate the arguments of distribution classes. Defaults to False.
            allow_nan_stats (bool, optional): Allow nan stats for the arguments of distribution classes. Defaults to False.
            name (str, optional): Class name. Defaults to "CTMMuSigma".
        """
        super(CTMMuSigma, self).__init__(
            n_topics=n_topics,
            n_words=n_words,
            topics_words_logits=topics_words_logits,
            layer_sizes=layer_sizes,
            activation=activation,
            jitter=jitter,
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

        del self.prior_eta  # unused

        dof = n_topics * (n_topics + 1) // 2
        self.bijector_ell = tfb.FillScaleTriL(diag_shift=tf.cast(1e-5, dtype))

        if prior_mu_loc is None:
            prior_mu_loc = np.random.standard_normal([1, n_topics]).astype(dtype=dtype)

        if prior_mu_scale_diag is None:
            prior_mu_scale_diag = tf.zeros([1, n_topics], dtype=dtype) + 0.01

        self.prior_mu = tfd.MultivariateNormalDiag(
            prior_mu_loc,
            scale_diag=prior_mu_scale_diag,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="prior_mu",
        )

        if m_mu is None:
            m_mu = np.random.standard_normal([1, n_topics]).astype(dtype=dtype)
        m_mu = tf.Variable(m_mu, name="m_mu")

        if sigma_mu is None:
            sigma_mu = tf.zeros([1, n_topics], dtype=dtype) + 0.01
        sigma_mu = tfp.util.TransformedVariable(
            sigma_mu,
            bijector=tfb.Chain([tfb.Shift(tf.cast(1e-3, dtype)), tfb.Softplus()]),
            name="sigma_mu",
        )
        self.surrogate_posterior_mu = tfd.MultivariateNormalDiag(
            m_mu,
            scale_diag=sigma_mu,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="posterior_mu",
        )

        if prior_ell_loc is None:
            prior_ell_loc = np.random.standard_normal([1, dof]).astype(dtype=dtype)

        if prior_ell_scale_diag is None:
            prior_ell_scale_diag = tf.zeros([1, dof], dtype=dtype) + 0.01

        self.prior_ell = tfd.MultivariateNormalDiag(
            prior_ell_loc,
            scale_diag=prior_ell_scale_diag,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="prior_ell",
        )

        if m_ell is None:
            m_ell = np.random.standard_normal([1, dof]).astype(dtype=dtype)
        m_ell = tf.Variable(m_ell, name="m_ell")

        if sigma_ell is None:
            sigma_ell = tf.zeros([1, dof], dtype=dtype) + 0.01
        sigma_ell = tfp.util.TransformedVariable(
            sigma_ell,
            bijector=tfb.Chain([tfb.Shift(tf.cast(1e-3, dtype)), tfb.Softplus()]),
            name="sigma_ell",
        )
        self.surrogate_posterior_ell = tfd.MultivariateNormalDiag(
            m_ell,
            scale_diag=sigma_ell,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="posterior_ell",
        )

    @tf.Module.with_name_scope
    @tf.function
    def elbo(
        self,
        X: tf.Tensor,
        observation_index_points: tf.Tensor = None,
        kl_weight: float = 1.0,
        sample_size: int = 1,
    ) -> tf.Tensor:
        """ELBO computation.

        Args:
            X (tf.Tensor): Input data. Should be at least 2-dimensional.
            observation_index_points (tf.Tensor, optional): Unused.
            kl_weight (float, optional): Weight for KL divergence.
                Useful for minibatch training. Defaults to 1.

        Returns:
            tf.Tensor: elbo for each sample.
        """
        del observation_index_points  # unused

        posterior_mu = self.surrogate_posterior_mu
        mu_sample = posterior_mu.sample(sample_size)

        posterior_ell = self.surrogate_posterior_ell
        ell_sample = posterior_ell.sample(sample_size)

        prior_eta = tfd.MultivariateNormalTriL(
            loc=mu_sample,
            scale_tril=self.bijector_ell(ell_sample),
            validate_args=self.validate_args,
            allow_nan_stats=False,
        )

        posterior_eta = self.surrogate_posterior_eta(X)
        eta_samples = tf.math.softmax(posterior_eta.sample(sample_size))

        reconstruction = tf.reduce_mean(
            self.expectation(
                X=X, eta_samples=eta_samples, beta_samples=self.topics_words_logits
            ),
            0,
        )

        kl = tf.reduce_mean(tfd.kl_divergence(posterior_eta, prior_eta), 0)
        # here we sum everything
        # reconstruction = tf.reduce_sum(reconstruction)
        # kl = tf.reduce_sum(kl, 0)
        # with tf.control_dependencies(
        #     [tf1.assert_greater(kl, self._kl_guard, message="kl")]
        # ):
        #     kl = tf.identity(kl)

        kl_mu = tfd.kl_divergence(posterior_mu, self.prior_mu)
        kl_ell = tfd.kl_divergence(posterior_ell, self.prior_ell)

        # if we sum everything this needs to be 1.
        norm_factor = np.prod(X.shape[:-1])
        kl_global = (kl_mu + kl_ell) / norm_factor

        # for the minibatch case, we need to reweight the prior
        # for the total number of samples in the dataset
        elbo = reconstruction - kl - kl_weight * kl_global
        return elbo


class CTM(CTMBeta, CTMMuSigma):
    """Correlated topic model. Both mu, Sigma and beta are variational parameters."""

    def __init__(
        self,
        n_words: int,
        n_topics: int = 3,
        prior_beta_loc: Union[None, tf.Variable, tf.Tensor] = None,
        prior_beta_scale_diag: Union[None, tf.Variable, tf.Tensor] = None,
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
        validate_args: bool = False,
        allow_nan_stats: bool = False,
        name: str = "CTM",
    ):
        """Correlated topic model.

        Args:
            n_words (int): Dimensionality of the dataset.
            n_topics (int, optional): Number of topics. Defaults to 3.
            prior_beta_loc (Union[None, tf.Variable, tf.Tensor], optional): Prior mean for beta. Defaults to None.
            prior_beta_scale_diag (Union[None, tf.Variable, tf.Tensor], optional): Prior scale for beta. Defaults to None.
            m_beta (Union[None, tf.Variable], optional): Posterior mean for beta. Defaults to None.
            sigma_beta (Union[None, tf.Variable], optional): Posterior scale for beta. Defaults to None.
            prior_mu_loc (Union[None, tf.Variable, tf.Tensor], optional): Prior mean for mu. Defaults to None.
            prior_mu_scale_diag (Union[None, tf.Variable, tf.Tensor], optional): Prior scale for mu. Defaults to None.
            m_mu (Union[None, tf.Variable], optional): Posterior mean for mu. Defaults to None.
            sigma_mu (Union[None, tf.Variable], optional): Posterior scale for mu. Defaults to None.
            prior_ell_loc (Union[None, tf.Variable, tf.Tensor], optional): Prior mean for ell. Defaults to None.
            prior_ell_scale_diag (Union[None, tf.Variable, tf.Tensor], optional): Prior scale for ell. Defaults to None.
            m_ell (Union[None, tf.Variable], optional): Posterior mean for ell. Defaults to None.
            sigma_ell (Union[None, tf.Variable], optional): Posterior scale for ell. Defaults to None.
            layer_sizes (List[Union[int, str]], optional): Layer sizes for the creation of
                the neural network parametrising mu and Sigma
                for eta posterior. Defaults to (300, 300, 300).
            activation (str, optional): Neural net activation. Defaults to "relu".
            dtype (Union[np.dtype, tf.DType, str], optional): Data type of the variables. Defaults to np.float64.
            jitter (float, optional): Jitter for encoder. Defaults to 1e-5.
            validate_args (bool, optional): Validate the arguments of distribution classes. Defaults to False.
            allow_nan_stats (bool, optional): Allow nan stats for the arguments of distribution classes. Defaults to False.
            name (str, optional): Class name. Defaults to "CTM".
        """
        super(CTM, self).__init__(
            n_topics=n_topics,
            n_words=n_words,
            prior_beta_loc=prior_beta_loc,
            prior_beta_scale_diag=prior_beta_scale_diag,
            m_beta=m_beta,
            sigma_beta=sigma_beta,
            layer_sizes=layer_sizes,
            activation=activation,
            jitter=jitter,
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )
        CTMMuSigma.__init__(
            self,
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
            jitter=jitter,
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

        del self.topics_words_logits

    @tf.Module.with_name_scope
    @tf.function
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
            observation_index_points (tf.Tensor): Unused. Here for compatibility.
            kl_weight (float, optional): Weight for KL divergence.
                Useful for minibatch training. Defaults to 1.

        Returns:
            tf.Tensor: ELBO of each sample.
        """
        # del observation_index_points  # unused

        posterior_mu = self.surrogate_posterior_mu
        mu_sample = posterior_mu.sample(sample_size)

        posterior_ell = self.surrogate_posterior_ell
        ell_sample = posterior_ell.sample(sample_size)

        prior_eta = tfd.MultivariateNormalTriL(
            loc=mu_sample,
            scale_tril=self.bijector_ell(ell_sample),
            validate_args=self.validate_args,
            allow_nan_stats=False,
        )

        posterior_eta = self.surrogate_posterior_eta(X)
        eta_samples = tf.math.softmax(posterior_eta.sample(sample_size))

        posterior_beta = self.surrogate_posterior_beta
        beta_samples = tf.math.softmax(posterior_beta.sample(sample_size))

        reconstruction = tf.reduce_mean(
            self.expectation(X=X, eta_samples=eta_samples, beta_samples=beta_samples), 0
        )

        kl = tf.reduce_mean(tfd.kl_divergence(posterior_eta, prior_eta), 0)

        # here we sum everything
        # reconstruction = tf.reduce_sum(reconstruction)
        # kl = tf.reduce_sum(kl, 0)

        # with tf.control_dependencies(
        #     [tf1.assert_greater(kl, self._kl_guard, message="kl")]
        # ):
        #     kl = tf.identity(kl)

        kl_mu = tfd.kl_divergence(posterior_mu, self.prior_mu)
        kl_ell = tfd.kl_divergence(posterior_ell, self.prior_ell)

        # this is equivalent to flattening the parameters and then computing kl
        kl_beta = tf.reduce_sum(
            tfd.kl_divergence(posterior_beta, self.prior_beta), axis=0
        )

        # if we sum everything this needs to be 1.
        norm_factor = np.prod(X.shape[:-1])
        kl_global = (kl_mu + kl_ell + kl_beta) / norm_factor

        # for the minibatch case, we need to reweight the prior
        # for the total number of samples in the dataset
        elbo = reconstruction - kl - kl_weight * kl_global
        return elbo


def get_topics_strings(
    topics_words, mu, sigma, vocabulary, topics_to_print=10, words_per_topic=30
):
    """Returns the summary of the learned topics.

  Arguments:
    topics_words: KxV tensor with topics as rows and words as columns.
    alpha: 1xK tensor of prior Dirichlet concentrations for the
        topics.
    vocabulary: A mapping of word's integer index to the corresponding string.
    topics_to_print: The number of topics with highest prior weight to
        summarize.
    words_per_topic: Number of wodrs per topic to return.
  Returns:
    summary: A np.array with strings.
  """
    mu = np.squeeze(mu, axis=0)
    sigma = np.squeeze(sigma, axis=0)
    # Use a stable sorting algorithm so that when alpha is fixed
    # we always get the same topics.
    highest_weight_topics = np.argsort(-mu, kind="mergesort")
    top_words = np.argsort(-topics_words, axis=1)

    res = []
    # try:
    for topic_idx in highest_weight_topics[:topics_to_print]:
        lst = [
            "index={} mu={:.2f} sigma={:.2f}".format(
                topic_idx, mu[topic_idx], sigma[topic_idx]
            )
        ]
        lst += [vocabulary[word] for word in top_words[topic_idx, :words_per_topic]]
        res.append(" ".join(lst))
    # except:
    #     res.append('')

    return np.array(res)


def print_top_words(components, feature_names, n_top_words: int = 10):
    """Print top words for a topic.

    Args:
        components (np.array): Matrix of topics x words.
        feature_names (list): Word vocabulary.
        n_top_words (int, optional): number of top words to display
    """
    for topic_idx, topic in enumerate(components):
        message = "Topic #%d: " % topic_idx
        message += " ".join(
            [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        )
        print(message)
    print()


def _get_fns(word_probs, batch_size):
    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(word_probs)
        dataset = dataset.batch(batch_size)
        return tf1.data.make_one_shot_iterator(dataset.repeat()).get_next()

    def eval_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(word_probs)
        dataset = dataset.batch(batch_size)
        return tf1.data.make_one_shot_iterator(dataset).get_next()

    return train_input_fn, eval_input_fn


def build_topics_fn(
    batch_size, n_words=55, n_topics=3, dtype=np.float64, correlated=False
):
    """Build fake data for correlated topic model."""
    mean = tf.random.normal([n_topics], dtype=dtype)

    if correlated:
        dof = n_topics * (n_topics + 1) // 2
        ell = tf.random.normal([dof], dtype=dtype)
        mdl = tfd.MultivariateNormalTriL(
            loc=mean, scale_tril=tfb.FillScaleTriL(diag_shift=tf.cast(1e-5, dtype))(ell)
        )
    else:
        ell = tf.random.normal([n_topics], dtype=dtype)
        mdl = tfd.MultivariateNormalDiag(loc=mean, scale_diag=tf.nn.softplus(ell))

    eta = mdl.sample(batch_size)
    topics = tfd.Multinomial(1, probs=tf.nn.softmax(eta)).sample()

    topics_words = np.zeros((n_topics, n_words), dtype=dtype)
    for i in range(n_topics):
        topics_words[i][i * n_words // n_topics : (i + 1) * n_words // n_topics] = 1

    word_probs = tf.matmul(topics, topics_words).numpy()
    vocabulary = [str(i) for i in range(n_words)]

    return (
        *_get_fns(word_probs, batch_size),
        vocabulary,
        word_probs,
        mean,
        ell,
        eta,
        topics,
        topics_words,
    )
