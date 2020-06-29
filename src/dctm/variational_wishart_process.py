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
"""Variational Wishart Process class.

References
[1] Heaukulani, Creighton, and Mark van der Wilk.
"Scalable Bayesian dynamic covariance modeling with variational Wishart
and inverse Wishart processes."
Advances in Neural Information Processing Systems. 2019.
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def gwp_sample_cholesky(gp_sample, lower_cholesky, time_last=False):
    """Build the Cholesky decomposition of Wishart process."""
    if time_last:
        gp_sample = tfb.Transpose(rightmost_transposed_ndims=3)(gp_sample)
    uut = tf.matmul(gp_sample, gp_sample, transpose_b=True)
    cholesky_uut = tf.linalg.cholesky(uut)
    cholesky_cov = tf.matmul(tf.expand_dims(lower_cholesky, -3), cholesky_uut)
    return cholesky_cov


class VariationalWishartProcessFullBayesian(tfd.VariationalGaussianProcess):
    def __init__(
        self,
        kernel,
        index_points,
        inducing_index_points,
        variational_inducing_observations_loc,
        variational_inducing_observations_scale,
        mean_fn=None,
        observation_noise_variance=None,
        predictive_noise_variance=None,
        jitter=1e-6,
        white_noise_jitter=1e-6,
        prior_lower_cholesky_loc=None,
        prior_lower_cholesky_scale_diag=None,
        m_lower_wishart=None,
        sigma_lower_wishart=None,
        validate_args=False,
        allow_nan_stats=False,
        name="VariationalWishartProcessFullBayesian",
    ):
        """Instantiate a VariationalWishartProcessFullBayesian Distribution.

    Args:
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        GP's covariance function.
      index_points: `float` `Tensor` representing finite (batch of) vector(s) of
        points in the index set over which the VGP is defined. Shape has the
        form `[b1, ..., bB, e1, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` and `e1` is the number
        (size) of index points in each batch (we denote it `e1` to distinguish
        it from the numer of inducing index points, denoted `e2` below).
        Ultimately the VariationalGaussianProcess distribution corresponds to an
        `e1`-dimensional multivariate normal. The batch shape must be
        broadcastable with `kernel.batch_shape`, the batch shape of
        `inducing_index_points`, and any batch dims yielded by `mean_fn`.
      inducing_index_points: `float` `Tensor` of locations of inducing points in
        the index set. Shape has the form `[b1, ..., bB, e2, f1, ..., fF]`, just
        like `index_points`. The batch shape components needn't be identical to
        those of `index_points`, but must be broadcast compatible with them.
      variational_inducing_observations_loc: `float` `Tensor`; the mean of the
        (full-rank Gaussian) variational posterior over function values at the
        inducing points, conditional on observed data. Shape has the form `[b1,
        ..., bB, e2]`, where `b1, ..., bB` is broadcast compatible with other
        parameters' batch shapes, and `e2` is the number of inducing points.
      variational_inducing_observations_scale: `float` `Tensor`; the scale
        matrix of the (full-rank Gaussian) variational posterior over function
        values at the inducing points, conditional on observed data. Shape has
        the form `[b1, ..., bB, e2, e2]`, where `b1, ..., bB` is broadcast
        compatible with other parameters and `e2` is the number of inducing
        points.
      mean_fn: Python `callable` that acts on index points to produce a (batch
        of) vector(s) of mean values at those index points. Takes a `Tensor` of
        shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
        (broadcastable with) `[b1, ..., bB]`. Default value: `None` implies
        constant zero function.
      observation_noise_variance: `float` `Tensor` representing the variance
        of the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
        Default value: `0.`
      predictive_noise_variance: `float` `Tensor` representing additional
        variance in the posterior predictive model. If `None`, we simply re-use
        `observation_noise_variance` for the posterior predictive noise. If set
        explicitly, however, we use the given value. This allows us, for
        example, to omit predictive noise variance (by setting this to zero) to
        obtain noiseless posterior predictions of function values, conditioned
        on noisy observations.
      jitter: `float` scalar `Tensor` added to the diagonal of the covariance
        matrix to ensure positive definiteness of the covariance matrix.
        Default value: `1e-6`.
      white_noise_jitter: `float` scalar `Tensor` added to the diagonal of Sigma to ensure its positive definiteness. Default value: `1e-6`.
      prior_lower_cholesky_loc: `float` `Tensor` representing the mean of the
        prior distribution for the lower Cholesky matrix. 
        Default value `None` implies lower Cholesky of an identity matrix.
      prior_lower_cholesky_scale_diag: `float` `Tensor` representing the scale of the
        prior distribution for the lower Cholesky matrix. 
        Default value `None` implies an array of ones.
      m_lower_wishart: `float` `Tensor` representing the mean of the
        posterior distribution for the lower Cholesky matrix. 
        Default value `None` implies lower Cholesky of an identity matrix.
      sigma_lower_wishart: `float` `Tensor` representing the mean of the
        posterior distribution for the lower Cholesky matrix. 
        Default value `None` implies an array of 1e-4.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
        Default value: `False`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "VariationalWishartProcessFullBayesian".

    Raises:
      ValueError: if `mean_fn` is not `None` and is not callable.
    """
        super(VariationalWishartProcessFullBayesian, self).__init__(
            kernel=kernel,
            index_points=index_points,
            inducing_index_points=inducing_index_points,
            variational_inducing_observations_loc=variational_inducing_observations_loc,
            variational_inducing_observations_scale=variational_inducing_observations_scale,
            mean_fn=mean_fn,
            observation_noise_variance=observation_noise_variance,
            predictive_noise_variance=predictive_noise_variance,
            jitter=jitter,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

        _, n_dim = variational_inducing_observations_loc.shape[:-1]
        dof = n_dim * (n_dim + 1) // 2
        self.bijector_ell = tfb.FillScaleTriL(diag_shift=tf.cast(1e-5, self.dtype))
        if prior_lower_cholesky_loc is None:
            prior_lower_cholesky_loc = self.bijector_ell.inverse(
                tf.eye(n_dim, dtype=self.dtype)
            )

        if prior_lower_cholesky_scale_diag is None:
            prior_lower_cholesky_scale_diag = tf.ones((dof,), dtype=self.dtype)

        self.prior_lower_cholesky_wishart = tfd.MultivariateNormalDiag(
            prior_lower_cholesky_loc,
            scale_diag=prior_lower_cholesky_scale_diag,
            validate_args=validate_args,
            name="prior_lower_cholesky_wishart",
        )

        if m_lower_wishart is None:
            m_lower_wishart = self.bijector_ell.inverse(tf.eye(n_dim, dtype=self.dtype))
        m_lower_wishart = tf.Variable(m_lower_wishart, name="m_lower_wishart")

        if sigma_lower_wishart is None:
            sigma_lower_wishart = tf.zeros((dof,), dtype=self.dtype) + 1e-4
        sigma_lower_wishart = tfp.util.TransformedVariable(
            sigma_lower_wishart,
            bijector=tfb.Chain([tfb.Shift(tf.cast(1e-6, self.dtype)), tfb.Softplus()]),
            name="sigma_lower_wishart",
        )

        self.surrogate_posterior_lower_cholesky_wishart = tfd.MultivariateNormalDiag(
            m_lower_wishart,
            scale_diag=sigma_lower_wishart,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="posterior_lower_cholesky_wishart",
        )

        # this implements the white noise jitter as in [1]
        self.white_noise_jitter = tfp.util.TransformedVariable(
            white_noise_jitter or 1e-5,
            bijector=tfb.Softplus(),
            name="white_noise_jitter",
            dtype=self.dtype,
        )

        # bijector to add the jitter to the diagonal
        self.bijector_jitter = tfb.Chain(
            [
                tfb.Invert(tfb.CholeskyOuterProduct()),
                tfb.TransformDiagonal(diag_bijector=tfb.Shift(self.white_noise_jitter)),
                tfb.CholeskyOuterProduct(),
            ]
        )

    def _call_sample_n(self, sample_shape, seed, name, **kwargs):
        with self._name_and_control_scope(name):
            sample_shape = tf.cast(sample_shape, tf.int32, name="sample_shape")
            sample_shape, n = self._expand_sample_shape_to_vector(
                sample_shape, "sample_shape"
            )
            samples = self._sample_n(
                n, seed=seed() if callable(seed) else seed, **kwargs
            )
            batch_event_shape = tf.shape(samples)[1:]
            final_shape = tf.concat([sample_shape, batch_event_shape], 0)
            samples = tf.reshape(samples, final_shape)
            # samples = self._set_sample_static_shape(samples, sample_shape)
            return samples

    def _sample_n(self, n, seed=None, index_points=None):
        f_sample = (
            super(VariationalWishartProcessFullBayesian, self)
            .get_marginal_distribution(index_points=index_points)
            .sample(n)
        )

        lower_cholesky_sample = self.surrogate_posterior_lower_cholesky_wishart.sample(
            n
        )

        scale_tril = self.bijector_jitter(
            gwp_sample_cholesky(
                f_sample, self.bijector_ell(lower_cholesky_sample), time_last=True
            )
        )
        return scale_tril

    def expectation(self, observations, f_samples, ell_samples):
        rv = tfd.MultivariateNormalTriL(
            0.0,
            scale_tril=self.bijector_jitter(
                gwp_sample_cholesky(
                    f_samples, ell_samples, time_last=True
                )
            ),
        )
        return rv.log_prob(observations)

    def surrogate_posterior_expected_log_likelihood(
        self,
        observations,
        observation_index_points=None,
        # log_likelihood_fn=None,
        sample_size=1,
        name=None,
    ):
        if observation_index_points is None:
            observation_index_points = self._index_points
        observation_index_points = tf.convert_to_tensor(
            observation_index_points, dtype=self.dtype, name="observation_index_points",
        )
        observations = tf.convert_to_tensor(
            observations, dtype=self.dtype, name="observations"
        )

        # qf = self._conditional(observation_index_points)
        # qf is what VariationalGaussianProcess represent.
        samples = (
            super(VariationalWishartProcessFullBayesian, self)
            .get_marginal_distribution(index_points=observation_index_points)
            .sample(sample_size)
        )
        # samples = super(VariationalWishartProcessFullBayesian, self).sample(
        #     sample_size, index_points=observation_index_points
        # )

        lower_cholesky_sample = self.surrogate_posterior_lower_cholesky_wishart.sample(
            sample_size
        )

        # log_likelihood_fn = lambda x: self.log_prob(x, observations, self.ell_var)
        # expected_logp = tfp.monte_carlo.expectation(
        #     f=lambda x: log_likelihood_fn(x),
        #     samples=samples,
        #     use_reparameterization=True,
        # )
        # equivalent to:
        # expected_logp = tf.reduce_mean(log_likelihood_fn(samples), 1)
        log_probability = self.expectation(
            observations,
            f_samples=samples,
            ell_samples=self.bijector_ell(lower_cholesky_sample),
        )
        log_probability = tf.reduce_mean(log_probability, 1)
        return log_probability

    def surrogate_posterior_kl_divergence_prior(self, name=None):
        with tf.name_scope(name or "surrogate_posterior_kl_divergence_prior"):
            kl = tf.reduce_sum(
                super(
                    VariationalWishartProcessFullBayesian, self
                ).surrogate_posterior_kl_divergence_prior(),
                axis=[-1, -2],
            )
            kl += tfd.kl_divergence(
                self.surrogate_posterior_lower_cholesky_wishart,
                self.prior_lower_cholesky_wishart,
            )
            return kl

    @tf.Module.with_name_scope
    @tf.function
    def variational_loss(
        self,
        observations,
        observation_index_points=None,
        # log_likelihood_fn=None,
        sample_size=1,
        kl_weight=1.0,
        name="variational_loss",
    ):
        if observation_index_points is None:
            observation_index_points = self._index_points
        observation_index_points = tf.convert_to_tensor(
            observation_index_points, dtype=self.dtype, name="observation_index_points",
        )

        observations = tf.convert_to_tensor(
            observations, dtype=self.dtype, name="observations"
        )
        kl_weight = tf.convert_to_tensor(kl_weight, dtype=self.dtype, name="kl_weight")

        recon = self.surrogate_posterior_expected_log_likelihood(
            observations=observations,
            observation_index_points=observation_index_points,
            # log_likelihood_fn=log_likelihood_fn,
            sample_size=sample_size,
        )
        kl_penalty = self.surrogate_posterior_kl_divergence_prior()
        kl_penalty /= np.prod(observations.shape[:-1])
        return -recon + kl_weight * kl_penalty


def optimize(
    model,
    observations,
    observation_index_points=None,
    n_iter=200,
    learning_rate=0.1,
    trainable_variables=None,
    sample_size=1,
):
    """Utility function to optimize a model, usign a progress bar.

    Also saves loss_values and watched_variables in the model.
    """
    from tqdm.notebook import tqdm

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    pbar = tqdm(range(n_iter))
    results = []
    watched_variables = None
    for i in pbar:
        with tf.GradientTape(
            watch_accessed_variables=trainable_variables is None
        ) as tape:
            for v in trainable_variables or []:
                tape.watch(v)
            loss_value = tf.reduce_sum(
                model.variational_loss(
                    observations=observations,
                    observation_index_points=observation_index_points,
                    sample_size=sample_size,
                )
            )

        results.append(loss_value)
        pbar.set_description("loss {:.3e}".format(loss_value))
        watched_variables = tape.watched_variables()
        grads = tape.gradient(loss_value, watched_variables)
        optimizer.apply_gradients(zip(grads, watched_variables))
    model.loss_values = results
    model.watched_variables = watched_variables
