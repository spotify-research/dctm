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
"""Plotting utilities."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

RIGHT_LEGEND_OUTSIDE = dict(loc="center left", bbox_to_anchor=(1, 0.5))
LOWER_LEGEND_OUTSIDE = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.175),
    fancybox=True,
    shadow=False,
    ncol=2,
)


def plot_predictions(
    mdl,
    predictions,
    index_points,
    topics,
    inverse_transform_fn,
    restrict_to=None,
    cmap="tab20c",
    min_prob=0.0,
    legend="right",
):
    """Plot model prediction on data.

    `predictions` should be the average of the mean of the posterior of eta
    computed across documents.

    Args:
        mdl (dynamic_correlated_topic_model.DCTM): Trained model.
        predictions (np.array): Average of mean of posterior of eta for each document.
        index_points (np.array): Index points associated to predictions.
        topics (list): List of names of topics.
        inverse_transform_fn (function): Mapping from index points to readable strings.
        restrict_to (list, optional): List of topics to show
            (None: plot everything). Defaults to None.
        cmap (str, optional): Color map. Defaults to "tab20c".
        min_prob (float, optional): Minimum value to plot.
            If the average value of the GP is less
            than `min_prob`, skip the plot. Defaults to 0.0.
        legend (str, optional): "right" or "bottom". Defaults to "right".
    """
    plt.title("Probability of topics over time")
    cm = plt.get_cmap(cmap)
    if restrict_to is None:
        restrict_to = range(predictions.shape[0])

    colors = cm(np.linspace(0, 1, max(len(restrict_to), 9)))
    prev = 0
    for i, t in enumerate(restrict_to):
        if predictions[t].mean() > min_prob:
            curr = prev + predictions[t]
            plt.fill_between(
                np.unique(index_points),
                prev,
                curr,
                label="{}:{}".format(t, topics[t][:20]),
                color=colors[i],
            )
            prev = curr

    plt.xticks(
        np.unique(index_points)[::2],
        inverse_transform_fn(np.unique(index_points)[:, None])[::2],
        rotation=45,
    )
    if legend == "right":
        legend_kwargs = RIGHT_LEGEND_OUTSIDE
    else:
        legend_kwargs = LOWER_LEGEND_OUTSIDE
    plt.gca().legend(**legend_kwargs)
    f = plt.gcf()
    return f


def plot_mu(
    mdl,
    test_points,
    topics,
    inverse_transform_fn,
    restrict_to=None,
    color_fn=plt.cm.jet,
    figsize=(9, 9),
    plot_if_higher_of=0,
    predictions=None,
    index_tr=None,
):
    f = plt.figure(figsize=figsize)
    mu = mdl.surrogate_posterior_mu.get_marginal_distribution(index_points=test_points)
    mu_sm = tf.nn.softmax(mu.mean(), axis=0)
    mu_sample = tf.nn.softmax(mu.sample(110), axis=1)
    mu_90p = tfp.stats.percentile(mu_sample, 95, axis=0)
    mu_10p = tfp.stats.percentile(mu_sample, 5, axis=0)

    if restrict_to is None:
        restrict_to = range(mu_sm.shape[0])

    colors = color_fn(np.linspace(0, 1, len(restrict_to)))

    for i, j in enumerate(restrict_to):
        if tf.reduce_mean(tf.abs(mu_sm[j])) > plot_if_higher_of:
            (line,) = plt.plot(
                test_points,
                mu_sm[j],
                label="{}:{}".format(j, topics[j][:30]),
                color=colors[i],
            )
            plt.fill_between(
                test_points[:, 0],
                mu_10p[j],
                mu_90p[j],
                color=line.get_color(),
                alpha=0.3,
                lw=1.5,
            )
            if predictions is not None and index_tr is not None:
                plt.plot(np.unique(index_tr), predictions[i], color=line.get_color())

    plt.xticks(test_points[::8], inverse_transform_fn(test_points)[::8], rotation=30)
    plt.title(r"Posterior topic probability $\mu$")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.gca().legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return f


def plot_mu_stacked(
    mean,
    test_points,
    topics,
    inverse_transform_fn,
    restrict_to=None,
    color_fn=plt.cm.jet,
    figsize=(9, 9),
    plot_if_higher_of=0,
    legend="right",
):
    f = plt.figure(figsize=figsize)

    if restrict_to is None:
        restrict_to = range(mean.shape[0])

    colors = color_fn(np.linspace(0, 1, max(len(restrict_to), 9)))
    prev = None

    c = 0
    for i, j in enumerate(restrict_to):
        if tf.reduce_mean(tf.abs(mean[j])) > plot_if_higher_of:
            prev = 0 if prev is None else prev
            plt.fill_between(
                test_points[:, 0],
                prev,
                prev + mean[j],
                label="{}:{}".format(j, topics[j][:20]),
                color=colors[i],
            )
            prev = prev + mean[j]
            c += 1

    plt.xticks(test_points[::8], inverse_transform_fn(test_points)[::8], rotation=30)
    plt.title(r"Posterior topic probability $\mu$")
    plt.xlabel("Time")
    plt.tight_layout()
    if legend == "right":
        legend_kwargs = RIGHT_LEGEND_OUTSIDE
    else:
        legend_kwargs = LOWER_LEGEND_OUTSIDE
    plt.gca().legend(**legend_kwargs)
    plt.show()
    return f


def plot_sigma(
    samples,
    index_points,
    topic_num,
    topics,
    inverse_transform_fn,
    restrict_to=None,
    color_fn=plt.cm.jet,
    legend="right",
    plot_if_higher_of=0,
):
    corr_10p = tfp.stats.percentile(samples, 5, axis=0)
    corr = tfp.stats.percentile(samples, 50, axis=0)
    corr_90p = tfp.stats.percentile(samples, 95, axis=0)
    plt.title("Topic {}: {}".format(topic_num, topics[topic_num][:45]))
    n_topics = samples.shape[-1]
    if restrict_to is None:
        restrict_to = range(n_topics)
    colors = color_fn(np.linspace(0, 1, len(restrict_to)))
    for i, t in enumerate(restrict_to):
        if (
            t == topic_num
            or tf.reduce_mean(np.abs(corr[:, topic_num, t])) < plot_if_higher_of
        ):
            continue
        (line,) = plt.plot(
            index_points[:, 0],
            corr[:, topic_num, t],
            label="{}:{}".format(t, topics[t][:20]),
            color=colors[i],
        )
        plt.fill_between(
            index_points[:, 0],
            corr_10p[:, topic_num, t],
            corr_90p[:, topic_num, t],
            color=line.get_color(),
            alpha=0.7,
            lw=1.5,
        )

    plt.hlines(0, -1, 1, ls="--", color="k")

    plt.xticks(index_points[::7], inverse_transform_fn(index_points)[::7], rotation=30)
    # plt.ylim([-0.75, 1])
    plt.ylabel("Temporal correlation")
    if legend == "right":
        legend_kwargs = RIGHT_LEGEND_OUTSIDE
    else:
        legend_kwargs = LOWER_LEGEND_OUTSIDE

    plt.gca().legend(**legend_kwargs)
    plt.tight_layout()
    return plt.gcf()


def plot_beta(
    mdl,
    test_points,
    topic_num,
    vocabulary,
    inverse_transform_fn,
    topics,
    restrict_words_to=None,
    color_fn=plt.cm.jet,
    figsize=(7, 3),
    legend="right",
):
    beta = mdl.surrogate_posterior_beta(test_points)
    beta_mean = tf.math.softmax(beta.mean(), axis=0)
    beta_sample = tf.math.softmax(beta.sample(50), axis=1)

    b_90p = tfp.stats.percentile(beta_sample, 95, axis=0)
    b_10p = tfp.stats.percentile(beta_sample, 5, axis=0)

    f = plt.figure(figsize=figsize)
    for w, word in enumerate(beta_mean[:, topic_num]):
        if vocabulary[w] in topics[topic_num].split():
            if not restrict_words_to or vocabulary[w] in restrict_words_to:
                (line,) = plt.plot(
                    test_points, word, label="{}".format(vocabulary[w])
                )  # , color=next(col_iter));
                plt.fill_between(
                    test_points[:, 0],
                    b_10p[w, topic_num],
                    b_90p[w, topic_num],
                    color=line.get_color(),
                    alpha=0.3,
                    lw=1.5,
                )

        plt.title("Topic: {}".format(topics[topic_num][:50]))
    plt.xticks(test_points[::8], inverse_transform_fn(test_points)[::8], rotation=45)

    if legend == "right":
        legend_kwargs = RIGHT_LEGEND_OUTSIDE
    else:
        legend_kwargs = LOWER_LEGEND_OUTSIDE

    plt.gca().legend(**legend_kwargs)
    plt.tight_layout()
    return f


def plot_beta_and_stacked(
    mdl,
    test_points,
    topic_num,
    vocabulary,
    inverse_transform_fn,
    topics,
    restrict_words_to=None,
    color_fn=plt.cm.jet,
    figsize=(7, 3),
    legend="right",
    split_by=None,
    min_proba=0.,
    n_samples=100,
):
    beta = mdl.surrogate_posterior_beta.get_marginal_distribution(
        index_points=test_points
    )
    beta_sample = tf.math.softmax(beta.sample(n_samples), axis=1)

    beta_mean = tf.reduce_mean(beta_sample, 0)
    b_90p = tfp.stats.percentile(beta_sample, 95, axis=0)
    b_10p = tfp.stats.percentile(beta_sample, 5, axis=0)

    f, ax = plt.subplots(2, 1, sharex=True, figsize=figsize)
    prev = None
    for w, word in enumerate(beta_mean[:, topic_num]):
        if vocabulary[w] in topics[topic_num].split(split_by):
            if not restrict_words_to or vocabulary[w] in restrict_words_to and tf.reduce_mean(word)>= min_proba:
                (line,) = ax[0].plot(
                    test_points, word, label="{}".format(vocabulary[w])
                )  # , color=next(col_iter));
                ax[0].fill_between(
                    test_points[:, 0],
                    b_10p[w, topic_num],
                    b_90p[w, topic_num],
                    color=line.get_color(),
                    alpha=0.3,
                    lw=1.5,
                )

                prev = 0 if prev is None else prev
                ax[1].fill_between(
                    test_points[:, 0],
                    prev,
                    prev + word,
                    color=line.get_color(),
                    label="{}".format(vocabulary[w]),
                )
                prev = prev + word

    ax[0].set_title("Topic: {}".format(topics[topic_num][:50]))
    ax[0].set_ylabel("Word probability")
    ax[1].set_ylabel("Stacked word probability")
    plt.xticks(test_points[::8], inverse_transform_fn(test_points)[::8], rotation=45)

    if legend == "right":
        legend_kwargs = RIGHT_LEGEND_OUTSIDE
    else:
        legend_kwargs = LOWER_LEGEND_OUTSIDE

    ax[1].legend(**legend_kwargs)
    f.tight_layout()
    return f


def plot_beta_words_topics(surrogate_posterior_beta, true_beta):
    mean = surrogate_posterior_beta.mean()
    stddev = surrogate_posterior_beta.stddev()

    (
        n_words,
        n_topics,
    ) = surrogate_posterior_beta.variational_inducing_observations_loc.shape[:2]
    f, ax = plt.subplots(n_words, n_topics, figsize=(12, 15), sharex=True, sharey=True)
    for i in range(n_words):
        for j in range(n_topics):
            ax[i, j].scatter(
                surrogate_posterior_beta.inducing_index_points[..., 0],
                surrogate_posterior_beta.variational_inducing_observations_loc[i, j],
                marker="x",
                s=50,
                zorder=10,
                label="Variational mean",
                color="C0",
            )

            ax[i, j].plot(
                surrogate_posterior_beta.index_points,
                true_beta[i, j],
                marker="x",
                label="True beta",
                alpha=0.7,
                color="C2",
            )
            #         ax[i,j].plot(index_points[:, 0], tf.transpose(
            #             posterior_beta.sample(20)[:, i,j]), alpha=0.2);

            m = mean[i, j]
            (line,) = ax[i, j].plot(surrogate_posterior_beta.index_points[:, 0], m, label="mean")
            ax[i, j].fill_between(
                surrogate_posterior_beta.index_points[:, 0],
                (m - 2 * stddev ** 0.5),
                (m + 2 * stddev ** 0.5),
                color=line.get_color(),
                alpha=0.6,
                lw=1.5,
            )
    ax[n_words // 2, -1].legend(**RIGHT_LEGEND_OUTSIDE)
    f.tight_layout()
    return f
