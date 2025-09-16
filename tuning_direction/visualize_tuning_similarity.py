from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
from scipy.stats import sem
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

from viv1t import data
from viv1t.utils import plot

plot.set_font()

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches

BLOCK_SIZE = 25

H_INTERVAL = 25  # horizontal distance interval
V_INTERVAL = 50  # vertical distance interval

DATA_DIR = Path("../data")
PLOT_DIR = Path("figures") / "tuning_similarity"


def get_dot_size(p_value: float) -> int:
    if p_value < 0.005:
        return 25
    elif p_value < 0.05:
        return 13
    else:
        return 3


def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([min(1, x + amount * (1 - x)) for x in c])


def rank_sum_test(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """
    Compute rank sum p-values furthest delta_h group, per delta_v.
    From Ringach et al. 2016:
    For each individual curve, the size of the data points denotes the
    significance of a rank-sum test comparing the median distribution of data
    at a given distance to the distribution of the rightmost bin near 200µm.
    """
    horizontal_distances = sorted(df["d"].unique())
    vertical_distances = sorted(df["v"].unique())
    p_values = {
        v: np.zeros(len(horizontal_distances), dtype=np.float32)
        for v in vertical_distances
    }
    for v in vertical_distances:
        right_most_group = df.loc[
            (df["v"] == v) & (df["d"] == horizontal_distances[-1]),
        ].tuning_similarity.values
        for i, d in enumerate(horizontal_distances):
            res = mannwhitneyu(
                x=df.loc[(df["v"] == v) & (df["d"] == d)].tuning_similarity.values,
                y=right_most_group,
            )
            p_values[v][i] = res.pvalue
    return p_values


def exp_decay(x, A, B, C):
    return A * np.exp(-B * x) + C


def negative_log_likelihood(y_true, y_pred):
    residuals = y_true - y_pred
    sigma_hat_sq = np.mean(residuals**2)

    return 0.5 * np.sum(
        (residuals**2) / sigma_hat_sq + np.log(2 * np.pi * sigma_hat_sq)
    )


def regularized_loss(params, x, y, lambda_reg):
    A, B, C = params
    y_pred = exp_decay(x, A, B, C)
    nll = negative_log_likelihood(y, y_pred)
    reg_term = lambda_reg * (A**2 + B**2 + C**2)  # L2 Regularization
    return nll + reg_term


def fit_exponential(x, y, lambda_values=[0.001, 0.01, 0.1, 0.2, 0.5]):
    best_params = None
    best_loss = float("inf")

    initial_guess = (1, 1, 0)

    for lambda_reg in lambda_values:
        result = minimize(
            regularized_loss,
            initial_guess,
            args=(x, y, lambda_reg),
            bounds=[(0, 1e7), (1e-7, 1e7), (-1e7, 1e7)],
            method="L-BFGS-B",
        )

        if result.success:
            current_loss = regularized_loss(result.x, x, y, lambda_reg)
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = result.x

    return best_params


def likelihood_ratio_test(x1, y1, x2, y2, model_name, filename=None):
    # Fit separate models
    params1 = fit_exponential(x1, y1)
    params2 = fit_exponential(x2, y2)

    # Compute log-likelihoods for separate fits
    ll_1 = -negative_log_likelihood(y1, exp_decay(x1, *params1))
    ll_2 = -negative_log_likelihood(y2, exp_decay(x2, *params2))
    ll_separate = ll_1 + ll_2

    # Fit a combined model
    x_combined = np.concatenate([x1, x2])
    y_combined = np.concatenate([y1, y2])
    params_combined = fit_exponential(x_combined, y_combined)

    # Compute log-likelihood for combined fit
    ll_combined = -negative_log_likelihood(
        y_combined, exp_decay(x_combined, *params_combined)
    )

    # Compute the test statistic
    LRT_stat = -2 * (ll_combined - ll_separate)
    # Difference in parameters (6 for separate fits - 3 for combined fit)
    p_value = 1 - chi2.cdf(LRT_stat, 4)

    if filename is not None:
        figure = plt.figure()
        plt.plot(x1, y1, "o", c="b")
        plt.plot(x1, exp_decay(x1, *params1), "b-", label=model_name)
        plt.plot(x2, y2, "o", c="g")
        plt.plot(x2, exp_decay(x2, *params2), "g-", label="recorded")
        plt.plot(x1, exp_decay(x1, *params_combined), "r-", label="combined")
        plt.xlabel("Cortical distance (d) (µm)")
        plt.ylabel("z-scored tuning similarity")
        plt.legend()
        plt.title(
            f"LL{model_name} = {ll_1:.3f}, LLrecorded = {ll_2:.3f}, LL_combined = {ll_combined:.3f}, p={p_value:.5f}"
        )
        plt.tight_layout()
        plot.save_figure(figure, filename=filename, dpi=DPI)
        plt.close()

    return LRT_stat, p_value, params1, params2, params_combined


def add_inset_plot(
    ax: Axes,
    df: pd.DataFrame,
    df_recorded: pd.DataFrame,
    model_name: str,
    filename: Path,
    x_ticks: np.ndarray,
):
    num_mice = df.mouse.nunique()
    if num_mice > 1:
        df = df.groupby(["d", "mouse"])["tuning_similarity"].mean().reset_index()
        df["tuning_similarity"] = (
            df["tuning_similarity"] - df["tuning_similarity"].mean()
        ) / df["tuning_similarity"].std()
        df_recorded = (
            df_recorded.groupby(["d", "mouse"])["tuning_similarity"]
            .mean()
            .reset_index()
        )
        df_recorded["tuning_similarity"] = (
            df_recorded["tuning_similarity"] - df_recorded["tuning_similarity"].mean()
        ) / df_recorded["tuning_similarity"].std()
        lrt_stat, overall_p_value, params1, params2, params_combined = (
            likelihood_ratio_test(
                x1=df["d"],
                y1=df["tuning_similarity"],
                x2=df_recorded["d"],
                y2=df_recorded["tuning_similarity"],
                model_name=model_name,
                filename=filename,
            )
        )
        print("\t\tLIKELIHOOD RATIO TEST: ")
        print(f"\t\t\tLRT Statistic: {lrt_stat:.04e}")
        print(f"\t\t\tOverall p-value: {overall_p_value:.03f}")
    else:
        df = df.groupby(["d", "v"])["tuning_similarity"].mean().reset_index()
        df["tuning_similarity"] = (
            df["tuning_similarity"] - df["tuning_similarity"].mean()
        ) / df["tuning_similarity"].std()
        df_recorded = (
            df_recorded.groupby(["d", "v"])["tuning_similarity"].mean().reset_index()
        )
        df_recorded["tuning_similarity"] = (
            df_recorded["tuning_similarity"] - df_recorded["tuning_similarity"].mean()
        ) / df_recorded["tuning_similarity"].std()
        overall_p_value = ttest_rel(
            df["tuning_similarity"],
            df_recorded["tuning_similarity"],
        ).pvalue
        print(f"\t\tPAIRED T-TEST: {overall_p_value:.04e}")

        overall_p_value = ttest_ind(
            df["tuning_similarity"],
            df_recorded["tuning_similarity"],
        ).pvalue
        print(f"\t\tUNPAIRED T-TEST: {overall_p_value:.04e}")

    linewidth = 1.5
    axin = inset_axes(
        ax,
        width="50%",
        height="60%",
        loc="upper right",
        borderpad=0.0,
    )
    min_value = np.inf
    max_value = -np.inf
    for label_name, model_df in [("recorded", df_recorded), ("predicted", df)]:
        values = model_df.groupby("d")["tuning_similarity"].mean()
        if num_mice > 1:
            model_df = (
                model_df.groupby(["d", "mouse"])["tuning_similarity"]
                .mean()
                .reset_index()
            )
        error_bars = model_df.groupby("d")["tuning_similarity"].sem()
        min_value = min(min_value, np.min(values - error_bars))
        max_value = max(max_value, np.max(values + error_bars))
        color = plot.get_color(model_name if label_name != "recorded" else "recorded")
        axin.plot(
            x_ticks,
            values,
            linestyle="-",
            linewidth=linewidth,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            color=color,
            alpha=0.7,
            markeredgecolor=None,
            clip_on=False,
            label=label_name.capitalize(),
        )
        axin.errorbar(
            x=x_ticks,
            y=values,
            yerr=error_bars,
            linestyle="",
            marker=None,
            elinewidth=linewidth,
            capsize=2,
            capthick=linewidth,
            color=color,
            alpha=0.8,
            clip_on=False,
        )

    max_value = np.ceil(max_value)
    min_value = np.floor(min_value)

    y_ticks = np.array([min_value, max_value], dtype=np.float32)
    axin.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axin,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label="z-scored\nsimilarity",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-9,
        linespacing=0.9,
    )
    axin.yaxis.set_minor_locator(MultipleLocator(1))
    axin.set_xlim(x_ticks[0] - 15, x_ticks[-1])
    plot.set_xticks(
        axin,
        ticks=x_ticks,
        tick_labels=[
            str(x_ticks[i].astype(int)) if i % 2 == 1 else ""
            for i in range(len(x_ticks))
        ],
        label="d",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    legend = axin.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=1,
        fontsize=TICK_FONTSIZE,
        frameon=False,
        alignment="left",
        handletextpad=0.3,
        handlelength=0.85,
        labelspacing=0.05,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    axin.add_artist(legend)
    # show p-value between recorded and predicted
    if overall_p_value is not None:
        bbox = legend.get_window_extent()
        (x0, y0), (x1, y1) = axin.transData.inverted().transform(bbox)
        left = x0 - 0.28 * np.max(x_ticks)
        right = left + 5
        top, bottom = y1 * 0.94, y1 * 0.74
        axin.plot(
            [right, left, left, right],
            [top, top, bottom, bottom],
            color="black",
            linewidth=1.0,
            clip_on=False,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )
        p_text = plot.get_p_value_asterisk(overall_p_value)
        axin.text(
            x=left - 1,
            y=0.5 * (top - bottom) + bottom,
            s=p_text,
            ha="right",
            va="center",
            rotation=90,
            fontsize=TICK_FONTSIZE,
            transform=axin.transData,
        )
    sns.despine(ax=axin)
    plot.set_ticks_params(axin, pad=1)


def plot_tuning_similarity(
    df: pd.DataFrame,
    df_recorded: pd.DataFrame | None,
    filename: Path,
    random_state: np.random.Generator | None = None,
    use_legend: bool = True,
    model_name: str = None,
    title: str = None,
    y_label: str = "Tuning similarity",
):
    # statistic test on each distance group in the same plane against the
    # furthest group which is used to set the dot size
    p_values = rank_sum_test(df)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 2) * PAPER_WIDTH, 1.6),
        dpi=DPI,
    )
    horizontal_distances = sorted(df["d"].unique())
    vertical_distances = sorted(df["v"].unique())
    color = plot.get_color(model_name)
    colors = [
        lighten_color(color, amount)
        for amount in np.linspace(0, 0.8, len(vertical_distances))
    ]

    linewidth, marker = 1.5, "o"
    for i, delta_v in enumerate(vertical_distances):
        color = colors[i]
        values = df[df["v"] == delta_v].groupby("d")["tuning_similarity"].mean()
        error_bars = df[df["v"] == delta_v].groupby("d")["tuning_similarity"].sem()
        zorder = len(vertical_distances) - i
        ax.plot(
            horizontal_distances,
            values,
            linestyle="-",
            linewidth=linewidth,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            color=color,
            alpha=0.7,
            marker=None,
            markeredgecolor=None,
            clip_on=False,
            zorder=zorder,
            label=rf"{delta_v}$\leq\Delta<${delta_v + V_INTERVAL}",
        )
        # p_value dot
        ax.scatter(
            horizontal_distances,
            values,
            s=[get_dot_size(p_value) for p_value in p_values[delta_v]],
            color=color,
            marker=marker,
            edgecolors="none",
            alpha=0.8,
            zorder=zorder,
            clip_on=False,
        )
        ax.errorbar(
            x=horizontal_distances,
            y=values,
            yerr=error_bars,
            linestyle="",
            marker=None,
            elinewidth=linewidth,
            capsize=2,
            capthick=linewidth,
            color=color,
            alpha=0.7,
            clip_on=False,
            zorder=zorder,
        )

    if random_state is not None:
        ax.axhline(
            y=df.tuning_similarity.sample(n=5000, random_state=random_state).mean(),
            color=plot.get_color("chance"),
            linestyle="dotted",
            dashes=(1, 1),
            alpha=0.7,
            zorder=-1,
            linewidth=linewidth,
            label="Chance",
            clip_on=False,
        )

    min_value, max_value = 0, 0.4

    y_ticks = np.array([min_value, max_value], dtype=np.float32)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label=y_label,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    x_ticks = np.array(horizontal_distances, dtype=np.float32)
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=[
            str(x_ticks[i].astype(int)) if i % 2 == 1 else ""
            for i in range(len(x_ticks))
        ],
        label=r"Cortical distance ($\it{d}$) (µm)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    ax.set_xlim(x_ticks[0] - 10, x_ticks[-1])

    if use_legend:
        # Legend for Δ (µm) vertical distance
        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(90, max_value),
            bbox_transform=ax.transData,
            ncols=1,
            fontsize=TICK_FONTSIZE,
            title_fontsize=TICK_FONTSIZE,
            frameon=False,
            title=r"$\Delta$ ($\mu$m)",
            alignment="left",
            handletextpad=0.3,
            handlelength=0.85,
            labelspacing=0.05,
            columnspacing=0,
            borderpad=0,
            borderaxespad=0,
        )
        for lh in legend.legend_handles:
            lh.set_alpha(1)
        for text in legend.texts:
            text.set_y(-0.5)
        ax.add_artist(legend)

        # Legend of dot sizes (i.e. p-value)
        custom_legend = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color="black",
                markerfacecolor="black",
                markersize=np.sqrt(get_dot_size(p_value)) * 0.8,
                linestyle="None",
                label=label,
            )
            for p_value, label in [(0.001, "P<0.005"), (0.04, "P<0.05"), (1, "NS")]
        ]
        ax.legend(
            handles=custom_legend,
            loc="upper left",
            bbox_to_anchor=(160, max_value),
            bbox_transform=ax.transData,
            ncols=1,
            fontsize=TICK_FONTSIZE,
            title_fontsize=TICK_FONTSIZE,
            frameon=False,
            alignment="left",
            handletextpad=0.4,
            handlelength=0.7,
            labelspacing=0.2,
            markerscale=1,
            columnspacing=0.5,
            borderpad=0,
            borderaxespad=0,
        )

    if df_recorded is not None:
        add_inset_plot(
            ax=ax,
            df=df,
            df_recorded=df_recorded,
            model_name=model_name,
            x_ticks=x_ticks,
            filename=filename.parent / f"exp_fit.{FORMAT}",
        )

    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    sns.despine(ax=ax)
    plot.set_ticks_params(ax, length=3, minor_length=3)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_tuning_similarity_comparison(
    tuning_similarities: dict[str, pd.DataFrame], filename: Path
):
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, (1 / 3) * PAPER_WIDTH),
        dpi=DPI,
    )

    model_names = list(tuning_similarities.keys())
    horizontal_distances = sorted(tuning_similarities[model_names[0]].d.unique())

    linewidth = 1.5
    min_value = np.inf
    max_value = -np.inf
    for model_name, tuning_similarity in tuning_similarities.items():
        tuning_similarity = (
            tuning_similarity.groupby(["d", "mouse"])["tuning_similarity"]
            .mean()
            .reset_index()
        )
        values = tuning_similarity.groupby("d")["tuning_similarity"].mean()
        error_bars = tuning_similarity.groupby("d")["tuning_similarity"].sem()
        min_value = min(min_value, np.min(values - error_bars))
        max_value = max(max_value, np.max(values + error_bars))
        color = plot.get_color(model_name)
        zorder = plot.get_zorder(model_name)
        ax.plot(
            horizontal_distances,
            values,
            linestyle="-",
            linewidth=linewidth,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            color=color,
            alpha=0.6,
            markeredgecolor=None,
            clip_on=False,
            label=model_name,
            zorder=zorder,
        )
        ax.errorbar(
            x=horizontal_distances,
            y=values,
            yerr=error_bars,
            linestyle="",
            marker="o",
            markersize=5,
            markeredgecolor="none",
            elinewidth=linewidth,
            capsize=2.5,
            capthick=linewidth,
            color=color,
            alpha=0.8,
            clip_on=False,
            zorder=zorder,
        )
    max_value = 0.1 * np.ceil(max_value * 10)

    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=1,
        fontsize=LABEL_FONTSIZE,
        frameon=False,
        alignment="left",
        handletextpad=0.3,
        handlelength=0.8,
        labelspacing=0.05,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    y_ticks = np.array([min_value, max_value], dtype=np.float32)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label="Tuning similarity",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-7,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    x_ticks = np.array(horizontal_distances, dtype=np.float32)
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=[
            str(x_ticks[i].astype(int)) if i % 2 == 1 else ""
            for i in range(len(x_ticks))
        ],
        label="Cortical distance (d) (µm)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    ax.set_xlim(x_ticks[0] - 10, x_ticks[-1])
    sns.despine(ax=ax)
    plot.set_ticks_params(ax, length=3, pad=2, minor_length=3)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def main():
    models = {
        "recorded": data.METADATA_DIR,
        "LN": Path("../runs/fCNN/036_linear_fCNN"),
        "fCNN": Path("../runs/fCNN/038_fCNN"),
        "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }
    mouse_ids = ["B", "C", "E"]
    tuning_types = ["orientation"]

    rng = np.random.default_rng(1234)

    df_recorded = pd.read_parquet(
        data.METADATA_DIR / "tuning" / "tuning_similarity.parquet"
    )

    for tuning_type in tuning_types:
        tuning_similarities = {}
        for model_name, output_dir in models.items():
            filename = output_dir / "tuning" / "tuning_similarity.parquet"
            if not filename.exists():
                print(f"Cannot find {filename}.")
                continue
            print(f"\nProcessing model {model_name} from {filename}...")
            tuning_similarity = pd.read_parquet(filename)
            tuning_similarity = tuning_similarity[
                tuning_similarity.tuning_type == tuning_type
            ]
            tuning_similarities[model_name] = tuning_similarity
            for mouse_id in mouse_ids:
                print(f"Processing mouse {mouse_id} {tuning_type}...")
                plot_tuning_similarity(
                    df=tuning_similarity[(tuning_similarity.mouse == mouse_id)],
                    df_recorded=(
                        None
                        if model_name == "recorded"
                        else df_recorded[(df_recorded.mouse == mouse_id)]
                    ),
                    filename=PLOT_DIR
                    / tuning_type
                    / model_name
                    / f"mouse{mouse_id}.{FORMAT}",
                    random_state=rng,
                    use_legend=model_name == "recorded",
                    model_name=model_name,
                )
            print("Combine neuron pairs from all mice.")
            plot_tuning_similarity(
                df=tuning_similarity,
                df_recorded=None if model_name == "recorded" else df_recorded,
                filename=PLOT_DIR
                / tuning_type
                / model_name
                / f"{model_name}_{tuning_type}_similarity.{FORMAT}",
                random_state=rng,
                use_legend=model_name == "recorded",
                model_name=model_name,
            )
        print(f"Plot model comparison for {tuning_type}...")
        plot_tuning_similarity_comparison(
            tuning_similarities=tuning_similarities,
            filename=PLOT_DIR
            / tuning_type
            / f"{tuning_type}_tuning_similarity_comparison.{FORMAT}",
        )


if __name__ == "__main__":
    main()
