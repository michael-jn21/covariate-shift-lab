from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from covshift.experiments import run_alpha_sweep


def main() -> None:
    output_dir = Path("results")
    figures_dir = output_dir / "figures"

    output_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    p = 40
    n_source = 600
    m_target = 100
    repetitions = 300
    noise_std = 1.0

    alphas = np.linspace(0.0, 3.0, 25)

    df = run_alpha_sweep(
        alphas=alphas,
        n_source=n_source,
        m_target=m_target,
        p=p,
        repetitions=repetitions,
        noise_std=noise_std,
        seed=123,
    )

    df.to_csv(output_dir / "alpha_sweep_results.csv", index=False)

    summary = (
        df.groupby(["alpha", "estimator"])["excess_risk"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = 1.96 * summary["se"]

    theory = df.groupby("alpha", as_index=False)[
        ["theoretical_T", "population_pooled_proxy"]
    ].first()

    fig, (comparison_ax, theory_ax) = plt.subplots(
        ncols=2, figsize=(11, 5), sharey=True
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.76, wspace=0.12)

    styles = {
        "target_only": ("Target data only", "#3B5267"),
        "pooled": ("Source + target data", "#116D76"),
    }
    for ax, estimators in (
        (comparison_ax, ("target_only", "pooled")),
        (theory_ax, ("pooled",)),
    ):
        for estimator in estimators:
            subset = summary[summary["estimator"] == estimator]
            label, color = styles[estimator]
            ax.plot(
                subset["alpha"],
                subset["mean"],
                marker="o",
                markersize=3,
                linewidth=2,
                color=color,
                label=label,
            )
            ax.fill_between(
                subset["alpha"],
                subset["mean"] - subset["ci95"],
                subset["mean"] + subset["ci95"],
                color=color,
                alpha=0.16,
            )

    theory_ax.plot(
        theory["alpha"],
        theory["theoretical_T"],
        linestyle="--",
        color="#984E23",
        linewidth=1.8,
        label=r"Theoretical $T$",
    )
    theory_ax.plot(
        theory["alpha"],
        theory["population_pooled_proxy"],
        linestyle=":",
        color="#655192",
        linewidth=2,
        label="Population approximation",
    )

    comparison_ax.set_title("Comparison of the fitted models", loc="left", fontsize=12)
    theory_ax.set_title("Pooled risk and theory", loc="left", fontsize=12)
    comparison_ax.set_ylabel("Excess prediction error on target data")
    comparison_ax.set_ylim(bottom=0)
    for ax in (comparison_ax, theory_ax):
        ax.set_xticks(range(4))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E1E6EB")
        ax.set_axisbelow(True)
    comparison_ax.legend(loc="center left", frameon=False)
    theory_ax.legend(loc="upper left", frameon=False)

    fig.suptitle(
        "Extra training data help less as the distributions diverge",
        y=0.96,
        fontsize=17,
        weight="semibold",
    )
    fig.text(
        0.5,
        0.86,
        f"{n_source} source observations, {m_target} target observations, {p} features",
        ha="center",
        fontsize=10,
        color="#435260",
    )
    fig.supxlabel(r"Source covariance decay exponent $\alpha$", y=0.14, fontsize=11)
    fig.text(
        0.5,
        0.088,
        r"At $\alpha=0$ the covariances match. "
        r"Higher $\alpha$ concentrates source variance in fewer directions.",
        ha="center",
        fontsize=9,
        color="#435260",
    )
    fig.text(
        0.5,
        0.038,
        f"Points are means over {repetitions} runs. "
        "Bands show approximate pointwise 95% confidence intervals for those means.",
        ha="center",
        fontsize=9,
        color="#435260",
    )

    figure_path = figures_dir / "alpha_sweep_pooled_vs_target.png"
    fig.savefig(figure_path, dpi=200)
    plt.show()

    print(summary)
    print(f"\nSaved results to: {output_dir / 'alpha_sweep_results.csv'}")
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
