from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    theory = (
        df.groupby("alpha")[
            [
                "theoretical_T",
                "population_pooled_proxy",
                "target_only_proxy",
            ]
        ]
        .first()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for estimator in ["target_only", "pooled"]:
        subset = summary[summary["estimator"] == estimator]

        plt.plot(
            subset["alpha"],
            subset["mean"],
            marker="o",
            label=f"{estimator} empirical mean",
        )

        plt.fill_between(
            subset["alpha"],
            subset["mean"] - subset["ci95"],
            subset["mean"] + subset["ci95"],
            alpha=0.2,
        )

    plt.plot(
        theory["alpha"],
        theory["theoretical_T"],
        linestyle="--",
        label="theoretical T(n,m,alpha,p)",
    )

    plt.plot(
        theory["alpha"],
        theory["population_pooled_proxy"],
        linestyle=":",
        label="population pooled proxy",
    )

    plt.axhline(
        y=theory["target_only_proxy"].iloc[0],
        linestyle="-.",
        label="target-only Wishart proxy",
    )

    plt.xlabel(r"$\alpha$")
    plt.ylabel("Target excess risk")
    plt.title(
        rf"Pooled vs target-only LS, $p={p}$, $n={n_source}$, $m={m_target}$"
    )
    plt.legend()
    plt.tight_layout()

    figure_path = figures_dir / "alpha_sweep_pooled_vs_target.png"
    plt.savefig(figure_path, dpi=200)
    plt.show()

    print(summary)
    print(f"\nSaved results to: {output_dir / 'alpha_sweep_results.csv'}")
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()