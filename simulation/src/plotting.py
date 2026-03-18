import matplotlib.pyplot as plt
import numpy as np


def _apply_style():
    plt.style.use("seaborn-v0_8-whitegrid")


def plot_power_vs_N(results_by_profile):
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for profile_name, results in results_by_profile.items():
        x_values = [item["N"] for item in results]
        y_values = [item["obj"] for item in results]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=profile_name)

    ax.set_title("Optimal Power vs Number of Stages")
    ax.set_xlabel("Number of stages (N)")
    ax.set_ylabel("Optimal objective value")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_power_vs_Tclk(results_by_profile):
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for profile_name, results in results_by_profile.items():
        x_values = [item["Tclk"] for item in results]
        y_values = [item["obj"] for item in results]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=profile_name)

    ax.set_title("Optimal Power vs Timing Budget")
    ax.set_xlabel("Clock period (Tclk)")
    ax.set_ylabel("Optimal objective value")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_width_profile(results_by_profile, title="Optimal Width Allocation"):
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for profile_name, result in results_by_profile.items():
        if result["W"] is None:
            continue
        x_values = np.arange(1, len(result["W"]) + 1)
        ax.plot(x_values, result["W"], marker="o", linewidth=2, label=profile_name)

    ax.set_title(title)
    ax.set_xlabel("Stage index")
    ax.set_ylabel("Optimal width")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_full_vs_baseline(
    comparison_results,
    full_width_result,
    baseline_width_result,
):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x_values = [item["Tclk"] for item in comparison_results]
    full_values = [item["full"]["obj"] for item in comparison_results]
    baseline_values = [item["baseline"]["obj"] for item in comparison_results]

    axes[0].plot(x_values, full_values, marker="o", linewidth=2, label="Full model")
    axes[0].plot(x_values, baseline_values, marker="s", linewidth=2, label="Boyd-style baseline")
    axes[0].set_title("Objective Comparison Across Timing Budgets")
    axes[0].set_xlabel("Clock period (Tclk)")
    axes[0].set_ylabel("Optimal objective value")
    axes[0].legend()

    if full_width_result["W"] is not None:
        stage_index = np.arange(1, len(full_width_result["W"]) + 1)
        axes[1].plot(stage_index, full_width_result["W"], marker="o", linewidth=2, label="Full model")
    if baseline_width_result["W"] is not None:
        stage_index = np.arange(1, len(baseline_width_result["W"]) + 1)
        axes[1].plot(
            stage_index,
            baseline_width_result["W"],
            marker="s",
            linewidth=2,
            label="Boyd-style baseline",
        )
    axes[1].set_title("Width Allocation Comparison")
    axes[1].set_xlabel("Stage index")
    axes[1].set_ylabel("Optimal width")
    if axes[1].lines:
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "No feasible width comparison for the selected case",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    fig.tight_layout()
    return fig, axes


def plot_free_length_profile(free_primal_result):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    stage_index = np.arange(1, len(free_primal_result["W"]) + 1)
    axes[0].plot(stage_index, free_primal_result["W"], marker="o", linewidth=2)
    axes[0].set_title("Free-Length Primal: Optimal Widths")
    axes[0].set_xlabel("Stage index")
    axes[0].set_ylabel("Optimal width")

    axes[1].plot(stage_index, free_primal_result["L"], marker="o", linewidth=2, color="#d97706")
    axes[1].set_title("Free-Length Primal: Optimal Wire Lengths")
    axes[1].set_xlabel("Stage index")
    axes[1].set_ylabel("Optimal wire length")

    fig.tight_layout()
    return fig, axes


def plot_free_length_primal_dual(results):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    x_values = [item["Tclk"] for item in results]
    primal_values = [item["primal"]["log_obj_unscaled"] for item in results]
    dual_values = [item["dual"]["dual_log_obj_unscaled"] for item in results]
    gap_values = [item["log_gap"] for item in results]

    axes[0].plot(x_values, primal_values, marker="o", linewidth=2, label="Primal log objective")
    axes[0].plot(x_values, dual_values, marker="s", linewidth=2, label="Dual objective")
    axes[0].set_title("Free-Length Primal vs Dual")
    axes[0].set_xlabel("Clock period (Tclk)")
    axes[0].set_ylabel("Log-domain objective")
    axes[0].legend()

    axes[1].plot(x_values, gap_values, marker="o", linewidth=2, color="#c2410c")
    axes[1].set_title("Duality Gap Across Timing Budgets")
    axes[1].set_xlabel("Clock period (Tclk)")
    axes[1].set_ylabel("Primal minus dual (log scale)")

    fig.tight_layout()
    return fig, axes
