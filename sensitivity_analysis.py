"""
sensitivity_analysis.py
=======================
Sensitivity / perturbation analysis module for the RNA binding kinetics project.

Workflow
--------
1. Create a SimulationSensitivity object (inherits from Simulation).
2. Call .run_perturbation() to scan one kinetic parameter across a range of values.
3. Inspect results with the helper functions at the bottom of this file.

Keep it simple: plots are clean line/scatter charts; no heavy statistics.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import simulation_core as sim   
import model_database as db       
import parameter_function as param  


# ---------------------------------------------------------------------------
# SimulationSensitivity  (extends Simulation)
# ---------------------------------------------------------------------------

class SimulationSensitivity(sim.Simulation):
    """
    A Simulation subclass bundling all sensitivity-analysis specific methods.

    Parameters
    ----------
    drug_conc   : list[float]   — drug concentrations to simulate
    epochs      : int           — random parameter sets per perturbation value
    drug_params : dict          — extra kinetic constants (K_hill, Vmax, …)
    k_2_COST    : float         — minimum k_2 / k_1 ratio enforced during sampling
    nodes_num   : int           — graph topology (3 or 4 nodes)
    assumption  : bool          — if True, apply biologically motivated constraints
    """

    def __init__(
        self,
        drug_conc: list[float],
        epochs: int,
        drug_params: dict,
        k_2_COST: float,
        nodes_num: int,
        assumption: bool = True,
    ):
        super().__init__(
            drug_conc     = drug_conc,
            epochs        = epochs,
            drug_params   = drug_params,
            k_2_COST      = k_2_COST,
            nodes_num     = nodes_num,
            assumption    = assumption,
        )

    # ------------------------------------------------------------------
    # Main entry point: scan one parameter, run all compatible models
    # ------------------------------------------------------------------

    def run_perturbation(
        self,
        param_name: str,
        param_values: list[float],
        model_names: list[str] | None = None,
        overrides_extra: dict | None = None,
    ) -> dict[str, list[dict]]:
        """
        Vary `param_name` across `param_values` and collect steady-state results.

        Parameters
        ----------
        param_name      : kinetic parameter to perturb
                          (any key from parameter_function.generate_parameter_sets)
        param_values    : values to test, e.g. np.logspace(-2, 3, 20).tolist()
        model_names     : subset of models to run; None = all compatible with nodes_num
        overrides_extra : fix additional parameters during the scan,
                          e.g. {"n": 2} to pin the Hill coefficient

        Returns
        -------
        dict[model_name -> list[dict]]
            Same structure as Simulation.run_all_models(), but the 'params' dict
            inside each run also carries the perturbed value for `param_name`.
        """
        return super().run_perturbation(
            param_name=param_name,
            param_values=param_values,
            model_names=model_names,
            overrides_extra=overrides_extra,
        )

    # ------------------------------------------------------------------
    # Convenience: build a DataFrame from perturbation results
    # ------------------------------------------------------------------

    @staticmethod
    def to_dataframe(
        results: dict[str, list[dict]],
        param_name: str,
        nodes_num: int,
        drug_agg: str | float = "mean",
    ) -> pd.DataFrame:
        """
        Flatten perturbation results into a tidy DataFrame.

        Each row = one (model, param_value) combination.
        Probability columns are averaged (or selected) over drug concentrations
        according to `drug_agg`.

        Parameters
        ----------
        results    : output of run_perturbation()
        param_name : the parameter that was varied
        nodes_num  : 3 or 4
        drug_agg   : "mean"  → average P values across all drug concentrations
                     "max"   → take the maximum
                     float   → pick the drug concentration closest to this value

        Returns
        -------
        pd.DataFrame with columns:
            model, <param_name>, P1, P2, P3[, P4]
        """
        states = [f"P{i}" for i in range(1, nodes_num + 1)]
        rows = []

        for model_name, runs in results.items():
            for run in runs:
                pval = run["params"][param_name]
                df_drug = pd.DataFrame(run["results"])

                if drug_agg == "mean":
                    agg = df_drug[states].mean()
                elif drug_agg == "max":
                    agg = df_drug[states].max()
                else:
                    target = float(drug_agg)
                    idx = (df_drug["drug"] - target).abs().idxmin()
                    agg = df_drug.loc[idx, states]

                row = {"model": model_name, param_name: pval}
                row.update(agg.to_dict())
                rows.append(row)

        df = pd.DataFrame(rows)
        # average duplicate (model, param_value) pairs that arise from multiple epochs
        df = df.groupby(["model", param_name], as_index=False)[states].mean()
        return df.sort_values([param_name])


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

# Consistent colour / style for each state across all plots
_STATE_STYLE = {
    "P1": dict(color="#94A3B8", ls="--", label="P_freeRNA"),
    "P2": dict(color="#38BDF8", ls="-",  label="P_BP_open"),
    "P3": dict(color="#E11D48", ls="-",  label="P_Decoy"),
    "P4": dict(color="#0F172A", ls="-",  label="P_BP_closed"),
}


def plot_perturbation_lines(
    results: dict[str, list[dict]],
    param_name: str,
    nodes_num: int,
    drug_agg: str | float = "mean",
    log_x: bool = True,
    title: str | None = None,
) -> None:
    """
    One subplot per model; each subplot shows P1…P(N) vs the perturbed parameter.

    This is the go-to plot for a quick overview of all models.

    Parameters
    ----------
    results    : output of SimulationSensitivity.run_perturbation()
    param_name : the parameter that was varied
    nodes_num  : graph topology (3 or 4)
    drug_agg   : how drug concentrations are collapsed
                 "mean" | "max" | float (pick nearest drug value)
    log_x      : use log scale on the x-axis (recommended for wide ranges)
    title      : optional figure title; auto-generated if None
    """
    df = SimulationSensitivity.to_dataframe(results, param_name, nodes_num, drug_agg)
    states = [f"P{i}" for i in range(1, nodes_num + 1)]
    models = df["model"].unique()

    n_cols = min(len(models), 3)
    n_rows = int(np.ceil(len(models) / n_cols))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(models):
        ax = axes_flat[idx]
        sub = df[df["model"] == model_name].sort_values(param_name)

        for state in states:
            s = _STATE_STYLE[state]
            ax.plot(sub[param_name], sub[state],
                    color=s["color"], ls=s["ls"], lw=2.2,
                    marker="o", ms=4, label=s["label"])

        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel("Probability", fontsize=10)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        clean = (model_name.replace("[", ": ").replace("]", "")
                           .replace("__", "  |  ").title())
        ax.set_title(clean, fontsize=10, fontweight="bold", color="#2C3E50", pad=8)
        ax.grid(True, ls=":", alpha=0.5)

    # legend from first subplot
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=nodes_num,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.04))

    # hide empty subplots
    for j in range(len(models), len(axes_flat)):
        axes_flat[j].axis("off")

    agg_label = f"drug≈{drug_agg:.2g}" if isinstance(drug_agg, float) else f"drug agg: {drug_agg}"
    suptitle = title or f"Sensitivity — {param_name}  ({agg_label})"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", color="#1A252F", y=1.02)

    plt.tight_layout()
    plt.show()


def plot_perturbation_heatmap(
    results: dict[str, list[dict]],
    param_name: str,
    nodes_num: int,
    drug_agg: str | float = "mean",
    title: str | None = None,
) -> None:
    """
    Heatmap of probability vs perturbed parameter — one panel per state.

    Rows = models, columns = states (P1…PN).
    Colour encodes probability (0 → 1).

    Good for comparing many models at a glance without reading individual lines.

    Parameters
    ----------
    (same as plot_perturbation_lines)
    """
    df = SimulationSensitivity.to_dataframe(results, param_name, nodes_num, drug_agg)
    states = [f"P{i}" for i in range(1, nodes_num + 1)]
    models = list(df["model"].unique())
    param_vals = sorted(df[param_name].unique())

    _CMAPS = {"P1": "Greys", "P2": "Blues", "P3": "Reds", "P4": "Purples"}
    _LABELS = {"P1": "P_freeRNA", "P2": "P_BP_open",
               "P3": "P_Decoy",   "P4": "P_BP_closed"}

    n_states = len(states)
    fig, axes = plt.subplots(len(models), n_states,
                             figsize=(4.5 * n_states, 3 * len(models)),
                             squeeze=False)

    for r, model_name in enumerate(models):
        sub = df[df["model"] == model_name].set_index(param_name)

        for c, state in enumerate(states):
            ax = axes[r, c]
            # build a (1 × n_param_vals) matrix so imshow gives a colour bar strip
            vals = np.array([sub.loc[v, state] if v in sub.index else np.nan
                             for v in param_vals]).reshape(1, -1)

            im = ax.imshow(vals, aspect="auto", cmap=_CMAPS[state],
                           vmin=0, vmax=1)

            tick_step = max(1, len(param_vals) // 6)
            ax.set_xticks(range(0, len(param_vals), tick_step))
            ax.set_xticklabels(
                [f"{param_vals[i]:.2g}" for i in range(0, len(param_vals), tick_step)],
                rotation=45, ha="right", fontsize=8)
            ax.set_yticks([])

            if r == 0:
                ax.set_title(_LABELS.get(state, state), fontsize=10,
                             fontweight="bold", color="#2C3E50")
            if c == 0:
                clean = (model_name.replace("[", ": ").replace("]", "")
                                   .replace("__", " | ").title())
                ax.set_ylabel(clean, fontsize=8, color="#555")

            ax.set_xlabel(param_name, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04).ax.tick_params(labelsize=7)

    agg_label = f"drug≈{drug_agg:.2g}" if isinstance(drug_agg, float) else drug_agg
    suptitle = title or f"Sensitivity heatmap — {param_name}  ({agg_label})"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_sensitivity_index(
    results: dict[str, list[dict]],
    param_name: str,
    nodes_num: int,
    drug_agg: str | float = "mean",
    title: str | None = None,
) -> None:
    """
    Bar chart of a simple sensitivity index per model.

    The index is the total variation (max − min) of each state probability
    as `param_name` is scanned.  Higher = more sensitive to this parameter.

    Useful for quickly ranking which models / states respond most to the
    parameter being perturbed.

    Parameters
    ----------
    (same as plot_perturbation_lines)
    """
    df = SimulationSensitivity.to_dataframe(results, param_name, nodes_num, drug_agg)
    states = [f"P{i}" for i in range(1, nodes_num + 1)]
    models = df["model"].unique()

    records = []
    for model_name in models:
        sub = df[df["model"] == model_name]
        for state in states:
            tv = sub[state].max() - sub[state].min()
            records.append({
                "model": model_name,
                "state": state,
                "total_variation": tv,
            })
    df_tv = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    x = np.arange(len(models))
    width = 0.8 / len(states)

    for i, state in enumerate(states):
        s = _STATE_STYLE[state]
        vals = [
            df_tv.loc[(df_tv["model"] == m) & (df_tv["state"] == state),
                      "total_variation"].values[0]
            for m in models
        ]
        offset = (i - len(states) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=s["color"], label=s["label"], alpha=0.85)

    clean_models = [
        m.replace("[", ": ").replace("]", "").replace("__", "\n").title()
        for m in models
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_models, fontsize=9)
    ax.set_ylabel("Total variation  (max P − min P)", fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", ls=":", alpha=0.5)

    suptitle = title or f"Sensitivity index — {param_name}"
    ax.set_title(suptitle, fontsize=12, fontweight="bold", color="#1A252F", pad=10)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Simple numerical summary
# ---------------------------------------------------------------------------

def sensitivity_summary(
    results: dict[str, list[dict]],
    param_name: str,
    nodes_num: int,
    drug_agg: str | float = "mean",
) -> pd.DataFrame:
    """
    Return a DataFrame with per-(model, state) sensitivity statistics.

    Columns
    -------
    model, state, min_P, max_P, total_variation, param_at_min, param_at_max

    Use this to quickly identify which models are most sensitive to the
    parameter being perturbed, and at which parameter value each state peaks.

    Parameters
    ----------
    (same as plot_perturbation_lines)
    """
    df = SimulationSensitivity.to_dataframe(results, param_name, nodes_num, drug_agg)
    states = [f"P{i}" for i in range(1, nodes_num + 1)]

    rows = []
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name].sort_values(param_name)
        for state in states:
            rows.append({
                "model":          model_name,
                "state":          state,
                "min_P":          sub[state].min(),
                "max_P":          sub[state].max(),
                "total_variation": sub[state].max() - sub[state].min(),
                "param_at_min":   sub.loc[sub[state].idxmin(), param_name],
                "param_at_max":   sub.loc[sub[state].idxmax(), param_name],
            })

    return pd.DataFrame(rows).sort_values("total_variation", ascending=False)


# ---------------------------------------------------------------------------
# Quick demo / entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Minimal end-to-end example — run this file directly to test the module.

    Adjust the parameters below to match your current analysis.
    """
    # ── 1. Simulation setup ──────────────────────────────────────────────────
    sim = SimulationSensitivity(
        drug_conc   = np.linspace(0, 10, 20).tolist(),
        epochs      = 5,
        drug_params = {"K_hill": 2.0, "Vmax": 5.0},
        k_2_COST    = 100,
        nodes_num   = 3,
        assumption  = True,
    )

    # ── 2. Perturbation run ──────────────────────────────────────────────────
    alpha_values = np.logspace(-1, 2, 12).tolist()   # scan alpha from 0.1 to 100

    results = sim.run_perturbation(
        param_name   = "alpha",
        param_values = alpha_values,
        # model_names = ["k_1[linear_increase]__3nodes"],  # uncomment to restrict models
    )

    # ── 3. Numerical summary ─────────────────────────────────────────────────
    summary = sensitivity_summary(results, "alpha", nodes_num=3)
    print("\nSensitivity summary:")
    print(summary.to_string(index=False))

    # ── 4. Visualisations ────────────────────────────────────────────────────
    plot_perturbation_lines(results, "alpha", nodes_num=3)
    plot_sensitivity_index(results, "alpha", nodes_num=3)


if __name__ == "__main__":
    main()
