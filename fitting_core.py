# Standard libraries used throughout the notebook
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import itertools
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
import model_database as db
import simulation_core as sim
import warnings
from scipy.optimize import minimize


# =============================================================================
# Observed Values
# =============================================================================

def _observed_lfc(data: pd.DataFrame, vector_conc_without_control) -> dict:
    """Avoid to consider Control explicitly because the LFC is computed with respect to Ctrl."""
    behaviour = {}
    vector_means = []
    vectors_std = []

    cols = [col for col in data.columns if 'lfc_' in col and 'Ctrl' not in col]
    data = data.loc[:, cols].copy()
    for col in cols:
        lfc_mean = np.mean(data[col])
        lfc_std = np.std(data[col])
        if lfc_std == 0:
            lfc_std = 1e-11
        vector_means.append(lfc_mean)
        vectors_std.append(lfc_std)

    ps = {
        'conc':    np.asarray(vector_conc_without_control),
        'means':   np.asarray(vector_means),
        'stds':    np.asarray(vectors_std),
        'weights': 1 / np.asarray(vectors_std)**2
    }

    return ps


# =============================================================================
# Predicted Values
# =============================================================================

def _p_bound(result: dict, nodes_number: int) -> float:
    """
    P_bound = P_BP_site + P_Decoy_site (+ P_BP_closed if 4-state).

    Node map (from simulation_core):
        3-state:  P1=free  P2=BP_open   P3=decoy
        4-state:  P1=free  P2=BP_open   P3=decoy   P4=BP_closed
    """
    if nodes_number == 3:
        return result["P2"] + result["P3"]
    elif nodes_number == 4:
        return result["P2"] + result["P3"] + result["P4"]
    else:
        raise ValueError(f"Unsupported nodes_number: {nodes_number}")


def _param_array_to_dict(
    free_parameters_values: list,
    free_param_names: list,
    fixed_params: dict | None = None,
) -> dict:

    # 1. Combine fixed and free parameters into the initial dictionary
    if fixed_params is not None:
        result_dict = {**dict(zip(free_param_names, free_parameters_values)), **fixed_params}
    else:
        result_dict = dict(zip(free_param_names, free_parameters_values))

    required_keys = ['k1', 'k_1', 'k2', 'k_2', 'k3', 'k4', 'alpha', 'beta', 'delta']
    missing_keys = []

    for key in required_keys:
        if key not in result_dict:
            result_dict[key] = 0
            missing_keys.append(key)

    if missing_keys:
        message = (
            f"The following parameters were missing and have been automatically "
            f"set to 0: {', '.join(missing_keys)}"
        )
        warnings.warn(message, category=UserWarning, stacklevel=2)

    return result_dict


def _predict_lfc(
    free_parameters_values: list,
    free_param_names: list,
    drug_conc: np.array,
    model_name: str,
    fixed_params: dict | None = None,
):
    """
    Predict log fold-change (LFC) values for a given parameter set and model.

    The function applies two class-specific scalers to baseline kinetic rates
    before computing steady-state probabilities:
        - delta   scales k_2 (decoy dissociation rate) — encodes decoy strength
        - beta scales k_1 (correct-site dissociation rate) — encodes BP strength

    Both scalers are expected to lie in [0, 1] and are injected as fixed_params
    by the multiclass fitting routine. A value of 1 means no scaling (reference
    class); lower values reduce the respective rate.

    Parameters
    ----------
    free_parameters_values : list of floats
        Values of the free (shared) kinetic parameters in linear scale.
    free_param_names : list of str
        Names corresponding to free_parameters_values.
    drug_conc : np.array
        Drug concentrations to evaluate (must include 0 or a near-zero value
        as the control reference).
    model_name : str
        Key identifying the drug-modulation model in db.ALL_MODELS.
    fixed_params : dict or None
        Parameters that are not optimised (e.g. class-specific scalers
        injected by the multiclass wrapper).

    Returns
    -------
    np.array
        LFC values at each non-control drug concentration.
    """

    drug_conc = np.sort(drug_conc.copy())
    if np.isclose(drug_conc[0], 0):
        drug_conc = drug_conc[1:]

    params = _param_array_to_dict(free_parameters_values, free_param_names, fixed_params)


    # Both scalers are applied multiplicatively so that a value of 1 = no effect.
    params['k_2'] = params['delta']   * params['k_2']

    params['k_1'] = params['beta']    * params['k_1']

    mod = None

    # Assume the uniqueness of the model ## WRONG !!! Same model, different topologies 
    for model in db.ALL_MODELS:
        if model_name in model:
            mod = model
            break
    if mod is None:
        raise ValueError(
            f"The model: '{model_name}' is not in the database. "
            f"Please check available models with: db.list_models()"
        )
    # Retrieve the model info (check in the model_database module for more information)
    model_info  = db.ALL_MODELS[mod]
    mods        = model_info['mods']
    model_nodes = model_info['nodes_number']

    # Bound probabilities at control (drug = 0)
    res_Ctrl = sim._steady_state(
        param_set   = params,
        drug        = 0.,
        mods        = mods,
        drug_params = {},
        nodes_number= model_nodes,
    )
    bound_probabilites_Ctrl = _p_bound(result=res_Ctrl, nodes_number=model_nodes)

    # Bound probabilities at each drug concentration
    bound_probabilites_drug = []
    for d in drug_conc:
        res_drug = sim._steady_state(
            param_set   = params,
            drug        = d,
            mods        = mods,
            drug_params = {},
            nodes_number= model_nodes,
        )
        bound_probabilites_drug.append(_p_bound(result=res_drug, nodes_number=model_nodes))

    lfc_pred = np.log(np.asarray(bound_probabilites_drug) / np.asarray(bound_probabilites_Ctrl))
    return lfc_pred


# =============================================================================
# Single-class loss
# =============================================================================

def _weighted_mse(
    log_theta,
    free_param_names,
    drug_conc: np.array,
    lfc_obs: np.array,
    weights: np.array,
    model_name: str,
    fixed_params: None | dict = None
):
    """Scalar loss — operates in log10(θ) space."""
    theta    = 10.0 ** log_theta
    lfc_pred = _predict_lfc(
        free_parameters_values = theta,
        free_param_names       = free_param_names,
        drug_conc              = drug_conc,
        model_name             = model_name,
        fixed_params           = fixed_params,
    )

    if np.any(~np.isfinite(lfc_pred)):
        return 1e10

    residuals = lfc_obs - lfc_pred
    return float(np.sum(weights * residuals**2) / weights.sum())


# =============================================================================
# MULTICLASS LOSS
# =============================================================================

def _weighted_mse_multiclass(
    log_theta,
    free_param_names: list,
    drug_conc: np.array,
    obs_list: list,    
    class_names: list,      
    model_name: str,
    fixed_params: None | dict = None,
):
    """
    Compute a scalar weighted-MSE loss summed over all structural classes.

    The parameter vector log_theta is partitioned into two segments:
      1. Shared kinetic parameters (in log10 scale, converted to linear internally).
      2. Class-specific scalers delta and beta (already in linear [0,1] scale,
         NOT exponentiated — they are optimised directly in their natural domain).

    For each class, delta scales k_2 (decoy strength) and beta scales k_1
    (branch-point strength). These are injected into fixed_params before calling
    _predict_lfc, so the shared kinetic scaffold remains unmodified across classes.

    The convention for identifying class-specific parameters relies on their name
    prefix: any parameter whose name starts with 'delta_' or 'beta_' is treated
    as class-specific. All others are treated as shared.

    Parameters
    ----------
    log_theta : np.array
        Flattened optimisation vector. First n_shared entries are log10(shared params);
        remaining entries are linear class-specific scalers ordered as
        [delta_class0, beta_class0, delta_class1, beta_class1, ...].
    free_param_names : list of str
        Full list of parameter names, in the same order as log_theta.
        Built automatically by multistart_fit_multiclass.
    drug_conc : np.array
        Drug concentrations (including control at 0).
    obs_list : list of dict
        One observed-LFC dict per class (output of _observed_lfc).
        Must be aligned with class_names.
    class_names : list of str
        Class identifiers (e.g. ['strongDecoy', 'noDecoy']).
        Used to look up the correct scaler values in log_theta.
    model_name : str
        Drug-modulation model key in db.ALL_MODELS.
    fixed_params : dict or None
        Any additional fixed parameters common to all classes.

    Returns
    -------
    float
        Sum of per-class weighted MSE values (penalised to 1e10 on failure).
    """

    # Identify shared vs class-specific parameters by name prefix.

    class_specific_names = [
        p for p in free_param_names
        if p.startswith('delta_') or p.startswith('beta_')
    ]
    shared_names = [p for p in free_param_names if p not in class_specific_names]
    n_shared     = len(shared_names)

    # Shared parameters: log10 → linear
    shared_values = 10.0 ** log_theta[:n_shared]

    # Class-specific scalers: already linear, just slice them out by index.
    def _build_class_params(class_name: str) -> dict:
        """
        Inject the two class-specific scalers for `class_name` into fixed_params.
        Scalers are read directly from log_theta (no exponentiation) because
        they are bounded to [0, 1] and optimised in linear space.
        """
        fp = dict(fixed_params) if fixed_params is not None else {}
        fp['delta']   = log_theta[free_param_names.index(f'delta_{class_name}')]
        fp['beta'] = log_theta[free_param_names.index(f'beta_{class_name}')]
        return fp

    # Compute per-class predictions and accumulate loss.
    total_loss = 0.0
    try:
        for obs, cname in zip(obs_list, class_names):
            fp_class = _build_class_params(cname)
            lfc_pred = _predict_lfc(
                free_parameters_values = shared_values,
                free_param_names       = shared_names,
                drug_conc              = drug_conc.copy(),
                model_name             = model_name,
                fixed_params           = fp_class,
            )
            if np.any(~np.isfinite(lfc_pred)):
                return 1e10

            residuals   = obs['means'] - lfc_pred
            class_loss  = np.sum(obs['weights'] * residuals**2) / obs['weights'].sum()
            total_loss += class_loss        # adding loss across all the classes -> global fitting

    except Exception:
        return 1e10

    return float(total_loss)


# =============================================================================
# MULTICLASS MULTI-START OPTIMIZER
# =============================================================================

def multistart_fit_multiclass(
    classes_data: list,       
    class_names: list,     
    shared_param_names: list,
    drug_conc: np.ndarray,
    model_name: str,
    fixed_params: dict | None = None,
    n_starts: int = 100,
    log_bounds: tuple = (-3, 3),
    show_messages: bool = True,
    show_history: bool = False,
    show_dict: bool = False,
):
    """
    Multi-start gradient-based fitting over an arbitrary number of structural classes.

    Each class shares a common set of kinetic parameters (rates, drug-effect
    parameters) but has two private scalers:
      - delta   in [0, 1]: scales k_2, encoding decoy-site strength.
      - beta in [0, 1]: scales k_1, encoding branch-point retention strength.

    The optimisation vector is partitioned as:
        [ log10(shared_params) | delta_c0, beta_c0, delta_c1, beta_c1, ... ]

    Shared parameters are optimised in log10 space (bounds: log_bounds).
    Class-specific scalers are optimised in linear space (bounds: [0, 1]).

    The loss is the sum of per-class inverse-variance-weighted MSE between
    observed and predicted LFC values across all drug concentrations.

    Parameters
    ----------
    classes_data : list of pd.DataFrame
        One DataFrame per class; each must contain lfc_* columns (excluding Ctrl).
        Must be aligned positionally with class_names.
    class_names : list of str
        Identifiers for each class. Used to name the class-specific parameters
        (e.g. class_name='strongDecoy' → 'delta_strongDecoy', 'beta_strongDecoy').
    shared_param_names : list of str
        Names of kinetic parameters shared across all classes (e.g. ['k1', 'k_1', ...]).
        Do NOT include delta or beta here — they are added automatically.
    drug_conc : np.ndarray
        Drug concentrations to fit (including 0 as control reference).
    model_name : str
        Key of the drug-modulation model in db.ALL_MODELS.
    fixed_params : dict or None
        Any additional parameters held constant during optimisation (e.g. Hill
        coefficient n, or any rate you want to pin).
    n_starts : int
        Number of random restarts for the multi-start routine.
    log_bounds : tuple of (float, float)
        Search bounds in log10 space for shared kinetic parameters.
    show_messages : bool
        If True, print a summary of the best fit after the run.
    show_history : bool
        If True, append the full history DataFrame to the return value.
    show_dict : bool
        If True, append the best-parameter dictionary to the return value.

    Returns
    -------
    scipy.optimize.OptimizeResult, or tuple thereof with history and/or best_dict
    depending on show_history and show_dict flags.
    """

    # Build the full parameter name list: shared first, then class-specific pairs.
    # Order within each class: delta first, then beta — must match _build_class_params.
    class_specific_names = [
        f'{scaler}_{name}'
        for name in class_names
        for scaler in ('delta', 'beta')
    ]
    full_param_names = shared_param_names + class_specific_names

    # Compute observed LFC once, outside the optimisation loop.
    obs_list = [_observed_lfc(df, drug_conc) for df in classes_data]

    # Bounds: log space for shared params, linear [0,1] for class-specific scalers.
    bounds = []
    for name in shared_param_names:
        if name == 'alpha':
            bounds.append((0.0, 2.0))   # log10(100) = 2  →  alpha max = 100
        else:
            bounds.append(log_bounds)
    bounds += [(0.0, 1.0)] * len(class_specific_names)


    best_loss   = np.inf
    best_result = None
    history     = []

    for k in range(n_starts):

        # Random starting point: log-uniform for shared, uniform [0,1] for scalers.
        log_theta_shared = np.array([
            np.random.uniform(*(0.0, 2.0) if name == 'alpha' else log_bounds)
            for name in shared_param_names
        ])
        class_specific_start = np.random.uniform(0.0, 1.0,
                                                  size=len(class_specific_names))
        x0 = np.concatenate([log_theta_shared, class_specific_start])

        result = minimize(
            fun    = _weighted_mse_multiclass,
            x0     = x0,
            args   = (full_param_names, drug_conc,
                      obs_list, class_names,
                      model_name, fixed_params),
            method = 'L-BFGS-B',
            bounds = bounds,
        )

        history.append({
            "run_id"      : k,
            "success"     : bool(result.success),
            "final_cost"  : float(result.fun),
            "params_start": x0.tolist(),
            "params_final": result.x.tolist(),
        })

        if result.fun < best_loss:
            best_loss   = result.fun
            best_result = result

    if show_messages:
        n_sh = len(shared_param_names)
        shared_linear = 10 ** best_result.x[:n_sh]
        cs_vals       = best_result.x[n_sh:]

        print(f"\n==================== FIT COMPLETED =====================")
        print(f"Best loss   : {best_loss:.4f}")
        print(f"Convergence : {best_result.success}")
        print(f"Shared params (linear):")
        for name, val in zip(shared_param_names, shared_linear):
            print(f"  {name:12s} = {val:.4e}")
        print(f"Class-specific scalers:")
        for cname, delta, beta in zip(
            class_names,
            cs_vals[0::2],   # delta values (even positions)
            cs_vals[1::2],   # beta values (odd positions)
        ):
            print(f"  {cname:20s}  delta={delta:.4f}  beta={beta:.4f}")

    # Reconstruct best-parameter dictionary in linear scale.
    n_sh          = len(shared_param_names)
    shared_linear = 10 ** best_result.x[:n_sh]
    cs_vals       = best_result.x[n_sh:]
    best_dict     = dict(zip(shared_param_names, shared_linear))
    best_dict.update(dict(zip(class_specific_names, cs_vals)))

    output = [best_result]
    if show_history:
        output.append(pd.DataFrame(history))
    if show_dict:
        output.append(best_dict)

    if len(output) == 1:
        return output[0]
    return tuple(output)


# =============================================================================
# MULTICLASS PLOT
# =============================================================================

def plot_fit_comparison_multiclass(
    classes_data: list,
    classes_names: list,
    best_fit,
    fit_dict: dict,
    drug_conc: np.ndarray,
    real_concentrations: list,
    model_name: str,
    fixed_params: dict | None = None,
    colors: list | None = None,
    titles: list | None = None,
):
    """
    Plot observed vs predicted LFC for each class after multiclass fitting.

    For each class, the class-specific scalers (delta, beta) are extracted
    from fit_dict and injected into fixed_params before calling _predict_lfc,
    exactly mirroring the logic used during optimisation.

    Parameters
    ----------
    classes_data : list of pd.DataFrame
    classes_names : list of str
        Must match the class identifiers used during fitting.
    best_fit : OptimizeResult
    fit_dict : dict
        Best-parameter dictionary as returned by multistart_fit_multiclass
        with show_dict=True. Keys follow the convention:
        shared params in linear scale; class-specific as 'delta_{name}' and
        'beta_{name}' in [0,1].
    drug_conc : np.ndarray
    real_concentrations : list
        Human-readable concentration labels for the x-axis.
    model_name : str
    fixed_params : dict or None
    colors : list or None
    titles : list or None
    """
    n = len(classes_data)
    if colors is None:
        colors = [f'C{i}' for i in range(n)]
    if titles is None:
        titles = classes_names

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    # Identify shared parameter names: everything that is not a class-specific scaler.
    shared_names  = [k for k in fit_dict if not k.startswith('delta')
                                         and not k.startswith('beta')]
    shared_values = np.array([fit_dict[k] for k in shared_names])

    for i, (data, name) in enumerate(zip(classes_data, classes_names)):
        ax    = axes[i]
        color = colors[i]

        obs = _observed_lfc(data, drug_conc)
        ax.errorbar(
            drug_conc, obs['means'],
            yerr   = obs['stds'],
            fmt    = 'o',
            color  = 'black',
            ecolor = 'black',
            capsize= 4,
            label  = 'Observed (mean ± std)',
            zorder = 3,
        )

        fp = dict(fixed_params) if fixed_params is not None else {}
        fp['delta']   = fit_dict[f'delta_{name}']
        fp['beta'] = fit_dict[f'beta_{name}']

        lfc_pred = _predict_lfc(
            free_parameters_values = shared_values,
            free_param_names       = shared_names,
            drug_conc              = drug_conc.copy(),
            model_name             = model_name,
            fixed_params           = fp,
        )

        ax.plot(
            drug_conc, lfc_pred,
            color  = color,
            lw     = 2,
            ls     = '--',
            label  = f'Predicted (loss = {best_fit.fun:.4f})',
            zorder = 2,
        )

        ax.axhline(0, color='grey', lw=0.8, ls=':')
        ax.set_xticks(drug_conc)
        ax.set_xticklabels(
            [f'{c:.1e}' for c in real_concentrations],
            rotation=45, ha='right', fontsize=8,
        )
        ax.set_xlabel('Drug concentration (M)', fontsize=10)
        ax.set_ylabel('LFC vs Ctrl', fontsize=10)
        ax.set_title(titles[i], fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Observed vs Predicted LFC — Kinetic Model Fit',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_optimization_diagnostic(fit_history: list):
    '''a
    
    to add
    
    '''
    # 1. Filter for successful optimization runs
    successful_runs = fit_history[fit_history['success'] == True].copy()

    # 2. Sort the runs by final cost from lowest (best) to highest
    successful_runs_sorted = successful_runs.sort_values(by='final_cost').reset_index(drop=True)

    # 3. Create the plot
    plt.figure(figsize=(10, 6))

    # Plotting as both lines and points for clear visualization
    plt.plot(successful_runs_sorted.index, successful_runs_sorted['final_cost'], 
            linestyle='-', marker='o', color='#1f77b4', markersize=4, label='Ordered Fits')

    # Add titles and labels in B2 English
    plt.title('Multi-Start Optimization Diagnostic: Final Cost Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sorted Successful Runs (Ranked from Best to Worst)', fontsize=12)
    plt.ylabel('Final Loss Value (Weighted MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    # Display the plot
    plt.tight_layout()
    plt.show()




def plot_parameter_summary(
    fit_dict: dict,
    class_names: list,
    log_bounds: tuple = (-3, 3),
    figsize: tuple | None = None,
):
    """
    Visualise fitted parameters from multistart_fit_multiclass in a two-panel layout.

    Left panel — Shared kinetic parameters
        Horizontal bar chart showing log10 of each shared parameter.
        One bar per parameter, not repeated per class.
        Bars are coloured by sign (positive = warm, negative = cool) relative
        to the log10 value, giving an immediate sense of magnitude.

    Right panel — Class-specific scalers (delta and beta)
        Heatmap with classes on the x-axis and the two scalers on the y-axis.
        Values are in linear [0, 1] scale (not log-transformed) because they
        are bounded multiplicative scalers, not rates.
        Annotated with the numeric value for readability.

    Parameters
    ----------
    fit_dict : dict
        Best-parameter dictionary as returned by multistart_fit_multiclass
        with show_dict=True. Shared params are in linear scale; class-specific
        scalers follow the naming convention 'delta_{class}' and 'beta_{class}'.
    class_names : list of str
        Class identifiers — must match those used during fitting.
    log_bounds : tuple
        (min, max) used during optimisation, for reference lines on the left panel.
    figsize : tuple or None
        Override figure size. Default scales with number of classes.
    """

    # ── Split fit_dict into shared and class-specific ─────────────────────────
    shared = {
        k: v for k, v in fit_dict.items()
        if not k.startswith('delta_') and not k.startswith('beta_')
    }
    # Filtra i parametri con valore 0 o negativo prima del log10
    shared = {k: v for k, v in shared.items() if v > 0}

    # Convert shared params to log10 for display
    shared_log = {k: np.log10(v) for k, v in shared.items()}

    # Build class-specific matrix: rows = ['delta', 'beta'], cols = class_names
    scaler_matrix = np.array([
        [fit_dict[f'delta_{c}']   for c in class_names],
        [fit_dict[f'beta_{c}'] for c in class_names],
    ])
    scaler_labels = ['δ  (decoy strength\nscales k₋₂)', 'β  (BP strength\nscales k₋₁)']

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_classes = len(class_names)
    if figsize is None:
        figsize = (6 + n_classes * 1.1, max(5, len(shared) * 0.55 + 2))

    fig = plt.figure(figsize=figsize)
    # Left panel gets fixed width; right panel expands with number of classes
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[2, max(2, n_classes * 0.9)],
        wspace=0.45,
    )
    ax_left  = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # ── Left panel: shared parameters barchart ────────────────────────────────
    param_names = list(shared_log.keys())
    values      = list(shared_log.values())
    y_pos       = np.arange(len(param_names))

    # Colour: diverging from 0 in log space
    norm   = mcolors.TwoSlopeNorm(vmin=log_bounds[0], vcenter=0, vmax=log_bounds[1])
    colors = [plt.cm.RdBu_r(norm(v)) for v in values]

    bars = ax_left.barh(y_pos, values, color=colors, height=0.6,
                        edgecolor='white', linewidth=0.8)

    # Annotate each bar with the linear value
    for bar, v, lin in zip(bars, values, shared.values()):
        x_ann = v + (0.05 if v >= 0 else -0.05)
        ha    = 'left' if v >= 0 else 'right'
        ax_left.text(x_ann, bar.get_y() + bar.get_height() / 2,
                     f'{lin:.2e}', va='center', ha=ha, fontsize=7.5,
                     color='#333333')

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(param_names, fontsize=9)
    ax_left.set_xlabel('log₁₀(parameter value)', fontsize=9)
    ax_left.set_title('Shared kinetic\nparameters', fontsize=10, fontweight='bold', pad=10)
    ax_left.axvline(0, color='#888888', linewidth=0.8, linestyle='--')
    ax_left.set_xlim(log_bounds[0] - 0.5, log_bounds[1] + 0.8)
    ax_left.spines[['top', 'right']].set_visible(False)
    ax_left.invert_yaxis()  # top-to-bottom reading order

    # ── Right panel: class-specific scalers heatmap ───────────────────────────
    im = ax_right.imshow(
        scaler_matrix,
        cmap='YlOrRd',
        vmin=0, vmax=1,
        aspect='auto',
    )

    # Annotate cells
    for row in range(2):
        for col in range(n_classes):
            val = scaler_matrix[row, col]
            text_color = 'white' if val > 0.65 else '#333333'
            ax_right.text(col, row, f'{val:.3f}',
                          ha='center', va='center',
                          fontsize=8, color=text_color, fontweight='bold')

    ax_right.set_xticks(np.arange(n_classes))
    ax_right.set_xticklabels(class_names, rotation=35, ha='right', fontsize=8)
    ax_right.set_yticks([0, 1])
    ax_right.set_yticklabels(scaler_labels, fontsize=9)
    ax_right.set_title('Class-specific scalers\n[0 – 1 linear scale]',
                        fontsize=10, fontweight='bold', pad=10)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)
    cbar.set_label('scaler value', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle('Fitted Parameter Summary', fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.show()



def plot_steady_state_probabilities(
    fit_dict: dict,
    class_names: list,
    drug_conc: np.ndarray,
    real_concentrations: list,
    model_name: str,
    fixed_params: dict | None = None,
    nodes_number: int = 4,
):
    """
    Plot steady-state probabilities (P1, P2, P3, P4) for each structural class
    across drug concentrations, using the parameters from a completed fit.

    For each class, the shared kinetic parameters are taken from fit_dict,
    and the class-specific scalers (delta, beta) are injected before calling
    _steady_state directly — bypassing LFC computation entirely.

    This is useful to:
      - verify that the model topology is behaving as expected
      - check whether drug modulation is actually changing state occupancy
      - inspect whether delta/beta are differentiating classes at steady state

    Layout: one row per class, one subplot per drug concentration point.
    Within each subplot, a bar chart shows P1..P4.

    Parameters
    ----------
    fit_dict : dict
        Best-parameter dictionary from multistart_fit_multiclass (show_dict=True).
        Shared params in linear scale; class-specific as 'delta_{name}', 'beta_{name}'.
    class_names : list of str
        Class identifiers — must match those used during fitting.
    drug_conc : np.ndarray
        Drug concentrations to evaluate (excluding control 0 — it is added internally).
    real_concentrations : list
        Human-readable labels for drug_conc (same length, used for x-axis titles).
    model_name : str
        Key in db.ALL_MODELS.
    fixed_params : dict or None
        Parameters fixed during fitting (e.g. {'k1': 1., 'k2': 1.}).
    nodes_number : int
        3 or 4, must match the model topology.
    """

    # ── Setup ─────────────────────────────────────────────────────────────────
    mod_info  = db.ALL_MODELS[model_name]
    all_concs = np.concatenate([[0.], np.asarray(drug_conc)])
    all_labels = ['Ctrl'] + [f'{c:.1e}' for c in real_concentrations]

    n_classes = len(class_names)
    n_concs   = len(all_concs)
    node_keys = [f'P{i}' for i in range(1, nodes_number + 1)]
    node_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']  # P1..P4

    # Identify shared params
    shared_names = [
        k for k in fit_dict
        if not k.startswith('delta_') and not k.startswith('beta_')
    ]

    fig, axes = plt.subplots(
        n_classes, n_concs,
        figsize=(n_concs * 1.8, n_classes * 2.8),
        sharey='row',
    )
    # Normalise axes indexing to 2D always
    if n_classes == 1:
        axes = axes[np.newaxis, :]
    if n_concs == 1:
        axes = axes[:, np.newaxis]

    for row, name in enumerate(class_names):

        # Build param dict for this class
        params = {k: fit_dict[k] for k in shared_names}
        if fixed_params:
            params.update(fixed_params)
        params['delta'] = fit_dict[f'delta_{name}']
        params['beta']  = fit_dict[f'beta_{name}']

        # Apply class-specific scalers (mirrors _predict_lfc)
        params_scaled = dict(params)
        params_scaled['k_2'] = params_scaled.get('delta', 1.0) * params_scaled.get('k_2', 0.0)
        params_scaled['k_1'] = params_scaled.get('beta',  1.0) * params_scaled.get('k_1', 0.0)

        for col, (drug, label) in enumerate(zip(all_concs, all_labels)):
            ax = axes[row, col]

            res = sim._steady_state(
                param_set    = params_scaled,
                drug         = float(drug),
                mods         = mod_info['mods'],
                drug_params  = {},
                nodes_number = nodes_number,
            )

            probs  = [res.get(k, 0.0) for k in node_keys]
            x_pos  = np.arange(len(node_keys))

            bars = ax.bar(x_pos, probs, color=node_colors[:len(node_keys)],
                          width=0.6, edgecolor='white', linewidth=0.6)

            # Annotate bars with value if > 0.05
            for bar, p in zip(bars, probs):
                if p > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, p + 0.01,
                            f'{p:.2f}', ha='center', va='bottom',
                            fontsize=6.5, color='#333333')

            ax.set_ylim(0, 1.12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(node_keys, fontsize=7)
            ax.set_title(label, fontsize=7.5, pad=3)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='y', labelsize=6.5)

            # Row label on the leftmost subplot only
            if col == 0:
                ax.set_ylabel(name.replace('_', ' '), fontsize=7.5, labelpad=6)

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    node_labels = ['P1 — free', 'P2 — BP open', 'P3 — decoy', 'P4 — BP closed']
    legend_elements = [
        Patch(facecolor=node_colors[i], label=node_labels[i])
        for i in range(nodes_number)
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=nodes_number,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f'Steady-state probabilities by class — {model_name}',
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.show()