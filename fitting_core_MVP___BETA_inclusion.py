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





        ## Observed Values

def _observed_lfc(data: pd.DataFrame, vector_conc_without_control) -> dict:
    '''Avoid to consider Control explicitely because the LFC is computed with respect to Ctrl'''
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
        'conc': np.asarray(vector_conc_without_control),
        'means': np.asarray(vector_means),
        'stds': np.asarray(vectors_std),
        'weights': 1 / np.asarray(vectors_std)**2
    }

    return ps



        ## Predicted Values 


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
        result_dict = { **dict(zip(free_param_names, free_parameters_values)), **fixed_params}
    else:
        result_dict = dict(zip(free_param_names, free_parameters_values))
    
    # 2. Define the required keys
    required_keys = ['k1', 'k_1', 'k2', 'k_2', 'k3', 'k4', 'alpha', 'beta']
    missing_keys = []
    
    # 3. Check for missing keys, set them to 0, and track them
    for key in required_keys:
        if key not in result_dict:
            result_dict[key] = 0
            missing_keys.append(key)
            
    # 4. Trigger a warning if any keys were missing
    if missing_keys:
        message = f"The following parameters were missing and have been automatically set to 0: {', '.join(missing_keys)}"
        # UserWarning is the standard category for user-facing configuration alerts
        warnings.warn(message, category=UserWarning, stacklevel=2 )
        
    return result_dict
    

def _predict_lfc(free_parameters_values: list,
    free_param_names: list,
    drug_conc: np.array,
    model_name: str,
    fixed_params: dict | None = None,
    ):
    '''the function ecc..
    
    
    autonomusly retrieve the nodes number'''

    drug_conc.sort()
    if np.isclose(drug_conc[0], 0):
        drug_conc = drug_conc[1:]
    params = _param_array_to_dict(free_parameters_values, free_param_names, fixed_params)
    params['k_2'] = params['beta'] * params['k_2']
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
    model_info = db.ALL_MODELS[mod]  
    mods       = model_info['mods']    
    model_nodes  = model_info['nodes_number']

    # Bound _probs Computation for Control to use as Reference 
    res_Ctrl = sim._steady_state(param_set = params,
                                 drug = 0.,
                                 mods= mods,
                                 drug_params = {},
                                 nodes_number = model_nodes)
    
    bound_probabilites_Ctrl = _p_bound(result = res_Ctrl, nodes_number = model_nodes)

    # Bound _probs Computation for Drug
    bound_probabilites_drug = []

    for d in drug_conc:
        res_drug = sim._steady_state(param_set = params,
                                 drug = d,
                                 mods= mods,
                                 drug_params = {},
                                 nodes_number = model_nodes)
        
        bound_probabilites_drug.append(_p_bound(result = res_drug, nodes_number = model_nodes))

    # LFC final computation
    lfc_pred = np.log( np.asarray(bound_probabilites_drug) / np.asarray(bound_probabilites_Ctrl) )
    return lfc_pred




def _weighted_mse(log_theta,
                  free_param_names,
                  drug_conc: np.array,
                  lfc_obs: np.array,
                  weights: np.array,
                  model_name: str,
                  fixed_params: None | dict = None
                  ):
    """Scalar loss — operates in log10(θ) space."""
    theta = 10.0 ** log_theta

    # Predict LFC values

    lfc_pred = _predict_lfc(free_parameters_values = theta,
                 free_param_names = free_param_names,
                 drug_conc = drug_conc,
                 model_name = model_name,
                 fixed_params = fixed_params)
    
    # Avoid absurd values 
    if np.any(~np.isfinite(lfc_pred)):
        return 1e10   # penalise degenerate parameter sets
    

    # Residuals to minimize 
    residuals = lfc_obs - lfc_pred

    return float(np.sum(weights * residuals**2) / weights.sum())





def multistart_fit(data: pd.DataFrame,
                   free_param_names: list,
                   vector_conc_without_control: np.ndarray,
                   model_name: str,
                   fixed_params: dict | None = None,
                   n_starts: int = 100,
                   log_bounds: tuple = (-3, 3),
                   show_history = False,
                   show_messages = True,
                   show_dict = False,
                   ):

    # Observed information
    obs = _observed_lfc(data, vector_conc_without_control)

    # Best initialization (loss for comparison set to infinite to compare with sequencially minima)
    best_loss   = np.inf
    best_result = None
    history = []

    for k in range(n_starts):

        # Random Starting poing: 1 random number for each parameter
        log_theta_0 = np.random.uniform(log_bounds[0], log_bounds[1],
                                        size=len(free_param_names))

        # Bounds
        bounds = [log_bounds] * len(free_param_names)

        # Minimizator call (L-BFGS-B)
        result = minimize(
            fun   = _weighted_mse,
            x0    = log_theta_0,
            args  = (free_param_names, vector_conc_without_control,
                     obs['means'], obs['weights'],
                     model_name, fixed_params),
            method = 'L-BFGS-B',
            bounds = bounds
        )

        history.append({
            "run_id": k,
            "success": bool(result.success),
            "final_cost": float(result.fun),
            "params_start_log": log_theta_0.tolist(),
            "params_final_log": result.x.tolist()
        })

        # Best update
        if result.fun < best_loss:
            best_loss   = result.fun
            best_result = result
    if show_messages:
        print(f"\n==================== FIT COMPLETED =====================")
        print(f"Best loss     : {best_loss:.4f}")
        print(f"Convergence   : {best_result.success}")


        
        print(f"log10(params) : {best_result.x}")
        print(f"Params linear: {10**best_result.x}")

    df_history = pd.DataFrame(history)
    
    output = [best_result]
    if show_history:
        output.append(df_history)
    if show_dict:
        best_dict = dict(zip(free_param_names, best_result.x))
        output.append(best_dict)
    if len(output) == 1:
        return output[0]
    return tuple(output)






# ====================================================================
# MULTICLASS FITTING (SIMULTANEOUSLY)
# ====================================================================

def _weighted_mse_multiclass(log_theta,
                              free_param_names,
                              drug_conc: np.array,
                              obs_maxdecoy: dict,
                              obs_nodecoy: dict,
                              model_name: str,
                              fixed_params: None | dict = None
                              ):
    """
    Scalar loss over two classes simultaneously.
    Beta enters as the last two elements of log_theta:
        log_theta[-2] = beta_maxdecoy  (expected ~ 1)
        log_theta[-1] = beta_nodecoy   (expected ~ 0)
    All other parameters are shared between classes.
    """

    theta = 10.0 ** log_theta

    # Split shared params from class-specific betas
    # Solo i parametri condivisi passano per 10**
    shared_names  = [p for p in free_param_names if p not in ('beta_maxdecoy', 'beta_nodecoy')]
    n_shared      = len(shared_names)

    shared_values = 10.0 ** log_theta[:n_shared]   # log10 → lineare
    beta_maxdecoy = log_theta[free_param_names.index('beta_maxdecoy')]  # già lineare
    beta_nodecoy  = log_theta[free_param_names.index('beta_nodecoy')]   # già lineare

    # Build fixed_params for each class, injecting the right beta as k2 scaler
    def _build_params(beta, fixed_params):
        fp = dict(fixed_params) if fixed_params is not None else {}
        fp['beta'] = float(beta)
        return fp

    fp_maxdecoy = _build_params(beta_maxdecoy, fixed_params)
    fp_nodecoy  = _build_params(beta_nodecoy,  fixed_params)

    # Predict LFC for each class
    try:
        lfc_pred_maxdecoy = _predict_lfc(
            free_parameters_values = shared_values,
            free_param_names       = shared_names,
            drug_conc              = drug_conc.copy(),
            model_name             = model_name,
            fixed_params           = fp_maxdecoy
        )
        lfc_pred_nodecoy = _predict_lfc(
            free_parameters_values = shared_values,
            free_param_names       = shared_names,
            drug_conc              = drug_conc.copy(),
            model_name             = model_name,
            fixed_params           = fp_nodecoy
        )
    except Exception:
        return 1e10

    # Penalise degenerate parameter sets
    if np.any(~np.isfinite(lfc_pred_maxdecoy)) or np.any(~np.isfinite(lfc_pred_nodecoy)):
        return 1e10

    # Loss for each class
    res_maxdecoy = obs_maxdecoy['means'] - lfc_pred_maxdecoy
    res_nodecoy  = obs_nodecoy['means']  - lfc_pred_nodecoy

    loss_maxdecoy = np.sum(obs_maxdecoy['weights'] * res_maxdecoy**2) / obs_maxdecoy['weights'].sum()
    loss_nodecoy  = np.sum(obs_nodecoy['weights']  * res_nodecoy**2)  / obs_nodecoy['weights'].sum()

    return float(loss_maxdecoy + loss_nodecoy)


def multistart_fit_multiclass(
    max_decoy_data: pd.DataFrame,
    no_decoy_data: pd.DataFrame,
    shared_param_names: list,
    drug_conc: np.ndarray,
    model_name: str,
    fixed_params: dict | None = None,
    n_starts: int = 100,
    log_bounds: tuple = (-3, 3),
    beta_bounds: tuple = (0.0, 1.0),
    show_messages: bool = True,
    show_history: bool = False,
    show_dict: bool = False,
):
    """
    Multi-start fitting over two classes simultaneously.
    shared_param_names : kinetic parameters shared between classes (no beta)
    beta_maxdecoy and beta_nodecoy are added automatically.
    """

    # Observed — computed once outside the loop
    obs_maxdecoy = _observed_lfc(max_decoy_data, drug_conc)
    obs_nodecoy  = _observed_lfc(no_decoy_data,  drug_conc)

    # Full parameter list: shared + two betas
    full_param_names = shared_param_names + ['beta_maxdecoy', 'beta_nodecoy']

    # Bounds: log space for shared, linear for betas
    bounds = (
        [log_bounds] * len(shared_param_names) +
        [beta_bounds, beta_bounds]
    )

    best_loss   = np.inf
    best_result = None
    history     = []

    for k in range(n_starts):

        # Random starting point
        log_theta_shared = np.random.uniform(log_bounds[0], log_bounds[1],
                                             size=len(shared_param_names))
        beta_start       = np.random.uniform(beta_bounds[0], beta_bounds[1],
                                             size=2)
        x0 = np.concatenate([log_theta_shared, beta_start])

        result = minimize(
            fun    = _weighted_mse_multiclass,
            x0     = x0,
            args   = (full_param_names, drug_conc,
                      obs_maxdecoy, obs_nodecoy,
                      model_name, fixed_params),
            method = 'L-BFGS-B',
            bounds = bounds
        )

        history.append({
            "run_id"          : k,
            "success"         : bool(result.success),
            "final_cost"      : float(result.fun),
            "params_start"    : x0.tolist(),
            "params_final"    : result.x.tolist(),
        })

        if result.fun < best_loss:
            best_loss   = result.fun
            best_result = result

    if show_messages:
        print(f"\n==================== FIT COMPLETED =====================")
        print(f"Best loss        : {best_loss:.4f}")
        print(f"Convergence      : {best_result.success}")
        print(f"Shared params (log10) : {dict(zip(shared_param_names, best_result.x[:len(shared_param_names)]))}")
        print(f"beta_maxdecoy    : {best_result.x[-2]:.4f}")
        print(f"beta_nodecoy     : {best_result.x[-1]:.4f}")

    # Reconstruct dictionary — shared in linear, betas as-is
    shared_linear = 10 ** best_result.x[:len(shared_param_names)]
    betas         = best_result.x[len(shared_param_names):]
    best_dict     = {
        **dict(zip(shared_param_names, shared_linear)),
        'beta_maxdecoy': float(betas[0]),
        'beta_nodecoy' : float(betas[1]),
    }

    output = [best_result]
    if show_history:
        output.append(pd.DataFrame(history))
    if show_dict:
        output.append(best_dict)

    if len(output) == 1:
        return output[0]
    return tuple(output)





# =======================================================
# PLOT



import time


def plot_multistart_prediction_from_callable(multistart_function, **kwargs):
    """
    Runs a fast benchmark by calling the provided multistart_fit function 3 times,
    measures the execution speed, and plots a linear time projection up to 500 starts.
    
    Parameters:
    - multistart_function: The actual function object (multistart_fit)
    - kwargs: All the specific arguments required by your function
    """
    print("Starting the benchmark using the provided function callable (3 test runs)...")
    
    # Force n_starts = 1 and show_history = True for the benchmark test runs
    test_kwargs = kwargs.copy()
    test_kwargs['n_starts'] = 1
    test_kwargs['show_history'] = True
    test_kwargs['show_messages'] = False
    
    run_times = []
    
    # Execute 3 quick runs to get an accurate average of your CPU speed
    for i in range(3):
        start_time = time.time()
        
        # Execute the callable function
        multistart_function(**test_kwargs)
        
        duration = time.time() - start_time
        run_times.append(duration)
        print(f"  Test run {i+1}/3 completed in {duration:.4f} seconds.")
    
    # Calculate the average duration of a single optimization loop
    average_single_time = np.mean(run_times)
    print(f"\nBenchmark completed! Average time per single start: {average_single_time:.4f} seconds.\n")
    
    # Generate data points from 1 to 500 starts
    start_values = np.arange(1, 501)
    predicted_seconds = start_values * average_single_time
    predicted_minutes = predicted_seconds / 60
    
    # Plotting the results (Double-axis graph)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    
    # Left Y-Axis: Seconds (Blue Line)
    color = '#1f77b4'
    ax1.set_xlabel('Number of Starts (n_starts)')
    ax1.set_ylabel('Estimated Time (Seconds)', color=color)
    ax1.plot(start_values, predicted_seconds, color=color, lw=2, label='Seconds')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Right Y-Axis: Minutes (Orange Dashed Line)
    ax2 = ax1.twinx()  
    color = '#ff7f0e'
    ax2.set_ylabel('Estimated Time (Minutes)', color=color)
    ax2.plot(start_values, predicted_minutes, color=color, lw=2, linestyle='--', label='Minutes')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Extract final prediction values at 500 starts
    final_time_sec = predicted_seconds[-1]
    final_time_min = predicted_minutes[-1]
    
    plt.title('Execution Time Prediction for Multi-Start Optimization', fontsize=12, fontweight='bold')
    
    # Add an information text box with the final 500-start prediction
    estimation_text = f"Estimation for 500 starts:\n~ {final_time_sec:.1f} seconds\n(~ {final_time_min:.2f} minutes)"
    ax1.text(20, final_time_sec * 0.7, estimation_text, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.show()


def plot_fit_comparison(
    max_decoy_data, no_decoy_data,
    best_fit_maxdecoy, best_fit_nodecoy,
    fit_dictionary_maxdecoy, fit_dictionary_nodecoy,
    drug_conc, real_concentrations,
    model_name_maxdecoy, model_name_nodecoy
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    configs = [
        {
            "ax"          : axes[0],
            "data"        : max_decoy_data,
            "best_fit"    : best_fit_maxdecoy,
            "fit_dict"    : fit_dictionary_maxdecoy,
            "model_name"  : model_name_maxdecoy,
            "color"       : "#e07b39",
            "title"       : "Max Decoy (Decoy 1 — Strong)",
            "free_params" : list(fit_dictionary_maxdecoy.keys()),
        },
        {
            "ax"          : axes[1],
            "data"        : no_decoy_data,
            "best_fit"    : best_fit_nodecoy,
            "fit_dict"    : fit_dictionary_nodecoy,
            "model_name"  : model_name_nodecoy,
            "color"       : "#3a7ebf",
            "title"       : "No Decoy (Decoy 7 — Strong)",
            "free_params" : list(fit_dictionary_nodecoy.keys()),
        },
    ]

    for cfg in configs:
        ax    = cfg["ax"]
        color = cfg["color"]

        # ── Observed ──────────────────────────────────────────────────────
        obs = _observed_lfc(cfg["data"], drug_conc)

        ax.errorbar(
            drug_conc, obs["means"],
            yerr   = obs["stds"],
            fmt    = "o",
            color  = 'black',
            capsize= 4,
            label  = "Observed (mean ± std)",
            zorder = 3,
        )

        # ── Predicted ─────────────────────────────────────────────────────
        log_params = np.array(list(cfg["fit_dict"].values()))
        theta      = 10.0 ** log_params

        lfc_pred = _predict_lfc(
            free_parameters_values = theta,
            free_param_names       = cfg["free_params"],
            drug_conc              = drug_conc.copy(),
            model_name             = cfg["model_name"],
        )

        ax.plot(
            drug_conc, lfc_pred,
            color  = color,
            lw     = 2,
            ls     = "--",
            label  = f"Predicted  (loss = {cfg['best_fit'].fun:.4f})",
            zorder = 2,
        )

        # ── Reference line at 0 ───────────────────────────────────────────
        ax.axhline(0, color="grey", lw=0.8, ls=":")

        # ── Axes formatting ───────────────────────────────────────────────
        ax.set_xticks(drug_conc)
        ax.set_xticklabels(
            [f"{c:.1e}" for c in real_concentrations],
            rotation=45, ha="right", fontsize=8
        )
        ax.set_xlabel("Drug concentration (M)", fontsize=10)
        ax.set_ylabel("LFC vs Ctrl", fontsize=10)
        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Observed vs Predicted LFC — Kinetic Model Fit", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()




# ======================================================
# MULTICLASS PLOT 

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
    n = len(classes_data)
    if colors is None:
        colors = [f'C{i}' for i in range(n)]
    if titles is None:
        titles = classes_names

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    # Parametri condivisi — stessi per tutte le classi
    shared_names  = [k for k in fit_dict if not k.startswith('beta_')]
    shared_values = np.array([fit_dict[k] for k in shared_names])

    for i, (data, name) in enumerate(zip(classes_data, classes_names)):
        ax    = axes[i]
        color = colors[i]

        # Observed
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

        # Predicted — inietta il beta della classe corrente
        beta = fit_dict[f'beta_{name}']
        fp   = dict(fixed_params) if fixed_params is not None else {}
        fp['beta'] = beta

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
            rotation=45, ha='right', fontsize=8
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