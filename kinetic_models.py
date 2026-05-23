"""

3-node directed graph:
    Node 1 (free)  <──>  Node 2 (correct site)    k1  (on), k_1 (off)
    Node 1 (free)  <──>  Node 3 (decoy   site)    k2  (on), k_2 (off)

4-node directed graph:
    Node 1 (free)               <──>  Node 2 (correct site)           k1  (on), k_1 (off)
    Node 1 (free)               <──>  Node 3 (decoy   site)           k2  (on), k_2 (off)
    Node 2 (correct site open)  <──>  Node 4 (correct site closed)    k4 (on), k4 (off)

Steady-state probabilities are computed via the Matrix-Tree Theorem
(Kirchhoff's theorem on the reversed weighted DiGraph).

────────────────────────────────────────────────────────────────────────────
Constraint – the drug must act on binding specificity:
    k1  can only DECREASE   (drug hinders correct-site association)
    k_1 can only INCREASE   (drug accelerates correct-site dissociation)
    k2  can only INCREASE   (drug promotes decoy-site association)
    k_2 can only DECREASE   (drug stabilises decoy binding)

Available drug-modulation functions
────────────────────────────────────
Decreasing  (applied to k1 or k_2 or k3):
    • hill_decay         k · (1 – drugⁿ / (K_hillⁿ + drugⁿ))
    • linear_decrease    k / alpha * drug

Increasing  (applied to k_1 or k2):
    • linear_increase    k + α · drug
    • hill_activation    k + Vmax · drugⁿ / (K_hillⁿ + drugⁿ)

────────────────────────────────────────────────────────────────────────────
Model catalogue (X models, explicitly defined):
   

        TO FILLL

"""

# =============================================================================
# Library importing
# =============================================================================

from typing import Callable
import networkx as nx
import numpy as np
import parameter_function as param
import model_database as db

# =============================================================================
# Constant definition
# =============================================================================
ALL_MODELS = db._database_generation()


# =============================================================================
# Steady-state core (single point)
# =============================================================================

def _steady_state(
    param_set: dict,
    drug: float,
    mods: dict[str, tuple[str, Callable]],
    drug_params: dict,
    nodes_number: int
) -> dict:
    """
    Compute steady-state probabilities for one (param_set, drug) pair.

    Parameters
    ----------
    param_set   : one dict from param.generate_parameter_sets
    drug        : scalar drug concentration
    mods        : {param_name: (func_name, func)} for each modulated parameter
    drug_params : extra kinetic constants consumed by modulation functions
                  (K_hill, Vmax).
                  alpha and n are taken from param_set automatically.
    nodes_number: specify the graph topology. Different tipology structure leads to
                  different computation 

    Returns
    -------
    dict  { 'drug', 'P1', 'P2', 'P3',
            '<param>_drug' for each modulated parameter }
    """
    k1  = float(param_set["k1"])
    k_1 = float(param_set["k_1"])
    k2  = float(param_set["k2"])
    k3 = float(param_set["k3"])
    k4 = float(param_set["k4"])

    # ── Topology Bifurcation ──────────────────────────────────────────────────
    '''if nodes_number == 3:
        k_2 = float(param_set["k_2"])
    elif nodes_number == 4:
        k_2 = 1.'''
    k_2 = float(param_set["k_2"])           # to change

    # Keyword arguments passed to every modulation function.
    # param_set values (alpha, n) shadow any same-key entries in drug_params.
    kw = {
        **drug_params,
        "alpha": float(param_set.get("alpha", drug_params.get("alpha", 1.0))),
        "n":     int(param_set.get("n", drug_params.get("n", 1))),
    }

    # Apply drug modulations
    rates = {"k1": k1, "k_1": k_1, "k2": k2, "k_2": k_2, 'k3': k3, 'k4': k4}
    for pname, (_, func) in mods.items():
        rates[pname] = func(rates[pname], drug, **kw)

    # ── Build directed graph ──────────────────────────────────────────────────
    G = nx.DiGraph()

    # ── Topology Bifurcation ──────────────────────────────────────────────────

    if nodes_number == 3:
        nodes = [1, 2, 3]
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from([
            (1, 2, rates["k1"]),   # association  → correct site
            (2, 1, rates["k_1"]),  # dissociation ← correct site
            (1, 3, rates["k2"]),   # association  → decoy site
            (3, 1, rates["k_2"])   # dissociation ← decoy site
        ])
    elif nodes_number == 4:
        nodes = [1, 2, 3, 4]
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from([
            (1, 2, rates["k1"]),   # association  → correct site
            (2, 1, rates["k_1"]),  # dissociation ← correct site
            (1, 3, rates["k2"]),   # association  → decoy site
            (3, 1, rates["k_2"]),  # dissociation ← decoy site
            (2, 4, rates["k3"]),   # association closed format - correct site
            (4, 1, rates["k4"]),   # dissociation closed format - correct site
        ])

    # ── Matrix-Tree Theorem on reversed graph ─────────────────────────────────
    G_rev = G.reverse()
    rho = {
        node: nx.number_of_spanning_trees(G_rev, root=node, weight="weight")
        for node in nodes
    }
    Z = sum(rho.values())
    if Z == 0:
        raise ValueError(
            f"Partition function Z = 0 for param_set={param_set}, drug={drug}. "
            "Check graph connectivity and rate positivity."
        )

    # ── Assemble result ───────────────────────────────────────────────────────
    if nodes_number == 3:
        result = {
            "drug": drug,
            "P1":   rho[1] / Z,
            "P2":   rho[2] / Z,
            "P3":   rho[3] / Z,
            'nodes_topology': nodes_number
        }
        for pname in mods:
            result[f"{pname}_drug"] = round(rates[pname], 8)

    elif nodes_number == 4:
        result = {
            "drug": drug,
            "P1":   rho[1] / Z,
            "P2":   rho[2] / Z,
            "P3":   rho[3] / Z,
            "P4":   rho[4] / Z,
            'nodes_topology': nodes_number
        }
        for pname in mods:
            result[f"{pname}_drug"] = round(rates[pname], 8)

    return result


# =============================================================================
# Generic model runner
# =============================================================================

def run_model(
    drug_conc: list[float],
    epochs: int,
    mods: dict[str, tuple[str, Callable]],
    drug_params: dict,
    k_2_COST: float,
    nodes_number: int,
    model_name: str = "unnamed"
) -> list[dict]:
    """
    Run the 3-node steady-state model for all parameter sets and drug
    concentrations with the specified drug–parameter modulations.

    Parameters
    ----------
    drug_conc   : list of drug concentrations to scan
    epochs      : number of random parameter sets (calls param.generate_parameter_sets)
    mods        : {param_name: (func_name, func)} — which rates are modulated and how
    drug_params : extra kinetic constants for modulation functions
                  e.g. {'K_hill': 2.0, 'Vmax': 5.0}
    model_name  : label stored in every output dict

    Returns
    -------
    list of dicts  { 'model', 'mods_applied', 'params', 'results' }
        where 'results' is [ { 'drug', 'P1', 'P2', 'P3', ... }, ... ]
    """
    parameters_sets = param.generate_parameter_sets(epochs, k_2_multiplic_cost= k_2_COST)
    mods_applied = {pname: fname for pname, (fname, _) in mods.items()}
    all_results = []

    for param_set in parameters_sets:
        drug_results = [
            _steady_state(param_set, drug, mods, drug_params, nodes_number)
            for drug in drug_conc
        ]
        all_results.append({
            "model":        model_name,
            "mods_applied": mods_applied,
            "params":       param_set,
            "results":      drug_results,
        })

    return all_results




# =============================================================================
# Convenience runners
# =============================================================================

class Simulation:
    """
    Raccoglie i parametri di simulazione in un posto solo.
    Cambi un valore qui e si propaga ovunque.
    """

    def __init__(
        self,
        drug_conc: list[float],
        epochs: int,
        drug_params: dict,
        k_2_COST: float,
        nodes_num: int,
        assumption: bool,
    ):
        self.drug_conc   = drug_conc
        self.epochs      = epochs
        self.drug_params = drug_params
        self.k_2_COST    = k_2_COST
        self.nodes_num   = nodes_num
        self.assumption = True

    # ------------------------------------------------------------------ #
    # Metodi: stessa logica di prima, ma senza ripetere i parametri        #
    # ------------------------------------------------------------------ #

    def run_model(self, mods: dict, model_name: str = "unnamed") -> list[dict]:
        """Esegue un modello con le modulations passate."""
        parameters_sets = param.generate_parameter_sets(
            self.epochs, k_2_multiplic_cost=self.k_2_COST,assumption= self.assumption
        )
        mods_applied = {pname: fname for pname, (fname, _) in mods.items()}
        all_results = []

        for param_set in parameters_sets:
            drug_results = [
                _steady_state(param_set, drug, mods, self.drug_params, self.nodes_num)
                for drug in self.drug_conc
            ]
            all_results.append({
                "model":        model_name,
                "mods_applied": mods_applied,
                "params":       param_set,
                "results":      drug_results,
            })

        return all_results

    def run_single_model(self, model_name: str) -> list[dict]:
        """Esegue un modello dal catalogo per nome."""
        if model_name not in db.ALL_MODELS:
            raise KeyError(
                f"Model '{model_name}' non trovato.\n"
                f"Disponibili:\n" + "\n".join(f"  {k}" for k in db.ALL_MODELS)
            )
        info = db.ALL_MODELS[model_name]
        return self.run_model(info["mods"], model_name=model_name)

    def run_all_models(self) -> dict[str, list[dict]]:
        """Runs all models matching self.nodes_num."""
        return {
            name: self.run_model(info["mods"], model_name=name)
            for name, info in db.ALL_MODELS.items()
            if info["nodes_number"] == self.nodes_num          # ← add this
        }

    def run_subset_models(self, model_names: list[str]) -> dict[str, list[dict]]:
        """Runs only the specified models, enforcing nodes_num compatibility."""
        mismatched = [
            name for name in model_names
            if db.ALL_MODELS[name]["nodes_number"] != self.nodes_num   # ← add check
        ]
        if mismatched:
            raise ValueError(
                f"The following models have incompatible nodes_number for this simulation "
                f"(expected {self.nodes_num}):\n" + "\n".join(f"  {m}" for m in mismatched)
            )
        return {
            name: self.run_single_model(name)
            for name in model_names
        }

    def run_perturbation(
    self,
    param_name: str,
    param_values: list[float],
    model_names: list[str] | None = None,   # None = tutti i modelli compatibili
    overrides_extra: dict | None = None,    # altri parametri da fissare insieme
    ) -> dict[float, dict[str, list[dict]]]:
        """
        Esegue una perturbation analysis su un singolo parametro.

        Fissa `param_name` ad ogni valore in `param_values` e gira tutti i
        modelli (o solo quelli in `model_names`). I restanti parametri
        vengono campionati normalmente, eccetto quelli in `overrides_extra`.

        Parametri
        ----------
        param_name      : nome del parametro da perturbare (es. "alpha", "k_2")
        param_values    : lista di valori da testare
        model_names     : subset di modelli; None = tutti compatibili con nodes_num
        overrides_extra : altri parametri da bloccare contemporaneamente
                        (es. {"n": 2} per fissare il Hill coefficient)

        Ritorna
        -------
        Ritorna dict[str, list[dict]] — identico a run_all_models().
        Il valore perturbato è leggibile da run["params"][param_name]

        Esempio
        -------
        results = sim.run_perturbation(
            param_name   = "alpha",
            param_values = np.logspace(-1, 2, 10).tolist(),
            model_names  = ["k1[linear_decrease]__3nodes"],
        )
        """
        overrides_extra = overrides_extra or {}

        # seleziona i modelli target
        if model_names is None:
            target_models = {
                name: info
                for name, info in db.ALL_MODELS.items()
                if info["nodes_number"] == self.nodes_num
            }
        else:
            target_models = {name: db.ALL_MODELS[name] for name in model_names}
        output: dict[str, list[dict]] = {name: [] for name in target_models}

        for val in param_values:
            overrides = {param_name: val, **overrides_extra}

            parameters_sets = param.generate_parameter_sets_override(
                self.epochs,
                k_2_multiplic_cost=self.k_2_COST,
                overrides=overrides,
            )

            for name, info in target_models.items():
                mods         = info["mods"]
                mods_applied = {pname: fname for pname, (fname, _) in mods.items()}

                for ps in parameters_sets:
                    drug_results = [
                        _steady_state(ps, drug, mods, self.drug_params, self.nodes_num)
                        for drug in self.drug_conc
                    ]
                    output[name].append({
                        "model":        name,
                        "mods_applied": mods_applied,
                        "params":       ps,          # ps già contiene param_name=val
                        "results":      drug_results,
                    })

        return output








# =============================================================================
# METHODS TO CHANGE ------ ALREADY CONTAINED IN CLASS ------ (down)
# =============================================================================

def run_single_model(
    model_name: str,
    drug_conc: list[float],
    epochs: int,
    drug_params: dict,
    k_2_COST: float,
    nodes_num: int
) -> list[dict]:
    """
    Run one named model from ALL_MODELS.

    Parameters
    ----------
    model_name  : key in ALL_MODELS, e.g. 'k_1[linear_increase]'
    drug_conc   : concentration grid
    epochs      : number of parameter sets
    drug_params : {'K_hill', 'Vmax'} — constants for drug functions

    Raises
    ------
    KeyError if model_name is not in ALL_MODELS.
    """
    if model_name not in db.ALL_MODELS:
        raise KeyError(
            f"Model '{model_name}' not found in catalogue.\n"
            f"Available names:\n" + "\n".join(f"  {k}" for k in db.ALL_MODELS)
        )
    info = db.ALL_MODELS[model_name]
    return run_model(drug_conc, 
                     epochs, 
                     info["mods"],
                    drug_params,
                    model_name=model_name,
                    k_2_COST=k_2_COST, 
                    nodes_number=  nodes_num)


def run_all_models(
    drug_conc: list[float],
    epochs: int,
    drug_params: dict,
    k_2_COST: float,
    nodes_num: int,
) -> dict[str, list[dict]]:
    """
    Run all 11 models sequentially.

    Parameters
    ----------
    drug_conc   : concentration grid
    epochs      : number of parameter sets per model
    drug_params : kinetic constants for drug modulation functions

    Returns
    -------
    { model_name: run_model(...) output }
    """
    return {
        name: run_model(drug_conc, epochs, info["mods"], drug_params, model_name=name, k_2_COST = k_2_COST, nodes_number= nodes_num)
        for name, info in db.ALL_MODELS.items()
    }


def run_subset_models(
    model_names: list[str],
    drug_conc: list[float],
    epochs: int,
    drug_params: dict,
    k_2_COST: float,
    nodes_num: int
    
) -> dict[str, list[dict]]:
    """
    Run a user-specified subset of models by name.

    Parameters
    ----------
    model_names : list of keys from ALL_MODELS
    drug_conc, epochs, drug_params : as in run_all_models
    """
    return {
        name: run_single_model(name, drug_conc, epochs, drug_params, k_2_COST = k_2_COST, nodes_num = nodes_num)
        for name in model_names
    }
 


# =============================================================================
# METHODS TO CHANGE ------ ALREADY CONTAINED IN CLASS ------ (up)
# =============================================================================









def filter_models_by_params(
    params: list[str],
    exact: bool = False,
) -> dict[str, dict]:
    """
    Return the sub-catalogue of models that modulate the given parameters.

    Parameters
    ----------
    params : subset of ['k1', 'k_1', 'k2', 'k_2']
    exact  : if True, the model must modulate *exactly* these parameters
             (no more, no fewer); if False, at least these parameters.
    """
    target = set(params)
    result = {}
    for name, info in ALL_MODELS.items():
        modulated = set(info["params_modulated"])
        if exact:
            if modulated == target:
                result[name] = info
        else:
            if target.issubset(modulated):
                result[name] = info
    return result





# =============================================================================
# Entry point (example)
# =============================================================================

def main() -> None:
    drug_conc = np.linspace(0, 10, 30).tolist()
    epochs = 3
    k_2_COST = 100
    drug_params = {
        "K_hill": 2.0,   # for hill_decay / hill_activation
        "Vmax":   5.0,   # for hill_activation
    }

    # ── Catalogue overview ────────────────────────────────────────────────────
    print("\nCatalogue summary:")
    for label, count in db.catalogue_summary().items():
        print(f"  {label}: {count}")

    # ── List all models ───────────────────────────────────────────────────────
    print("\nAll models:")
    db.list_models(verbose=True)

    # ── Run a single model ────────────────────────────────────────────────────
    target = "k_1[linear_increase]"
    print(f"\nRunning model: {target}")
    results = run_single_model(target, drug_conc, epochs, drug_params, k_2_COST= k_2_COST)

    print("  Sample output – first param set, first 3 drug concentrations:")
    for entry in results[0]["results"][:3]:
        print("   ", entry)


if __name__ == "__main__":
    main()








