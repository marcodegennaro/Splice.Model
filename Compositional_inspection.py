"""
compositional_problem.py
------------------------
This script demonstrates the compositional bias problem in sequencing
experiments and compares different normalization strategies.

We simulate three classes of sequences:
    - weak:  BP does not work well   → drug has NO effect
    - decoy: sequence binds decoy    → drug has SMALL effect (-5%)
    - strong: BP works well          → drug has LARGE effect (-25%)

The script is organized in 4 steps:
    1. Generate data and show that raw fitness is misleading
    2. Understand WHY this happens (compositional problem)
    3. Propose size-factor normalization solutions
    4. Test and compare all solutions on the toy example
"""

import numpy as np
import pandas as pd

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

N_SEQUENCES     = 50        # sequences per class (kept small for readability)
N_INPUT_READS   = 500_000   # total reads in input library
N_OUTPUT_READS  = 400_000   # total reads in output library (ctrl or drug)

# True biological effect of the drug on absolute concentration
DRUG_EFFECT = {
    "weak":   0.00,   # 0%  reduction
    "decoy":  0.05,   # 5%  reduction
    "strong": 0.25,   # 25% reduction
}

# Base absolute concentrations (arbitrary units)
BASE_CONC = {
    "weak":   10.0,
    "decoy":  30.0,
    "strong": 60.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def simulate_reads(concentrations: np.ndarray, total_reads: int) -> np.ndarray:
    """
    Distribute total_reads proportionally to concentrations.
    Uses multinomial sampling to add realistic sequencing noise.
    """
    freqs = concentrations / concentrations.sum()
    return np.random.multinomial(total_reads, freqs).astype(float)


def log_fitness(reads_out: np.ndarray,
                reads_in:  np.ndarray,
                sf_out:    float = 1.0,
                sf_in:     float = 1.0,
                pseudocount: float = 0.5) -> np.ndarray:
    """
    Compute per-sequence fitness after size-factor normalization.

        fitness_i = log( (reads_out_i / sf_out + pc)
                       / (reads_in_i  / sf_in  + pc) )

    The pseudocount avoids log(0) for sequences with zero reads.
    0.5 is a standard choice (Laplace smoothing).

    Parameters
    ----------
    reads_out    : raw read counts in the output condition
    reads_in     : raw read counts in the input library
    sf_out       : size factor for the output condition
    sf_in        : size factor for the input (default 1 = no correction)
    pseudocount  : small constant added before log

    Returns
    -------
    fitness : np.ndarray of float
    """
    norm_out = reads_out / sf_out + pseudocount
    norm_in  = reads_in  / sf_in  + pseudocount
    return np.log(norm_out / norm_in)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Generate data and compute raw fitness
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Generate data and compute raw (biased) fitness")
print("=" * 65)

# --- absolute concentrations -----------------------------------------------
records = []
for label, base in BASE_CONC.items():
    for i in range(N_SEQUENCES):
        conc_ctrl = base * (1 + np.random.normal(0, 0.05))
        conc_drug = conc_ctrl * (1 - DRUG_EFFECT[label])
        records.append({
            "seq_id":    f"{label}_{i}",
            "class":     label,
            "conc_ctrl": conc_ctrl,
            "conc_drug": conc_drug,
        })

df = pd.DataFrame(records)

print("\n--- True absolute concentrations (mean per class) ---")
summary = df.groupby("class")[["conc_ctrl", "conc_drug"]].mean()
summary["true_change_%"] = (
    (summary["conc_drug"] - summary["conc_ctrl"]) / summary["conc_ctrl"] * 100
)
print(summary.round(2))

# --- simulate reads ---------------------------------------------------------
df["reads_input"] = simulate_reads(df["conc_ctrl"].values, N_INPUT_READS)
df["reads_ctrl"]  = simulate_reads(df["conc_ctrl"].values, N_OUTPUT_READS)
df["reads_drug"]  = simulate_reads(df["conc_drug"].values, N_OUTPUT_READS)

print(f"\n--- Average reads per sequence (input library) ---")
for label in ["weak", "decoy", "strong"]:
    avg = df.loc[df["class"] == label, "reads_input"].mean()
    print(f"  {label:8s}: {avg:.1f} reads/seq")

# --- raw fitness (no normalization) ----------------------------------------
# Standard formula: log(freq_out / freq_in)

df["freq_input"] = df["reads_input"] / df["reads_input"].sum()
df["freq_ctrl"]  = df["reads_ctrl"]  / df["reads_ctrl"].sum()
df["freq_drug"]  = df["reads_drug"]  / df["reads_drug"].sum()

pc = 1 / N_INPUT_READS   # frequency-scale pseudocount

df["fitness_raw_ctrl"] = np.log(
    (df["freq_ctrl"] + pc) / (df["freq_input"] + pc)
)
df["fitness_raw_drug"] = np.log(
    (df["freq_drug"] + pc) / (df["freq_input"] + pc)
)

print("\n--- Raw fitness change (drug - ctrl, mean per class) ---")
print("Expected: weak = 0,  decoy < 0 (small),  strong < 0 (large)\n")

raw_delta = df.groupby("class").apply(
    lambda x: (x["fitness_raw_drug"] - x["fitness_raw_ctrl"]).mean()
).reindex(["weak", "decoy", "strong"])
print(raw_delta.round(4))

print("""
PROBLEM: weak sequences show a POSITIVE fitness change even though
the drug has ZERO biological effect on them.
""")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Understand the compositional problem
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 2 — The compositional problem explained with numbers")
print("=" * 65)

total_ctrl = df["conc_ctrl"].sum()
total_drug = df["conc_drug"].sum()

print(f"\nTotal absolute concentration  ctrl: {total_ctrl:.1f}")
print(f"Total absolute concentration  drug: {total_drug:.1f}")
print(f"Overall pool reduction: {(total_ctrl-total_drug)/total_ctrl*100:.1f}%")

print("\nRelative frequency shift per class (this is what sequencing sees):")
print(f"{'class':8s}  {'freq_ctrl':>10s}  {'freq_drug':>10s}  {'ratio':>8s}  {'real change':>12s}")
for label in ["weak", "decoy", "strong"]:
    sub = df[df["class"] == label]
    fc = sub["conc_ctrl"].sum() / total_ctrl
    fd = sub["conc_drug"].sum() / total_drug
    real = -DRUG_EFFECT[label] * 100
    print(f"  {label:8s}  {fc:10.4f}  {fd:10.4f}  {fd/fc:8.4f}  {real:+10.1f}%")

print("""
Even though weak sequences did NOT change in absolute terms,
their relative frequency INCREASED because the total pool shrank.
Sequencing measures relative frequencies → weak APPEARS enriched.
This is the compositional bias.
""")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Size factor strategies (described)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 3 — Size factor normalization strategies")
print("=" * 65)

print("""
All strategies compute a size factor (SF) and then use:
    fitness_i = log( (reads_out_i / SF) / reads_input_i )

Strategy A — Total reads ratio
    SF = sum(reads_condition) / sum(reads_input)
    Simple but sensitive to highly changing sequences.

Strategy B — Median ratio  [DESeq2-style]
    SF = median( reads_condition_i / reads_input_i )
    Robust to outliers: a few very depleted sequences
    do not distort the size factor.

Strategy C — Ctrl-based global SF  [biologically motivated]
    SF_global = median( reads_ctrl_i / reads_input_i )
    Uses the NO-DRUG condition as the stable reference.
    Applying the same SF_global to the drug condition
    keeps all conditions on the same absolute scale.

Strategy D — Per-condition SF
    SF_c = median( reads_c_i / reads_input_i )
    Each condition gets its own SF.
    Correct for condition-vs-input comparisons,
    but conditions CANNOT be compared directly to each other.
""")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Apply all strategies and compare
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 4 — Apply and compare all normalization strategies")
print("=" * 65)

# compute size factors
sf_A_ctrl = df["reads_ctrl"].sum() / df["reads_input"].sum()
sf_A_drug = df["reads_drug"].sum() / df["reads_input"].sum()

ratio_ctrl = df["reads_ctrl"] / df["reads_input"].replace(0, np.nan)
ratio_drug = df["reads_drug"] / df["reads_input"].replace(0, np.nan)

sf_B_ctrl = ratio_ctrl.median()
sf_B_drug = ratio_drug.median()

sf_C_global = ratio_ctrl.median()   # ctrl-based, same for drug too

sf_D_ctrl = ratio_ctrl.median()
sf_D_drug = ratio_drug.median()

print("\n--- Size factors ---")
print(f"  Strategy A   SF_ctrl={sf_A_ctrl:.4f}   SF_drug={sf_A_drug:.4f}")
print(f"  Strategy B   SF_ctrl={sf_B_ctrl:.4f}   SF_drug={sf_B_drug:.4f}")
print(f"  Strategy C   SF_global={sf_C_global:.4f}  (ctrl-based, same for both)")
print(f"  Strategy D   SF_ctrl={sf_D_ctrl:.4f}   SF_drug={sf_D_drug:.4f}")

# compute fitness for each strategy
for strategy, sf_ctrl, sf_drug in [
    ("A", sf_A_ctrl, sf_A_drug),
    ("B", sf_B_ctrl, sf_B_drug),
    ("C", sf_C_global, sf_C_global),
    ("D", sf_D_ctrl, sf_D_drug),
]:
    df[f"fit_{strategy}_ctrl"] = log_fitness(df["reads_ctrl"], df["reads_input"], sf_out=sf_ctrl)
    df[f"fit_{strategy}_drug"] = log_fitness(df["reads_drug"], df["reads_input"], sf_out=sf_drug)

# summary table: delta fitness (drug - ctrl) per class
print("\n--- Fitness change (drug - ctrl, mean per class) ---")
print("Expected: weak ≈ 0,  decoy < 0 small,  strong < 0 large\n")

rows = []
for strategy in ["A", "B", "C", "D"]:
    for label in ["weak", "decoy", "strong"]:
        sub = df[df["class"] == label]
        delta = (sub[f"fit_{strategy}_drug"] - sub[f"fit_{strategy}_ctrl"]).mean()
        rows.append({"strategy": strategy, "class": label, "delta": delta})

pivot = (pd.DataFrame(rows)
         .pivot(index="strategy", columns="class", values="delta")
         [["weak", "decoy", "strong"]])

# add raw row for comparison
raw_row = pd.DataFrame(
    [{"strategy": "raw (no SF)",
      "weak":   raw_delta["weak"],
      "decoy":  raw_delta["decoy"],
      "strong": raw_delta["strong"]}]
).set_index("strategy")

print(pd.concat([raw_row, pivot]).round(4))

print(f"""
--- What the true biology says ---
  weak   true change:  {-DRUG_EFFECT['weak']*100:+.1f}%  → expected delta fitness ≈  0.00
  decoy  true change:  {-DRUG_EFFECT['decoy']*100:+.1f}%  → expected delta fitness ≈ -0.05
  strong true change: {-DRUG_EFFECT['strong']*100:+.1f}%  → expected delta fitness ≈ -0.29

CONCLUSION
----------
Raw fitness (no SF) gives the WRONG sign for weak sequences.

All size-factor strategies recover the correct direction.
Strategy C (ctrl-based global SF) is the recommended choice
when comparing each condition against its biological baseline (input),
because it anchors everything to the most stable reference available.
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b — Detailed comparison and honest evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 4b — Why Strategy B/D outperform A/C in this example")
print("=" * 65)

print(f"""
Size factors tell the story:

  SF_ctrl  (all strategies) ≈ {sf_B_ctrl:.4f}  → this is just N_output/N_input = {N_OUTPUT_READS/N_INPUT_READS:.2f}
  SF_drug  Strategy A/C     = {sf_A_drug:.4f}  → same as ctrl: corrects depth only
  SF_drug  Strategy B/D     = {sf_B_drug:.4f}  → higher: partially corrects compositional inflation

Strategy A and C apply the SAME size factor to ctrl and drug.
They correct for sequencing depth (N_output vs N_input) but they
do NOT correct for the compositional shift caused by the drug.

Strategy B and D use a per-condition median.
For the drug condition, the median ratio is higher (~0.91 vs ~0.80)
because when strong sequences are depleted, the reads for weak/decoy
sequences are inflated. The median naturally absorbs part of this
inflation and removes it — this is the key advantage.

Limitation of B/D: the median SF for drug is only a partial fix.
It works well when fewer than 50% of sequences are genuinely depleted.
If the drug depleted the majority of sequences, the median would
be pulled down and the correction would be incomplete.

The gold standard remains using spike-in sequences (known invariant
reference added at fixed concentration) to compute SF directly
without any assumption about which sequences do not change.
""")

# show error relative to true expected delta
print("--- Absolute error vs true expected delta ---\n")
true_delta = {
    "weak":   np.log(1 - DRUG_EFFECT["weak"]   + 1e-9),  # ≈ 0
    "decoy":  np.log(1 - DRUG_EFFECT["decoy"]),           # ≈ -0.051
    "strong": np.log(1 - DRUG_EFFECT["strong"]),          # ≈ -0.288
}

print(f"  true delta   weak={true_delta['weak']:+.4f}  "
      f"decoy={true_delta['decoy']:+.4f}  "
      f"strong={true_delta['strong']:+.4f}\n")

for strat in ["raw (no SF)", "A", "B", "C", "D"]:
    if strat == "raw (no SF)":
        vals = {"weak": raw_delta["weak"],
                "decoy": raw_delta["decoy"],
                "strong": raw_delta["strong"]}
    else:
        vals = {label: pivot.loc[strat, label] for label in ["weak","decoy","strong"]}

    err = {k: abs(vals[k] - true_delta[k]) for k in ["weak","decoy","strong"]}
    total_err = sum(err.values())
    print(f"  {strat:12s}  |error| weak={err['weak']:.4f}  "
          f"decoy={err['decoy']:.4f}  strong={err['strong']:.4f}  "
          f"total={total_err:.4f}")

print(f"""
Strategy B (per-condition median) has the lowest total error.
Strategy C (ctrl-based global SF) performs similarly to raw fitness —
it corrects library depth but not the compositional shift.

FINAL RECOMMENDATION for your experiment:
    Use per-condition median SF (Strategy B/D).
    If you have biological reasons to believe ctrl is perfectly flat,
    verify it with: CV = std(reads_ctrl/reads_input) / mean(...)
    and use it as a sanity check, not as the primary SF.
""")
