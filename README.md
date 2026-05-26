$Spliceosome Competition Modeling$

This project focuses on modeling the assembly kinetics of the spliceosome in the presence of SF3B1 modulators (PladB and H3B). The main goal is to understand how Branch Point (BP) sequence features and the presence of "decoy" sites influence binding selectivity under pharmacological stress.

Project Structure -- 
The repository is divided into two main areas that reflect the scientific workflow:

📂 data_analysis/

This section contains tools for processing experimental data extracted from the reference paper.

- Clustering & Stratification: Scripts to group BP sequences based on their behavior (e.g., sensitive, resistant, or decoy-driven).
- Sequence Analysis: A study of BP characteristics (SVM scores, PYT strength, Anchor sequences) to identify the biological determinants of each cluster
- Preprocessing for Fitting: Calculation of the differential fitness ($\Delta Fit$) between sequences with and without decoys to isolate the net contribution of kinetic relocation.

📂 modeling/

This section hosts the mathematical core of the project, where the biological process is translated into a system of kinetic equations.
- 3-Node Model: Implementation of a system based on three main states: Free, BP (Correct Site), and Decoy (Incorrect Site).
- Parametric Fitting: Optimization of kinetic constants ($k_1, k_{-1}, k_2, k_{-2}$) using a step-wise strategy:
- Calibration of basal parameters using control data (DMSO).
- Modeling of BP inhibition using Hill functions.
- Modeling of relocation or "stalling" at decoy sites.
- Simulations: Generation of steady-state probability profiles to mechanistically validate the "loss of selectivity" observed in experiments.


Scientific Objective -
The project aims to demonstrate that relocation toward decoys is not merely a statistical effect of read distribution. Instead, it is a direct consequence of kinetic competition between sites, where the drug drastically alters the energetic barriers required for correct site recognition.
