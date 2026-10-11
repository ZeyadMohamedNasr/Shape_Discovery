# PN Junction Shape Discovery - Script Overview

## 1. DiscoverJuncBW.py
* Main genetic algorithm script optimizing PN junction shapes for both efficiency (VpiLpi) and RC bandwidth (f3dB).
* Supports parallel Lumerical worker sessions (`N_WORKERS`) for faster shape evaluation.
* Calculates capacitance and series resistance using CHARGE voltage sweeps (0V to -2V).
* Implements complex predefined seed genomes (custom W-shape) to kickstart generation 0.
* Outputs comprehensive metrics, RC bandwidth constraints, and convergence plots.

## 2. DiscoverJunction.py
* Simplified genetic algorithm script focused strictly on optimizing VpiLpi and optical loss.
* Omits bandwidth (f3dB) and capacitance calculations for faster, lighter FDE/CHARGE runs.
* Evaluates designs strictly at 0V and -2V using independent Lumerical projects.
* Uses standard predefined generic seed shapes (vertical, M-like, stacked, comb).
* Best suited for basic efficiency optimizations where RF bandwidth is not a constraint.

## 3. DiscoverdiffParents.py
* Modified genetic algorithm script focused on injecting specialized parent shapes.
* Introduces unique L-shape, S-shape, and U-shape templates into the first generation.
* Uses template resampling to dynamically adapt complex shapes to any NX/NY grid size.
* Contains bandwidth calculation logic but keeps it disabled by default (`USE_BANDWIDTH = False`).
* Designed to explore alternative topological starting points beyond standard rectilinear grids.

## 4. refine_from_historyBW.py
* Advanced refinement script paired specifically with the output of `DiscoverJuncBW.py`.
* Trains a Random Forest surrogate model on previously simulated shapes to predict log(cost).
* Generates thousands of virtual mutations from top parents without opening Lumerical.
* Selects the most promising, diverse candidates based on predictions and exploration uncertainty.
* Evaluates selected candidates in parallel Lumerical sessions via multiprocessing.

## 5. refine_from_history_2.py
* Standard refinement script paired specifically with the simplified `DiscoverJunction.py`.
* Filters history files to match the current doping model (e.g., pearson4) before training.
* Predicts basic VpiLpi and loss costs without requiring f3dB bandwidth metrics.
* Runs sequential Lumerical evaluations without multiprocessing for simple debugging.
* Refines top shapes without testing random unguided mutations.
