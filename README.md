(
echo # PN Junction Shape Discovery - Script Overview
echo.
echo ## 1. DiscoverJuncBW.py
echo * Main genetic algorithm script optimizing PN junction shapes for both efficiency (VpiLpi) and RC bandwidth (f3dB).
echo * Supports parallel Lumerical worker sessions (N_WORKERS) for faster shape evaluation.
echo * Calculates capacitance and series resistance using CHARGE voltage sweeps (0V to -2V).
echo * Implements complex predefined seed genomes (custom W-shape) to kickstart generation 0.
echo * Outputs comprehensive metrics, RC bandwidth constraints, and convergence plots.
echo.
echo ## 2. DiscoverJunction.py
echo * Simplified genetic algorithm script focused strictly on optimizing VpiLpi and optical loss.
echo * Omits bandwidth (f3dB) and capacitance calculations for faster, lighter FDE/CHARGE runs.
echo * Evaluates designs strictly at 0V and -2V using independent Lumerical projects.
echo * Uses standard predefined generic seed shapes (vertical, M-like, stacked, comb).
echo * Best suited for basic efficiency optimizations where RF bandwidth is not a constraint.
echo.
echo ## 3. DiscoverdiffParents.py
echo * Modified genetic algorithm script focused on injecting specialized parent shapes.
echo * Introduces unique L-shape, S-shape, and U-shape templates into the first generation.
echo * Uses template resampling to dynamically adapt complex shapes to any NX/NY grid size.
echo * Contains bandwidth calculation logic but keeps it disabled by default (USE_BANDWIDTH = False).
echo * Designed to explore alternative topological starting points beyond standard rectilinear grids.
echo.
echo ## 4. refine_from_historyBW.py
echo * Advanced refinement script paired specifically with the output of DiscoverJuncBW.py.
echo * Trains a Random Forest surrogate model on previously simulated shapes to predict log(cost).
echo * Generates thousands of virtual mutations from top parents without opening Lumerical.
echo * Selects the most promising, diverse candidates based on predictions and exploration uncertainty.
echo * Evaluates selected candidates in parallel Lumerical sessions via multiprocessing.
echo.
echo ## 5. refine_from_history_2.py
echo * Standard refinement script paired specifically with the simplified DiscoverJunction.py.
echo * Filters history files to match the current doping model (e.g., pearson4) before training.
echo * Predicts basic VpiLpi and loss costs without requiring f3dB bandwidth metrics.
echo * Runs sequential Lumerical evaluations without multiprocessing for simple debugging.
echo * Refines top shapes without testing random unguided mutations.
echo.
echo ## 6. refine_from_history.py (Obsolete)
echo * Outdated refinement script containing a broken import statement (import shape_discovery).
echo * Causes ModuleNotFoundError due to script renaming; superseded by refine_from_history_2.py.
echo * Excluded from active repository tracking.
) > README.md