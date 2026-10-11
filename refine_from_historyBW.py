"""
Reuse ALL earlier simulations (history.json from DiscoverJuncBW.py) to find better shapes.

  1. load + merge history files (duplicates removed), print run count, top shapes, what matters
  2. fit a surrogate (random forest) on the shapes already simulated: grid -> log(cost)
  3. generate thousands of candidates around the best shapes (1-cell flips, mutations, crossovers)
  4. rank them with the surrogate (predicted cost minus an exploration bonus), pick a diverse few
  5. simulate only those few in Lumerical, add them to the data, refit, repeat
"""
import os
import json
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# CHANGE: We must import your NEW file, DiscoverJuncBW.py, not shape_discovery.py
import DiscoverJuncBW as sd
from DiscoverJuncBW import NX, NROWS, NY_SLAB

# =============================================================================
# Configuration VVCVCV
# =============================================================================
HISTORY_FILES = ["results/metrics/history_bw.json"]
MERGED_HISTORY = "results/metrics/history_bw_refined.json"
RUN_SIMULATIONS = True
N_ROUNDS = 4                    # refit-and-propose cycles
N_NEW_PER_ROUND = 6             # Lumerical simulations per round  -> total = N_ROUNDS * N_NEW_PER_ROUND
TOP_K_PARENTS = 6               # candidates are built around the K best shapes
N_RANDOM_PER_PARENT = 400       # random 2-flip / block mutations per parent (plus all 1-cell flips)
KAPPA = 1.0                     # exploration: score = predicted log10(cost) - KAPPA * uncertainty
MIN_HAMMING = 2                 # picked shapes must differ from each other by >= this many cells
SEED = 5

DOPING_MODEL_OVERRIDE = None
if DOPING_MODEL_OVERRIDE:
    sd.DOPING_MODEL = DOPING_MODEL_OVERRIDE

# =============================================================================
# Data
# =============================================================================
def load_history(paths):
    recs, seen, skipped = [], set(), {}
    tag = sd.setup_tag()
    for p in paths:
        if not os.path.exists(p):
            print(f"[warn] {p} not found")
            continue
        for r in json.load(open(p)):
            if r.get("setup") != tag:
                skipped[r.get("setup", "untagged")] = skipped.get(r.get("setup", "untagged"), 0) + 1
                continue
            if r["ok"]:
                if sd.USE_BANDWIDTH and "f3db_ghz" not in r:
                    continue
                # NOTE: The cost automatically updates if you changed W_LOSS in DiscoverJuncBW.py!
                c = sd.cost_function(r["vpilpi"], r["avg_loss"], r.get("f3db_ghz"))
                r["ok"], r["cost"] = bool(np.isfinite(c)), (float(c) if np.isfinite(c) else sd.PENALTY)
            k = sd.key_of(np.array(r["grid"], dtype=bool))
            if k not in seen:
                seen.add(k)
                recs.append(r)
    print(f"[info] current setup = {tag}")
    for t, n in skipped.items():
        print(f"[info] skipped {n} records with setup '{t}' (different physics - not comparable)")
    return recs, seen

def show(g):
    return "\n".join("     " + "".join("p" if v else "n" for v in row) for row in g)

def summary(recs):
    ok = [r for r in recs if r["ok"]]
    print(f"\nSimulated shapes: {len(recs)}   successful: {len(ok)}   failed: {len(recs) - len(ok)}")
    print(f"Lumerical solves so far: {len(recs)} x (2 CHARGE + 2 FDE) = {2 * len(recs)} CHARGE + {2 * len(recs)} FDE")
    times = [r["time_s"] for r in recs if "time_s" in r]
    if times:
        print(f"Mean time per shape: {np.mean(times) / 60:.1f} min")
    print("\nTop shapes (p = left side):")
    for i, r in enumerate(sorted(ok, key=lambda r: r["cost"])[:6], 1):
        lab = f" [{r['label']}]" if r["label"] else ""
        bw = f"  f3dB={r['f3db_ghz']:.1f} GHz" if "f3db_ghz" in r else ""
        print(f"  #{i}  cost={r['cost']:.2f}  VpiLpi={r['vpilpi']:.3f} V.cm  loss={r['avg_loss']:.2f} dB/cm{bw}{lab}")
        print(show(np.array(r["grid"], dtype=bool)))

# =============================================================================
# Surrogate model
# =============================================================================
def features(g):
    f = g.astype(float)
    ridge, slab = f[:NROWS - NY_SLAB], f[NROWS - NY_SLAB:]
    wall = np.abs(np.diff(f, axis=1))                       
    cols = np.where(wall > 0)[1]
    return np.concatenate([
        f.ravel(),                                          
        f.sum(axis=1),                                      
        [ridge.mean(), slab.mean(), f.mean(),               
         wall.sum(), np.abs(np.diff(f, axis=0)).sum(),      
         cols.mean() if len(cols) else NX / 2]])             

FEATURE_NAMES = ([f"cell r{r} c{c}" for r in range(NROWS) for c in range(NX)] +
                 [f"p cells in row {r}" for r in range(NROWS)] +
                 ["p fraction ridge", "p fraction slab", "p fraction all",
                  "vertical wall length", "horizontal wall length", "mean wall x"])

def fit_surrogate(recs):
    ok = [r for r in recs if r["ok"]]
    X = np.array([features(np.array(r["grid"], dtype=bool)) for r in ok])
    y = np.log10([r["cost"] for r in ok])
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=1, random_state=SEED, n_jobs=-1).fit(X, y)

def predict(model, grids):
    X = np.array([features(g) for g in grids])
    per_tree = np.stack([t.predict(X) for t in model.estimators_])
    return per_tree.mean(0), per_tree.std(0)

# =============================================================================
# Candidate generation + selection
# =============================================================================
def propose(recs, seen, rng):
    ok = sorted([r for r in recs if r["ok"]], key=lambda r: r["cost"])
    parents = [np.array(r["grid"], dtype=bool) for r in ok[:TOP_K_PARENTS]]

    cands = {}
    def add(g):
        g = sd.repair(g)
        k = sd.key_of(g)
        if k not in seen and k not in cands:
            cands[k] = g

    for p in parents:
        for r in range(NROWS):                              
            for c in range(NX):
                q = p.copy()
                q[r, c] = ~q[r, c]
                add(q)
        for _ in range(N_RANDOM_PER_PARENT):                
            add(sd.mutate(sd.mutate(p, rng), rng))
    for a, b in itertools.combinations(parents, 2):         
        for _ in range(20):
            add(sd.mutate(sd.crossover(a, b, rng), rng))

    grids = list(cands.values())
    model = fit_surrogate(recs)
    mean, std = predict(model, grids)
    score = mean - KAPPA * std                              

    chosen = []
    for i in np.argsort(score):
        g = grids[i]
        if all((g != h).sum() >= MIN_HAMMING for h, _, _ in chosen):
            chosen.append((g, 10 ** mean[i], std[i]))
        if len(chosen) == N_NEW_PER_ROUND:
            break

    imp = np.argsort(model.feature_importances_)[::-1][:6]
    print("\nWhat the data says matters most (surrogate feature importance):")
    for j in imp:
        print(f"   {FEATURE_NAMES[j]:<26s} {model.feature_importances_[j]:.2f}")
    print(f"\n{len(grids)} candidates generated, {len(chosen)} selected for simulation")
    return chosen

# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    records, seen = load_history(HISTORY_FILES)
    if not records:
        raise SystemExit(f"No usable history for doping model '{sd.DOPING_MODEL}' - check HISTORY_FILES "
                         "and DOPING_MODEL_OVERRIDE.")
    summary(records)
    sd.plot_gallery(records, "results/plots/top_shapes_before_refine.png")

    if not RUN_SIMULATIONS:
        for g, pred, std in propose(records, seen, rng):
            print(f"\n  proposed shape - surrogate predicts cost ~ {pred:.1f} (+- {std:.2f} in log10)")
            print(show(g))
        print("\nSet RUN_SIMULATIONS = True to simulate the proposed shapes in Lumerical.")
        raise SystemExit

    lum = sd.get_lumapi()
    charge_app = mode_app = pool = None
    gen0 = max(r["gen"] for r in records)
    try:
        charge_app = lum.DEVICE(hide=not sd.SHOW_GUI)
        mode_app = lum.MODE(hide=not sd.SHOW_GUI)
        if sd.DOPING_MODEL == "pearson4" and sd.USE_LUMERICAL_MOMENTS:
            print("Implant moments from Lumerical:")
            sd.apply_lumerical_moments(charge_app)

        pool = sd.make_pool()                              
        if pool is not None:
            print(f"Parallel evaluation with {sd.N_WORKERS} worker sessions")
        cache = {sd.key_of(np.array(r["grid"], dtype=bool)): r for r in records}

        for rnd in range(1, N_ROUNDS + 1):
            print(f"\n################ Refinement round {rnd}/{N_ROUNDS} ################")
            chosen = propose(records, seen, rng)
            sd.run_candidates([(g, "refine", gen0 + rnd) for g, _, _ in chosen], cache, records, pool,
                              (charge_app, mode_app), MERGED_HISTORY)
            seen.update(cache)
            best_now = min((r for r in records if r["ok"]), key=lambda r: r["cost"])
            print(f"[round {rnd}] best cost so far = {best_now['cost']:.3f}")
        if pool is not None:
            pool.shutdown()
            pool = None

        # ---- final run of the best shape + outputs ---------------------------------------
        best_rec = min((r for r in records if r["ok"]), key=lambda r: r["cost"])
        best_g = np.array(best_rec["grid"], dtype=bool)
        print("\n=== Final CHARGE + FDE simulation of the best shape ===")
        print(show(best_g))
        final = sd.evaluate(charge_app, mode_app, best_g, "best_refined")
        final.update(cost=sd.cost_function(final["vpilpi"], final["avg_loss"], final.get("f3db_ghz")),
                     label=best_rec["label"])
        print(f"  Vpi*Lpi = {final['vpilpi']:.3f} V.cm | loss(0V) = {final['loss0']:.3f}, "
              f"loss(-2V) = {final['loss2']:.3f} dB/cm | cost = {final['cost']:.3f}")
        if "f3db_ghz" in final:
            print(f"  C = {final['C_pF_per_mm']:.3f} pF/mm | R = {final['R_ohm_mm']:.2f} ohm.mm | "
                  f"f3dB(RC) = {final['f3db_ghz']:.1f} GHz")
        dims = sd.report_dimensions(best_g, "results/metrics/best_shape_refined_dimensions.json")

        with open("results/metrics/best_shape_refined.json", "w") as f:
            json.dump({"setup": sd.setup_tag(), "grid_rows_top_to_bottom_p_is_1": best_g.astype(int).tolist(),
                       "dimensions": dims, "metrics": final, "n_simulations_total": len(records)}, f, indent=2)
                       
        sd.plot_full_section(best_g, "results/plots/best_refined_full_section.png")
        sd.plot_best_shape(best_g, f"Best refined junction   cost = {final['cost']:.2f}",
                           "results/plots/best_refined_doping.png",
                           "results/metrics/best_refined_doping.npz")
        sd.plot_gallery(records, "results/plots/top_shapes_refined.png")
        sd.plot_scatter(records, final, "results/plots/scatter_and_convergence_refined.png")
        print(f"\nSaved results, plots and Lumerical projects to results/ folders!")

        plt.show(block=False)
        plt.pause(0.5)
        if sd.SHOW_GUI:
            input("--> Lumerical is still open. Press Enter to close it and exit...")
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        for app in (charge_app, mode_app):
            if app is not None:
                try:
                    app.close()
                except Exception:
                    pass