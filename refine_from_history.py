"""
Reuse ALL earlier simulations (history.json from shape_discovery.py) to find better shapes.

  1. load + merge history files (duplicates removed), print run count, top shapes, what matters
  2. fit a surrogate (random forest) on the shapes already simulated: grid -> log(cost)
  3. generate thousands of candidates around the best shapes (1-cell flips, mutations, crossovers)
  4. rank them with the surrogate (predicted cost minus an exploration bonus), pick a diverse few
  5. simulate only those few in Lumerical, add them to the data, refit, repeat

So every Lumerical run is spent on a shape the data says is promising, and no old run is wasted.
Set RUN_SIMULATIONS = False first: it only analyses your history and prints/plots the proposals.

Needs: numpy, matplotlib, scikit-learn, and shape_discovery.py in the same folder.
"""
import os
import json
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

import DiscoverJunction as sd
from DiscoverJunction import NX, NROWS, NY_SLAB

# =============================================================================
# Configuration
# =============================================================================
# every history file you want to reuse (old runs, backups ...). Duplicated shapes are merged.
HISTORY_FILES = ["results/metrics/history_junction.json"]
MERGED_HISTORY = "results/metrics/history_junction_refined.json" # written after every new simulation

RUN_SIMULATIONS = True         # False = analysis + proposals only (no Lumerical)
N_ROUNDS = 10                   # refit-and-propose cycles
N_NEW_PER_ROUND = 6             # Lumerical simulations per round  -> total = N_ROUNDS * N_NEW_PER_ROUND
TOP_K_PARENTS = 6               # candidates are built around the K best shapes
N_RANDOM_PER_PARENT = 400       # random 2-flip / block mutations per parent (plus all 1-cell flips)
KAPPA = 1.0                     # exploration: score = predicted log10(cost) - KAPPA * uncertainty
MIN_HAMMING = 2                 # picked shapes must differ from each other by >= this many cells
SEED = 5

# Must match the doping model of the shape_discovery run that produced the history:
#   None         -> use whatever shape_discovery.py is currently set to (DOPING_MODEL)
#   "constant"   -> your history comes from the older constant-doping version
#   "pearson4"   -> your history comes from the Pearson IV implant version
DOPING_MODEL_OVERRIDE = None
if DOPING_MODEL_OVERRIDE:
    sd.DOPING_MODEL = DOPING_MODEL_OVERRIDE


# =============================================================================
# Data
# =============================================================================
def load_history(paths):
    recs, seen, skipped = [], set(), 0
    for p in paths:
        if not os.path.exists(p):
            print(f"[warn] {p} not found")
            continue
        for r in json.load(open(p)):
            if r.get("doping", "constant") != sd.DOPING_MODEL:     # costs are only comparable within one model
                skipped += 1
                continue
            k = sd.key_of(np.array(r["grid"], dtype=bool))
            if k not in seen:
                seen.add(k)
                recs.append(r)
    if skipped:
        print(f"[info] skipped {skipped} records made with a different doping model than '{sd.DOPING_MODEL}'")
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
        print(f"  #{i}  cost={r['cost']:.2f} V.dB  VpiLpi={r['vpilpi']:.3f} V.cm  loss={r['avg_loss']:.2f} dB/cm{lab}")
        print(show(np.array(r["grid"], dtype=bool)))


# =============================================================================
# Surrogate model
# =============================================================================
def features(g):
    f = g.astype(float)
    ridge, slab = f[:NROWS - NY_SLAB], f[NROWS - NY_SLAB:]
    wall = np.abs(np.diff(f, axis=1))                       # vertical p/n walls
    cols = np.where(wall > 0)[1]
    return np.concatenate([
        f.ravel(),                                          # every cell
        f.sum(axis=1),                                      # p cells per row
        [ridge.mean(), slab.mean(), f.mean(),               # p fraction (ridge, slab, all)
         wall.sum(), np.abs(np.diff(f, axis=0)).sum(),      # wall length (vertical, horizontal)
         cols.mean() if len(cols) else NX / 2]])             # mean x of the walls


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
        for r in range(NROWS):                              # every single-cell flip
            for c in range(NX):
                q = p.copy()
                q[r, c] = ~q[r, c]
                add(q)
        for _ in range(N_RANDOM_PER_PARENT):                # multi-cell / block mutations
            add(sd.mutate(sd.mutate(p, rng), rng))
    for a, b in itertools.combinations(parents, 2):         # crossovers between good shapes
        for _ in range(20):
            add(sd.mutate(sd.crossover(a, b, rng), rng))

    grids = list(cands.values())
    model = fit_surrogate(recs)
    mean, std = predict(model, grids)
    score = mean - KAPPA * std                              # low predicted cost OR high uncertainty

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
            print(f"\n  proposed shape - surrogate predicts cost ~ {pred:.1f} V.dB (+- {std:.2f} in log10)")
            print(show(g))
        print("\nSet RUN_SIMULATIONS = True to simulate the proposed shapes in Lumerical.")
        raise SystemExit

    lum = sd.get_lumapi()
    charge_app = mode_app = None
    gen0 = max(r["gen"] for r in records)
    try:
        charge_app = lum.DEVICE(hide=not sd.SHOW_GUI)
        mode_app = lum.MODE(hide=not sd.SHOW_GUI)
        if sd.DOPING_MODEL == "pearson4" and sd.USE_LUMERICAL_MOMENTS:
            print("Implant moments from Lumerical:")
            sd.apply_lumerical_moments(charge_app)

        for rnd in range(1, N_ROUNDS + 1):
            print(f"\n################ Refinement round {rnd}/{N_ROUNDS} ################")
            for g, pred, std in propose(records, seen, rng):
                print(f"\n=== Simulation {len(records) + 1}  (surrogate predicted ~{pred:.1f} V.dB) ===")
                print(show(g))
                rec = {"grid": g.astype(int).tolist(), "label": "refine", "gen": gen0 + rnd, "doping": sd.DOPING_MODEL}
                t0 = time.time()
                try:
                    r = sd.evaluate(charge_app, mode_app, g, "eval")
                    cost = r["vpilpi"] * abs(r["avg_loss"])
                    ok = bool(np.isfinite(cost))
                    rec.update(r, ok=ok, cost=float(cost) if ok else sd.PENALTY)
                    print(f"  RESULT  Vpi*Lpi = {r['vpilpi']:.3f} V.cm   loss = {r['avg_loss']:.3f} dB/cm   "
                          f"cost = {rec['cost']:.3f} V.dB")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    rec.update(ok=False, cost=sd.PENALTY)
                rec["time_s"] = time.time() - t0
                records.append(rec)
                seen.add(sd.key_of(g))
                json.dump(records, open(MERGED_HISTORY, "w"), indent=2)
            best_now = min((r for r in records if r["ok"]), key=lambda r: r["cost"])
            print(f"[round {rnd}] best cost so far = {best_now['cost']:.3f} V.dB")

        # ---- final run of the best shape + outputs ---------------------------------------
        best_rec = min((r for r in records if r["ok"]), key=lambda r: r["cost"])
        best_g = np.array(best_rec["grid"], dtype=bool)
        print("\n=== Final CHARGE + FDE simulation of the best shape ===")
        print(show(best_g))
        final = sd.evaluate(charge_app, mode_app, best_g, "best_refined")
        final.update(cost=final["vpilpi"] * abs(final["avg_loss"]), label=best_rec["label"])
        print(f"  Vpi*Lpi = {final['vpilpi']:.3f} V.cm | loss(0V) = {final['loss0']:.3f}, "
              f"loss(-2V) = {final['loss2']:.3f} dB/cm | cost = {final['cost']:.3f} V.dB")

        with open("results/metrics/best_shape_refined.json", "w") as f:
            json.dump({"grid_rows_top_to_bottom_p_is_1": best_g.astype(int).tolist(),
                       "metrics": final, "n_simulations_total": len(records)}, f, indent=2)
                       
        sd.plot_best_shape(best_g, f"Best refined junction   cost = {final['cost']:.2f} V.dB",
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
        for app in (charge_app, mode_app):
            if app is not None:
                try:
                    app.close()
                except Exception:
                    pass