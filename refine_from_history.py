"""
Reuse ALL earlier simulations to find better shapes (VpiLpi-oriented refinement).

  1. load + merge history files (dedupe by shape; only records of the CURRENT setup), recompute cost
  2. surrogate (random forest) on shape features -> log10(cost); failed runs enter as penalties
  3. candidates around the best (diverse) shapes + the template parents + crossovers
  4. rank by predicted cost minus exploration bonus, simulate a diverse few, refit, repeat
Set RUN_SIMULATIONS = False first: analysis + proposals only (no Lumerical).
"""
import os
import json
import time
import importlib
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

MODULE_NAME = "DiscoverJunction"          # <- file name of your main script (without .py)
sd = importlib.import_module(MODULE_NAME)
NX, NROWS, NY_SLAB = sd.NX, sd.NROWS, sd.NY_SLAB

# =============================================================================
# Configuration
# =============================================================================
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)
MERGED_HISTORY = "results/metrics/history_junction_refined.json"       # written after every simulation
# the merged file is read back too, so refinement runs are never lost; duplicates are merged
HISTORY_FILES = ["results/metrics/history_junction.json", MERGED_HISTORY]

RUN_SIMULATIONS = True
N_ROUNDS = 10
N_NEW_PER_ROUND = 6
TOP_K_PARENTS = 6               # candidates are built around the K best (mutually different) shapes
PARENT_MIN_HAMMING = 4          # those K parents must differ by at least this many cells
INCLUDE_TEMPLATE_PARENTS = True # also build candidates around every SEED_TEMPLATES parent
N_RANDOM_PER_PARENT = 400
KAPPA = 1.0                     # score = predicted log10(cost) - KAPPA * uncertainty
MIN_HAMMING = 2                 # proposed shapes must differ from each other by >= this many cells
FAIL_PENALTY_FACTOR = 3.0       # failed run enters the surrogate as 3 x the worst successful cost
SEED = 5

DOPING_MODEL_OVERRIDE = None
if DOPING_MODEL_OVERRIDE:
    sd.DOPING_MODEL = DOPING_MODEL_OVERRIDE


def cost_of(r):
    """The ONE cost definition (from the main script), used for history, new runs and the final run."""
    return float(sd.cost_function(r["vpilpi"], r["avg_loss"], r.get("f3db_ghz")))


# =============================================================================
# Data
# =============================================================================
def load_history(paths):
    tag = sd.setup_tag()
    recs, seen, skipped = [], set(), 0
    for p in paths:
        if not os.path.exists(p):
            print(f"[warn] {p} not found")
            continue
        for r in json.load(open(p)):
            same = (r["setup"] == tag) if "setup" in r else (r.get("doping", "constant") == sd.DOPING_MODEL)
            if not same:                                   # different physics -> costs not comparable
                skipped += 1
                continue
            k = sd.key_of(np.array(r["grid"], dtype=bool))
            if k in seen:
                continue
            seen.add(k)
            if r.get("ok"):
                try:
                    r["cost"] = cost_of(r)
                except Exception:
                    r["cost"] = float("nan")
                if not (np.isfinite(r["cost"]) and r["cost"] > 0):
                    r["ok"] = False                        # negative / NaN VpiLpi would break log10
                    r["cost"] = sd.PENALTY
            recs.append(r)
    if skipped:
        print(f"[info] skipped {skipped} records of a different setup / doping model "
              "(different doping, mesh, NX ... -> not comparable)")
    return recs, seen


def show(g):
    return "\n".join("     " + "".join("p" if v else "n" for v in row) for row in g)


def summary(recs):
    ok = [r for r in recs if r["ok"]]
    print(f"\nrefine_from_history: simulated shapes: {len(recs)}   successful: {len(ok)}   failed: {len(recs) - len(ok)}")
    times = [r["time_s"] for r in recs if "time_s" in r]
    if times:
        print(f"Mean time per shape: {np.mean(times) / 60:.1f} min")
    print("\nTop shapes (p = left side):")
    for i, r in enumerate(sorted(ok, key=lambda r: r["cost"])[:6], 1):
        lab = f" [{r.get('label', '')}]" if r.get("label") else ""
        print(f"  #{i}  cost={r['cost']:.3f}  VpiLpi={r['vpilpi']:.3f} V.cm  loss={r['avg_loss']:.2f} dB/cm{lab}")
        print(show(np.array(r["grid"], dtype=bool)))


# =============================================================================
# Surrogate model
# =============================================================================
CENTER = slice(max(NX // 2 - 2, 0), min(NX // 2 + 2, NX))        # columns under the optical mode


def features(g):
    f = g.astype(float)
    ridge, slab = f[:NROWS - NY_SLAB], f[NROWS - NY_SLAB:]
    wall_v = np.abs(np.diff(f, axis=1))                     # p/n interface between horizontal neighbours
    wall_h = np.abs(np.diff(f, axis=0))                     # p/n interface between vertical neighbours
    cols = np.where(wall_v > 0)[1]
    alt = np.abs(np.diff(f[:NROWS - NY_SLAB], axis=1)).sum(axis=1)     # p/n alternations per ridge row
    return np.concatenate([
        f.ravel(),
        f.sum(axis=1),
        [ridge.mean(), slab.mean(), f.mean(),
         wall_v.sum(), wall_h.sum(),
         cols.mean() if len(cols) else NX / 2,
         wall_v[:NROWS - NY_SLAB, max(CENTER.start - 1, 0):CENTER.stop].sum(),     # interface length in the mode
         wall_h[:NROWS - NY_SLAB - 1, CENTER].sum() if NROWS - NY_SLAB > 1 else 0.0,
         wall_h[NROWS - NY_SLAB - 1:NROWS - NY_SLAB, :].sum(),                     # interface at the slab step
         alt.max(), alt.mean(),
         ridge[:, CENTER].mean()],                                                # p fraction under the mode
    ])


FEATURE_NAMES = ([f"cell r{r} c{c}" for r in range(NROWS) for c in range(NX)] +
                 [f"p cells in row {r}" for r in range(NROWS)] +
                 ["p fraction ridge", "p fraction slab", "p fraction all",
                  "lateral interface length", "vertical interface length", "mean interface x",
                  "lateral interface in mode", "vertical interface in mode", "interface at slab step",
                  "max alternations/row", "mean alternations/row", "p fraction under mode"])


def fit_surrogate(recs):
    ok_costs = [r["cost"] for r in recs if r["ok"]]
    y_fail = np.log10(FAIL_PENALTY_FACTOR * max(ok_costs))
    X = np.array([features(np.array(r["grid"], dtype=bool)) for r in recs])
    y = np.array([np.log10(r["cost"]) if r["ok"] else y_fail for r in recs])
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=1, random_state=SEED, n_jobs=-1).fit(X, y)


def predict(model, grids):
    X = np.array([features(g) for g in grids])
    per_tree = np.stack([t.predict(X) for t in model.estimators_])
    return per_tree.mean(0), per_tree.std(0)


# =============================================================================
# Candidate generation + selection
# =============================================================================
def pick_parents(recs):
    ok = sorted([r for r in recs if r["ok"]], key=lambda r: r["cost"])
    parents = []
    for r in ok:
        g = np.array(r["grid"], dtype=bool)
        if all((g != q).sum() >= PARENT_MIN_HAMMING for q in parents):
            parents.append(g)
        if len(parents) == TOP_K_PARENTS:
            break
    if INCLUDE_TEMPLATE_PARENTS:
        for g in sd.all_seeds().values():
            if all((g != q).any() for q in parents):
                parents.append(g)
    return parents


def propose(recs, seen, rng):
    parents = pick_parents(recs)
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
        for _ in range(N_RANDOM_PER_PARENT):
            add(sd.mutate(sd.mutate(p, rng), rng))
    for a, b in itertools.combinations(parents[:TOP_K_PARENTS], 2):
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
        print(f"   {FEATURE_NAMES[j]:<30s} {model.feature_importances_[j]:.2f}")
    print(f"\n{len(parents)} parents, {len(grids)} candidates generated, {len(chosen)} selected for simulation")
    return chosen


# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    records, seen = load_history(HISTORY_FILES)
    if not records or not any(r["ok"] for r in records):
        raise SystemExit("No usable successful history for the current setup - check HISTORY_FILES / settings.")
    summary(records)
    sd.plot_gallery(records, "results/plots/top_shapes_before_refine.png")

    if not RUN_SIMULATIONS:
        for g, pred, std in propose(records, seen, rng):
            print(f"\n  proposed shape - surrogate predicts cost ~ {pred:.3f} (+- {std:.2f} in log10)")
            print(show(g))
        print("\nSet RUN_SIMULATIONS = True to simulate the proposed shapes in Lumerical.")
        raise SystemExit

    lum = sd.get_lumapi()
    charge_app = mode_app = None
    gen0 = max(r.get("gen", 0) for r in records)
    tag = sd.setup_tag()
    try:
        charge_app = lum.DEVICE(hide=not sd.SHOW_GUI)
        mode_app = lum.MODE(hide=not sd.SHOW_GUI)
        if sd.DOPING_MODEL == "pearson4" and sd.USE_LUMERICAL_MOMENTS:
            print("Implant moments from Lumerical:")
            sd.apply_lumerical_moments(charge_app)

        for rnd in range(1, N_ROUNDS + 1):
            print(f"\n################ Refinement round {rnd}/{N_ROUNDS} ################")
            for g, pred, std in propose(records, seen, rng):
                print(f"\n=== Simulation {len(records) + 1}  (surrogate predicted ~{pred:.3f}) ===")
                print(show(g))
                rec = {"grid": g.astype(int).tolist(), "label": "refine", "gen": gen0 + rnd,
                       "doping": sd.DOPING_MODEL, "setup": tag}
                t0 = time.time()
                try:
                    r = sd.evaluate(charge_app, mode_app, g, f"refine_{len(records)}")
                    cost = cost_of(r)
                    ok = bool(np.isfinite(cost) and cost > 0)
                    rec.update(r, ok=ok, cost=cost if ok else sd.PENALTY)
                    print(f"  RESULT  Vpi*Lpi = {r['vpilpi']:.3f} V.cm   loss = {r['avg_loss']:.3f} dB/cm   "
                          f"cost = {rec['cost']:.3f}")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    rec.update(ok=False, cost=sd.PENALTY)
                rec["time_s"] = time.time() - t0
                records.append(rec)
                seen.add(sd.key_of(g))
                json.dump(records, open(MERGED_HISTORY, "w"), indent=2)
            best_now = min((r for r in records if r["ok"]), key=lambda r: r["cost"])
            print(f"[round {rnd}] best cost so far = {best_now['cost']:.3f}  (VpiLpi {best_now['vpilpi']:.3f} V.cm)")

        # ---- final run of the best shape + outputs ---------------------------------------
        best_rec = min((r for r in records if r["ok"]), key=lambda r: r["cost"])
        best_g = np.array(best_rec["grid"], dtype=bool)
        print("\n=== Final CHARGE + FDE simulation of the best shape ===")
        print(show(best_g))
        final = sd.evaluate(charge_app, mode_app, best_g, "best_refined")
        final.update(cost=cost_of(final), label=best_rec.get("label", ""))
        print(f"  Vpi*Lpi = {final['vpilpi']:.3f} V.cm | loss(0V) = {final.get('loss0', float('nan')):.3f}, "
              f"loss(-2V) = {final.get('loss2', float('nan')):.3f} dB/cm | cost = {final['cost']:.3f}")

        with open("results/metrics/best_shape_refined.json", "w") as f:
            json.dump({"grid_rows_top_to_bottom_p_is_1": best_g.astype(int).tolist(),
                       "metrics": final, "n_simulations_total": len(records)}, f, indent=2)
        sd.plot_best_shape(best_g, f"Best refined junction   VpiLpi = {final['vpilpi']:.3f} V.cm",
                           "results/plots/best_refined_doping.png",
                           "results/metrics/best_refined_doping.npz")
        sd.plot_gallery(records, "results/plots/top_shapes_refined.png")
        sd.plot_scatter(records, final, "results/plots/scatter_and_convergence_refined.png")
        print("\nSaved results and plots in results/")

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