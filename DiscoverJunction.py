"""
Shape discovery for a lateral PN junction in a silicon rib phase shifter.

The doping of the 500 x 220 nm core is a coarse GRID of rectangular cells, each p or n.
A genetic algorithm evolves the grid (row/column crossover + bit flips + block flips), so the
interface can become an M, an E, a comb, a staircase, an L, ... whatever scores best.

  every candidate:  repair (no floating p/n islands) -> merge cells into doping boxes
                    -> CHARGE @ 0 V and -2 V -> export n/p -> MODE (FDE) @ 0 V and -2 V
  cost = Vpi*Lpi [V.cm] x loss [dB/cm]   (lower is better)

Doping: by default the p and n regions follow an ion-implant depth profile (Pearson IV, measured from
the top silicon surface) instead of a constant value; P_DOPING / N_DOPING are the PEAK concentrations.
Set DOPING_MODEL = "constant" for uniform doping.

Seeds in the first generation: vertical, M-like, stacked, comb and W-shape (p pocket + n tooth), plus
the best shapes of earlier runs (OLD_HISTORY_FILES), so the GA starts from known designs and you can
see directly whether the evolved shape beats them.
Lumerical stays open for the whole run (same lumapi usage as lateral_junc.py).
"""
import os
import json
import time
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================
LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python\lumapi.py"
PROJECT_DIR = os.path.abspath(".")  # Assuming you run scripts from the Shape_Discovery folder
SIM_DIR = os.path.join(PROJECT_DIR, "simulations")
DATA_DIR = os.path.join(PROJECT_DIR, "data", "carriers")
RES_DIR = os.path.join(PROJECT_DIR, "results")

for d in [SIM_DIR, DATA_DIR, RES_DIR]:
    os.makedirs(d, exist_ok=True)
    SHOW_GUI = True                 # True = Lumerical windows stay open and visible

# Device geometry
CORE_WIDTH = 500e-9
CORE_HEIGHT = 220e-9
SLAB_HEIGHT = 90e-9
SLAB_WIDTH = 6.0e-6
Z_SPAN = 2.0e-6

Y_BOT = -CORE_HEIGHT / 2.0
Y_TOP = CORE_HEIGHT / 2.0
Y_SLAB = Y_BOT + SLAB_HEIGHT
M = 50e-9                       # doping boxes overshoot the silicon by this much

# Domains
CHARGE_OXIDE_Y_SPAN = 1.0e-6
MODE_OXIDE_Y_SPAN = 4.0e-6
FDE_X_SPAN = CORE_WIDTH + 2.0e-6
FDE_Y_SPAN = 2.0e-6
MON_X_SPAN = FDE_X_SPAN
MON_Y_SPAN = 0.6e-6

# Doping (cm^-3)
CM3_TO_M3 = 1e6
P_DOPING = 5e17
N_DOPING = 5e17
P_PLUS = 1e20
N_PLUS = 1e20
CONTACT_START = 1.5e-6
CONTACT_WIDTH = 0.6e-6

# ---- Doping model ----------------------------------------------------------------------------
# "pearson4": doping follows an implant depth profile measured from the local top silicon surface
#             (ridge top inside the core, slab top outside it). P_DOPING / N_DOPING = PEAK value.
# "constant": uniform P_DOPING / N_DOPING.
DOPING_MODEL = "pearson4"
SLICE = 25e-9                   # the profile is built from constant-doping slices of about this thickness
# One dict per implant (several energies may be listed; they are summed and rescaled so max = peak).
#   w = relative weight, Rp = projected range, dRp = straggle, skew = skewness,
#   kurt = kurtosis (None -> 3 + 2.4*skew^2, the value Lumerical suggests when it is unknown).
# ILLUSTRATIVE numbers: use your process values, or set USE_LUMERICAL_MOMENTS = True below.
P_IMPLANTS = [dict(w=1.0, Rp=110e-9, dRp=50e-9, skew=-0.4, kurt=None)]    # boron-like
N_IMPLANTS = [dict(w=1.0, Rp=110e-9, dRp=50e-9, skew=0.3, kurt=None)]     # phosphorus-like
USE_LUMERICAL_MOMENTS = False   # True: ask Lumerical for Rp, dRp, skew, kurt at the energies below
P_ENERGY_EV = 40e3              # boron energy
N_ENERGY_EV = 80e3              # phosphorus energy

# ---- Shape grid ----------------------------------------------------------------------------
# The core area (x: +-250 nm, y: -110..+110 nm) is cut into NX columns and NY_SLAB + NY_RIDGE rows.
# Row edges are aligned with the slab top, so cells never straddle the rib/slab step.
# Smallest feature = cell size (here 50 nm x 43-45 nm). More cells = richer shapes, bigger search.
NX = 10
NY_SLAB = 2                     # rows between the bottom and the slab top
NY_RIDGE = 3                    # rows between the slab top and the top of the core
NROWS = NY_SLAB + NY_RIDGE

# ---- Genetic algorithm ---------------------------------------------------------------------
POP_SIZE = 5
N_GENERATIONS = 6               # max simulated designs ~ POP_SIZE * (N_GENERATIONS + 1)
N_ELITE = 2
# Best shapes of earlier runs (e.g. a constant-doping run) are re-simulated with the current doping model
# and inserted in generation 0.  Example: ["./Shape_Discovery/history_backup_20250101_120000.json"]
OLD_HISTORY_FILES = []
N_OLD_SEEDS = 3
P_MUTATE_BIT = 2.0 / (NX * NROWS)   # ~2 single-cell flips per child
P_MUTATE_BLOCK = 0.35               # chance of flipping a random rectangle (keeps shapes blocky)
SEED = 3
PENALTY = 1e6

# Operating point / mesh
VOLTAGE_SWING = 2.0
LAMBDA = 1.55e-6
GLOBAL_MAX_EDGE = 100e-9
GLOBAL_MIN_EDGE = 1e-9
MESH_JUNCTION = 8e-9
MESH_OPTICAL = 25e-9
FDE_MESH = 10e-9
N_MODES = 3
N_CLAD = 1.45

os.makedirs(PROJECT_DIR, exist_ok=True)

def get_settings():
    """Return the current doping configuration as a dictionary."""
    import copy
    return {
        "P_DOPING": P_DOPING,
        "N_DOPING": N_DOPING,
        "P_PLUS": P_PLUS,
        "N_PLUS": N_PLUS,
        "P_IMPLANTS": copy.deepcopy(P_IMPLANTS),
        "N_IMPLANTS": copy.deepcopy(N_IMPLANTS),
        "DOPING_MODEL": DOPING_MODEL
    }

def set_settings(s):
    """Update the global doping configuration and rebuild the physical profiles."""
    global P_DOPING, N_DOPING, P_PLUS, N_PLUS, P_IMPLANTS, N_IMPLANTS, DOPING_MODEL
    
    if "P_DOPING" in s: P_DOPING = s["P_DOPING"]
    if "N_DOPING" in s: N_DOPING = s["N_DOPING"]
    if "P_PLUS" in s: P_PLUS = s["P_PLUS"]
    if "N_PLUS" in s: N_PLUS = s["N_PLUS"]
    if "P_IMPLANTS" in s: P_IMPLANTS = s["P_IMPLANTS"]
    if "N_IMPLANTS" in s: N_IMPLANTS = s["N_IMPLANTS"]
    if "DOPING_MODEL" in s: DOPING_MODEL = s["DOPING_MODEL"]
    
    build_profiles()  # Crucial: This forces the script to recalculate the new Pearson curves
# =============================================================================
# Implant (Pearson IV) doping profile
# =============================================================================
def pearson4(depth, Rp, dRp, skew, kurt=None):
    """Relative Pearson-IV implant profile (maximum = 1) on an increasing depth grid [m].

    Pearson system:  d ln f / dy = -(y + c1) / (c0 + c1*y + c2*y^2),  y = depth - Rp,
    with c0, c1, c2 fixed by the straggle, skewness and kurtosis. Integrated numerically.
    """
    if kurt is None:
        kurt = 3.0 + 2.4 * skew ** 2
    den = 10.0 * kurt - 12.0 * skew ** 2 - 18
    c0 = dRp ** 2 * (4.0 * kurt - 3.0 * skew ** 2) / den
    c1 = skew * dRp * (kurt + 3.0) / den
    c2 = (2.0 * kurt - 3.0 * skew ** 2 - 6.0) / den
    y = depth - Rp
    quad = c0 + c1 * y + c2 * y ** 2
    if den <= 0 or np.any(quad <= 0):
        raise ValueError("These moments do not give a valid Pearson profile (check skew / kurt).")
    g = -(y + c1) / quad
    lnf = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(y))])
    lnf -= np.interp(0.0, y, lnf)                      # anchor at the mean
    return np.exp(lnf - lnf.max())


def make_profile(implants, peak_cm3):
    """depth [m] -> concentration [cm^-3]; sum of the implants, rescaled so the maximum equals `peak`."""
    d = np.linspace(0.0, CORE_HEIGHT + 100e-9, 3001)
    total = sum(i.get("w", 1.0) * pearson4(d, i["Rp"], i["dRp"], i["skew"], i.get("kurt")) for i in implants)
    total = total / total[d <= CORE_HEIGHT].max()
    return lambda depth: peak_cm3 * np.interp(depth, d, total)


PROFILES = {}


def build_profiles():
    PROFILES["p"] = make_profile(P_IMPLANTS, P_DOPING)
    PROFILES["n"] = make_profile(N_IMPLANTS, N_DOPING)


build_profiles()


def lumerical_moments(app, dopant, energy_ev):
    app.eval(f'imp_r = implantrange("{dopant}", "silicon", {energy_ev}); '
             f'imp_s = implantstraggle("{dopant}", "silicon", {energy_ev}); '
             f'imp_g = implantskewness("{dopant}", "silicon", {energy_ev}); '
             f'imp_k = implantkurtosis("{dopant}", "silicon", {energy_ev});')
    return dict(Rp=float(app.getv("imp_r")), dRp=float(app.getv("imp_s")),
                skew=float(app.getv("imp_g")), kurt=float(app.getv("imp_k")))


def apply_lumerical_moments(app):
    """Replace the implant moments by the values Lumerical computes for the chosen energies."""
    for imps, dopant, energy in ((P_IMPLANTS, "boron", P_ENERGY_EV), (N_IMPLANTS, "phosphorous", N_ENERGY_EV)):
        try:
            imps[:] = [dict(w=1.0, **lumerical_moments(app, dopant, energy))]
            print(f"  {dopant} {energy / 1e3:.0f} keV: " + ", ".join(f"{k}={v:.3g}" for k, v in imps[0].items()))
        except Exception as e:
            print(f"  [warn] could not get {dopant} moments from Lumerical ({e}); keeping the config values")
    build_profiles()


# =============================================================================
# Shape representation   (genome: bool array [NROWS, NX], row 0 = top, True = p-type)
# =============================================================================
HC = CORE_WIDTH / 2.0
X_EDGES = np.linspace(-HC, HC, NX + 1)
_ye_slab = np.linspace(Y_BOT, Y_SLAB, NY_SLAB + 1)
_ye_ridge = np.linspace(Y_SLAB, Y_TOP, NY_RIDGE + 1)
Y_EDGES = np.concatenate([_ye_slab, _ye_ridge[1:]])          # ascending, NROWS + 1 values
ROW_Y = [(Y_EDGES[NROWS - 1 - r], Y_EDGES[NROWS - r]) for r in range(NROWS)]   # row 0 = top
SLAB_ROWS = list(range(NROWS - NY_SLAB, NROWS))              # rows that touch the outer slabs


def _reach(mask, seeds):
    """4-connected flood fill over `mask` starting from the cells in `seeds`."""
    seen = np.zeros_like(mask)
    stack = [s for s in seeds if mask[s]]
    for s in stack:
        seen[s] = True
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= rr < NROWS and 0 <= cc < NX and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))
    return seen


def repair(g):
    """Remove floating regions: every p cell must reach the left slab, every n cell the right slab."""
    g = g.copy()
    p_seeds = [(r, 0) for r in SLAB_ROWS]
    n_seeds = [(r, NX - 1) for r in SLAB_ROWS]
    g[~_reach(g, p_seeds) & g] = False          # unreachable p  -> n
    g[~_reach(~g, n_seeds) & ~g] = True         # unreachable n  -> p
    return g


def key_of(g):
    return g.tobytes()


def from_p_counts(fracs):
    """Staircase genome: row r has round(frac*NX) p columns on the left (fracs listed top -> bottom)."""
    fr = np.interp(np.linspace(0, 1, NROWS), np.linspace(0, 1, len(fracs)), fracs)
    g = np.zeros((NROWS, NX), dtype=bool)
    for r, f in enumerate(fr):
        g[r, :int(round(f * NX))] = True
    return repair(g)


SEED_SHAPES = {                                   # p fraction per row, top -> bottom
    "vertical": [0.5, 0.5, 0.5, 0.5, 0.5],
    "M-like":   [1.0, 0.4, 0.7, 0.4, 0.5],
    "stacked":  [1.0, 1.0, 1.0, 0.1, 0.1],
    "comb":     [0.3, 0.7, 0.3, 0.7, 0.3],
}


def w_shape_genome(x_left=2, w2=3, w1=3, h_left=2, h_right=3):
    """W-shaped interface (all sizes in cells; rows counted from the top).

      x_left : p columns left of the n tooth      w2 : width of the n tooth      (image: W2)
      w1     : width of the p pocket right of it  (image: W1)
      h_left : rows of p above the n tooth        (image: H-left)
      h_right: rows the p pocket reaches down     (image: H-right)
    """
    g = np.zeros((NROWS, NX), dtype=bool)
    g[:, :x_left] = True                                       # left wall, p all the way down
    g[:h_left, x_left:x_left + w2] = True                      # p cap above the n tooth
    g[:h_right, x_left + w2:x_left + w2 + w1] = True           # p pocket beside the tooth
    return repair(g)


SEED_GENOMES = {"W-shape": w_shape_genome()}                   # shapes a row-count cannot describe


def all_seeds():
    seeds = {name: from_p_counts(fr) for name, fr in SEED_SHAPES.items()}
    seeds.update(SEED_GENOMES)
    return seeds


def load_old_seeds(files, n):
    """Best n shapes of earlier history files -> [(genome, label)] (re-simulated with the current model)."""
    recs, seen = [], set()
    for f in files:
        for r in json.load(open(f)):
            g = np.array(r["grid"], dtype=bool)
            if r.get("ok") and key_of(g) not in seen:
                seen.add(key_of(g))
                recs.append((r["cost"], g))
    recs.sort(key=lambda t: t[0])
    return [(g, f"old#{i + 1}") for i, (_, g) in enumerate(recs[:n])]


def grid_to_boxes(g):
    """Merge horizontal runs of equal type into doping boxes (non-overlapping)."""
    W2, CS = SLAB_WIDTH / 2.0, CONTACT_START
    yb, yt = Y_BOT - M, Y_TOP + M
    boxes = [
        ("p_plus", "p", P_PLUS, -W2 - M, -CS, yb, Y_SLAB + M),
        ("n_plus", "n", N_PLUS, CS, W2 + M, yb, Y_SLAB + M),
        ("p_out", "p", P_DOPING, -CS, -HC, yb, Y_SLAB + M),      # slab left of the core
        ("n_out", "n", N_DOPING, HC, CS, yb, Y_SLAB + M),        # slab right of the core
    ]
    for r in range(NROWS):
        ylo, yhi = ROW_Y[r]
        if r == 0:
            yhi = Y_TOP + M
        if r == NROWS - 1:
            ylo = Y_BOT - M
        c = 0
        while c < NX:
            c2 = c
            while c2 + 1 < NX and g[r, c2 + 1] == g[r, c]:
                c2 += 1
            is_p = bool(g[r, c])
            boxes.append((f"{'p' if is_p else 'n'}_r{r}_c{c}", "p" if is_p else "n",
                          P_DOPING if is_p else N_DOPING, X_EDGES[c], X_EDGES[c2 + 1], ylo, yhi))
            c = c2 + 1
    return apply_doping_model(boxes)


def apply_doping_model(boxes):
    """Constant model: unchanged. Pearson4: every box is cut into depth slices of constant doping whose
    value follows the implant profile at the slice depth (contacts stay uniform p++/n++)."""
    if DOPING_MODEL == "constant":
        return boxes
    out = []
    for name, dtype, conc, x0, x1, y0, y1 in boxes:
        if name.endswith("_plus"):
            out.append((name, dtype, conc, x0, x1, y0, y1))
            continue
        top = Y_SLAB if name.endswith("_out") else Y_TOP       # implant enters from the local top surface
        segs = []
        if y0 < Y_BOT:                                          # overshoot below the silicon
            segs.append((y0, min(Y_BOT, y1)))
        s0, s1 = max(y0, Y_BOT), min(y1, top)
        if s1 > s0:                                             # silicon part, sliced
            e = np.linspace(s0, s1, int(np.ceil((s1 - s0) / SLICE - 1e-9)) + 1)
            segs += list(zip(e[:-1], e[1:]))
        if y1 > top:                                            # overshoot above the surface
            segs.append((max(y0, top), y1))
        for k, (a, b) in enumerate(segs):
            depth = np.clip(top - 0.5 * (a + b), 0.0, top - Y_BOT)
            out.append((f"{name}_s{k}", dtype, max(float(PROFILES[dtype](depth)), 1e14), x0, x1, a, b))
    return out


# =============================================================================
# Lumerical helpers (same calls as the reference script)
# =============================================================================
def get_lumapi():
    spec = importlib.util.spec_from_file_location("lumapi", LUMAPI_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def try_set(app, names, value, required=True):
    if isinstance(names, str):
        names = (names,)
    last = None
    for n in names:
        try:
            app.set(n, value)
            return n
        except Exception as e:
            last = e
    if required:
        raise RuntimeError(f"None of the properties {names} could be set to {value!r}: {last}")
    return None


def add_rect(app, name, material, x0, x1, y0, y1, order, override_order=False):
    app.addrect()
    app.set("name", name)
    app.set("material", material)
    app.set("x min", x0)
    app.set("x max", x1)
    app.set("y min", y0)
    app.set("y max", y1)
    app.set("z", 0)
    app.set("z span", Z_SPAN)
    if override_order:  # MODE only
        try_set(app, "override mesh order from material database", True, required=False)
    app.set("mesh order", order)


def add_doping(app, name, dtype, conc_cm3, x0, x1, y0, y1):
    app.adddope()
    app.set("name", name)
    app.set("dopant type", dtype)
    app.set("concentration", conc_cm3 * CM3_TO_M3)   # script API uses /m^3
    app.set("x min", x0)
    app.set("x max", x1)
    app.set("y min", y0)
    app.set("y max", y1)
    app.set("z", 0)
    app.set("z span", Z_SPAN)


# =============================================================================
# CHARGE
# =============================================================================
def add_mesh_region(charge, name, xspan, yspan, max_edge):
    try:
        charge.addmesh()
        charge.set("name", name)
        charge.set("x", 0.0)
        charge.set("x span", xspan)
        charge.set("y", 0.0)
        charge.set("y span", yspan)
        try_set(charge, "z", 0.0, required=False)
        try_set(charge, "z span", Z_SPAN, required=False)
        charge.set("max edge length", max_edge)
    except Exception as e:
        print(f"     [mesh] WARNING: could not add mesh region '{name}': {e}")


def configure_solver(charge):
    charge.addchargesolver()
    done = False
    for _ in range(2):
        for cand in ("simulation region", "CHARGE::simulation region", "CHARGE"):
            try:
                charge.select(cand)
                charge.set("dimension", "2D Z-normal")
                done = True
                break
            except Exception:
                continue
        if done:
            break
        try:
            charge.addsimulationregion()
            charge.set("name", "simulation region")
        except Exception:
            pass
    if not done:
        raise RuntimeError("Could not find the CHARGE simulation region object.")

    charge.set("x", 0.0)
    charge.set("x span", SLAB_WIDTH + 2e-6)
    charge.set("y", 0.0)
    charge.set("y span", CHARGE_OXIDE_Y_SPAN)

    charge.select("CHARGE")
    try_set(charge, "max edge length", GLOBAL_MAX_EDGE, required=False)
    try_set(charge, "min edge length", GLOBAL_MIN_EDGE, required=False)

    add_mesh_region(charge, "mesh_optical", 2.6e-6, 0.3e-6, MESH_OPTICAL)
    add_mesh_region(charge, "mesh_junction", 0.5e-6, 0.3e-6, MESH_JUNCTION)



def run_charge(charge, boxes, voltage, tag):
    print(f"  -> CHARGE  V={voltage:+.1f} V  ({len(boxes)} doping boxes)")
    t0 = time.time()
    charge.eval("newproject;")

    # Materials
    charge.eval("addmodelmaterial; set('name', 'Si_Electrical'); addmaterialproperties('CT', 'Si (Silicon)');")
    charge.addmodelmaterial()
    charge.set("name", "SiO2_Electrical")
    try:
        charge.addmaterialproperties("CT", "SiO2 (Glass) - Sze")
    except Exception:
        charge.addctmaterialproperty("Insulator")
        charge.set("name", "SiO2_ins")
        for prop in ("permittivity", "dc permittivity", "eps dc", "relative permittivity"):
            try:
                charge.set(prop, 3.9)
                break
            except Exception:
                continue
    charge.eval("""
        addmodelmaterial; set('name', 'Metal_Electrical');
        addctmaterialproperty('Conductor'); set('name', 'Al');
        set('work function', 4.28);
    """)

    # Geometry
    W2, hc = SLAB_WIDTH / 2.0, CORE_WIDTH / 2.0
    add_rect(charge, "Background_SiO2", "SiO2_Electrical",
             -(SLAB_WIDTH + 2e-6) / 2, (SLAB_WIDTH + 2e-6) / 2,
             -CHARGE_OXIDE_Y_SPAN / 2, CHARGE_OXIDE_Y_SPAN / 2, 5)
    add_rect(charge, "Si_Slab", "Si_Electrical", -W2, W2, Y_BOT, Y_SLAB, 3)
    add_rect(charge, "Si_Core", "Si_Electrical", -hc, hc, Y_BOT, Y_TOP, 2)

    mx0, mx1 = W2 - CONTACT_WIDTH, W2
    add_rect(charge, "Anode_Metal", "Metal_Electrical", -mx1, -mx0, Y_SLAB, Y_SLAB + 100e-9, 1)
    add_rect(charge, "Cathode_Metal", "Metal_Electrical", mx0, mx1, Y_SLAB, Y_SLAB + 100e-9, 1)

    configure_solver(charge)

    # Contacts (anode = p side)
    charge.addelectricalcontact()
    charge.set("name", "anode")
    charge.set("surface type", "solid")
    charge.set("solid", "Anode_Metal")
    charge.set("sweep type", "single")
    charge.set("voltage", float(voltage))

    charge.addelectricalcontact()
    charge.set("name", "cathode")
    charge.set("surface type", "solid")
    charge.set("solid", "Cathode_Metal")
    charge.set("sweep type", "single")
    charge.set("voltage", 0.0)

    # Doping boxes of the candidate shape
    for name, dtype, conc, x0, x1, y0, y1 in boxes:
        add_doping(charge, name, dtype, conc, x0, x1, y0, y1)

    # Export monitor
    mat_path = os.path.join(DATA_DIR, f"carriers_{tag}_{abs(voltage):.1f}V.mat").replace("\\", "/")
    if os.path.exists(mat_path):
        try:
            os.remove(mat_path)
        except Exception:
            pass

    charge.addchargemonitor()
    charge.set("name", "charge_export")
    try_set(charge, ("monitor type", "geometry type"), "2D Z-normal", required=False)
    charge.set("x", 0.0)
    charge.set("x span", MON_X_SPAN)
    charge.set("y", 0.0)
    charge.set("y span", MON_Y_SPAN)
    charge.set("save data", True)
    charge.set("filename", mat_path)

    save_path = os.path.join(SIM_DIR, f"S_{tag}_{abs(voltage):.1f}V.ldev").replace("\\", "/")
    charge.save(save_path)
    t_start = time.time()
    charge.run()

    # wait for a FRESH export (guards against a stale file from a previous run)
    for _ in range(80):
        if (os.path.exists(mat_path) and os.path.getsize(mat_path) > 1000
                and os.path.getmtime(mat_path) >= t_start - 1.0):
            break
        time.sleep(0.25)
    else:
        raise RuntimeError("CHARGE did not export a fresh .mat file (did it converge?)")
    time.sleep(0.3)
    print(f"     CHARGE done in {time.time() - t0:.0f} s")
    return mat_path


# =============================================================================
# MODE (FDE)
# =============================================================================
def setup_fde_waveguide(mode):
    mode.eval("newproject;")
    mode.eval('mat_temp = addmaterial("Index perturbation"); setmaterial(mat_temp, "name", "Si_np_perturbed");')
    mode.eval('setmaterial("Si_np_perturbed", "base material", "Si (Silicon) - Palik");')
    for prop in ("include np density", "np density", "enable np density", "include np"):
        try:
            mode.setmaterial("Si_np_perturbed", prop, 1)
            break
        except Exception:
            continue
    for model in ("Soref and Bennet", "Soref and Bennett", 1):
        try:
            mode.setmaterial("Si_np_perturbed", "np density model", model)
            break
        except Exception:
            continue

    W2, hc = SLAB_WIDTH / 2.0, CORE_WIDTH / 2.0
    add_rect(mode, "Background_SiO2", "SiO2 (Glass) - Palik",
             -(SLAB_WIDTH + 2e-6) / 2, (SLAB_WIDTH + 2e-6) / 2,
             -MODE_OXIDE_Y_SPAN / 2, MODE_OXIDE_Y_SPAN / 2, 5, True)
    add_rect(mode, "Si_Slab", "Si_np_perturbed", -W2, W2, Y_BOT, Y_SLAB, 3, True)
    add_rect(mode, "Si_Core", "Si_np_perturbed", -hc, hc, Y_BOT, Y_TOP, 2, True)

    mode.addfde()
    try_set(mode, ("solver type", "dimension"), "2D Z normal", required=False)
    mode.set("x", 0.0)
    mode.set("x span", FDE_X_SPAN)
    mode.set("y", 0.0)
    mode.set("y span", FDE_Y_SPAN)
    mode.set("wavelength", LAMBDA)
    mode.set("number of trial modes", N_MODES)
    try_set(mode, "define x mesh by", "maximum mesh step", required=False)
    try_set(mode, "define y mesh by", "maximum mesh step", required=False)
    try_set(mode, "dx", FDE_MESH, required=False)
    try_set(mode, "dy", FDE_MESH, required=False)

    mode.addgridattribute("np density")
    mode.set("name", "np density")
    for prop, val in (("x", 0.0), ("y", 0.0), ("z", 0.0)):
        try_set(mode, prop, val, required=False)


def solve_te_modes(mode):
    """Guided TE-like modes sorted as returned by the solver (index 0 = fundamental)."""
    mode.run()
    mode.findmodes()
    rows = []
    for i in range(1, N_MODES + 1):
        name = f"FDE::data::mode{i}"
        try:
            neff = complex(np.squeeze(mode.getdata(name, "neff")))
            loss = float(np.real(np.squeeze(mode.getdata(name, "loss")))) / 100.0   # dB/cm
            te = float(np.real(np.squeeze(mode.getdata(name, "TE polarization fraction"))))
        except Exception:
            continue
        rows.append({"neff": neff, "loss": loss, "te": te})
    return [r for r in rows if r["te"] > 0.5 and r["neff"].real > N_CLAD]


def run_fde(mode, mat_0v, mat_2v, tag):
    found = {}
    for label, mat_file in (("0V", mat_0v), ("-2V", mat_2v)):
        setup_fde_waveguide(mode)                    # fresh project: no stale np-density data
        mode.select("np density")
        mode.importdataset(mat_file)
        modes = solve_te_modes(mode)
        if not modes:
            raise RuntimeError(f"No guided TE mode at {label}")
        found[label] = modes[0]                      # fundamental TE
        mode.save(os.path.join(SIM_DIR, f"S_{tag}_optical_{label}.lms"))

    a, b = found["0V"], found["-2V"]
    dn = abs(b["neff"].real - a["neff"].real)
    vpilpi = (LAMBDA / (2.0 * dn)) * VOLTAGE_SWING * 100.0 if dn > 1e-12 else float("inf")   # V.cm
    return {"neff0": a["neff"].real, "dneff": dn, "vpilpi": vpilpi,
            "loss0": a["loss"], "loss2": b["loss"], "avg_loss": 0.5 * (a["loss"] + b["loss"])}



def evaluate(charge, mode, g, tag):
    boxes = grid_to_boxes(g)
    mat_0v = run_charge(charge, boxes, 0.0, tag)
    mat_2v = run_charge(charge, boxes, -VOLTAGE_SWING, tag)
    print("  -> MODE (FDE)")
    return run_fde(mode, mat_0v, mat_2v, tag)


# =============================================================================
# Genetic algorithm
# =============================================================================
def random_staircase(rng):
    return from_p_counts(rng.random(NROWS))


def crossover(a, b, rng):
    """Swap whole rows or whole columns -> offspring keep rectilinear structure."""
    c = a.copy()
    if rng.random() < 0.5:
        take = rng.random(NROWS) < 0.5
        c[take] = b[take]
    else:
        take = rng.random(NX) < 0.5
        c[:, take] = b[:, take]
    return c


def mutate(g, rng):
    g = g.copy()
    flip = rng.random(g.shape) < P_MUTATE_BIT
    g[flip] = ~g[flip]
    if rng.random() < P_MUTATE_BLOCK:                   # flip a random rectangle
        r0, c0 = rng.integers(0, NROWS), rng.integers(0, NX)
        r1, c1 = r0 + rng.integers(1, 3), c0 + rng.integers(1, 4)
        g[r0:r1, c0:c1] = ~g[r0:r1, c0:c1]
    return g


def tournament(pop, fit, rng, k=3):
    idx = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    return pop[idx[np.argmin([fit[i] for i in idx])]]


def run_ga(fitness, cache, rng, extra_seeds=()):
    """fitness(genome, label, gen) -> cost.  `cache` (key -> record) prevents repeated simulations."""
    pop, labels = [], []
    for name, g in list(all_seeds().items()) + [(lab, g) for g, lab in extra_seeds]:
        if all(key_of(g) != key_of(q) for q in pop):
            pop.append(g)
            labels.append(name)
    while len(pop) < POP_SIZE:
        pop.append(random_staircase(rng))
        labels.append("")
    pop, labels = pop[:POP_SIZE], labels[:POP_SIZE]

    fit = [fitness(g, lab, 0) for g, lab in zip(pop, labels)]
    print(f"[GA] gen 0   best cost = {min(fit):.3f}")

    for gen in range(1, N_GENERATIONS + 1):
        order = np.argsort(fit)
        new_pop = [pop[i].copy() for i in order[:N_ELITE]]
        while len(new_pop) < POP_SIZE:
            for _ in range(20):                          # try to produce an unseen design
                child = repair(mutate(crossover(tournament(pop, fit, rng), tournament(pop, fit, rng), rng), rng))
                if key_of(child) not in cache:
                    break
            new_pop.append(child)
        pop = new_pop
        fit = [fitness(g, "", gen) for g in pop]
        print(f"[GA] gen {gen}   best cost = {min(fit):.3f}")


# =============================================================================
# Plots
# =============================================================================
def draw_doping(ax, boxes, title):
    """p = red, n = blue; darker = higher concentration (log scale). Black line = p/n interface."""
    xs = np.linspace(-0.4e-6, 0.4e-6, 640)
    ys = np.linspace(Y_BOT - 30e-9, Y_TOP + 30e-9, 270)
    X, Y = np.meshgrid(xs, ys)
    net = np.zeros_like(X)
    for _, dtype, conc, x0, x1, y0, y1 in boxes:
        m = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
        net[m] += conc if dtype == "p" else -conc
    si = ((np.abs(X) <= HC) & (Y >= Y_BOT) & (Y <= Y_TOP)) | ((Y >= Y_BOT) & (Y <= Y_SLAB))
    net[~si] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        p_map = np.ma.masked_where(~(net > 0), np.log10(np.abs(net)))
        n_map = np.ma.masked_where(~(net < 0), np.log10(np.abs(net)))
    ax.pcolormesh(xs * 1e9, ys * 1e9, p_map, cmap="Reds", vmin=15.5, vmax=19.0, shading="auto")
    ax.pcolormesh(xs * 1e9, ys * 1e9, n_map, cmap="Blues", vmin=15.5, vmax=19.0, shading="auto")
    ax.contour(xs * 1e9, ys * 1e9, np.ma.masked_invalid(np.sign(net)), levels=[0], colors="k", linewidths=1.8)
    outline = [(-400, Y_BOT * 1e9), (-400, Y_SLAB * 1e9), (-HC * 1e9, Y_SLAB * 1e9), (-HC * 1e9, Y_TOP * 1e9),
               (HC * 1e9, Y_TOP * 1e9), (HC * 1e9, Y_SLAB * 1e9), (400, Y_SLAB * 1e9), (400, Y_BOT * 1e9),
               (-400, Y_BOT * 1e9)]
    ax.plot(*zip(*outline), color="0.35", lw=0.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    return xs, ys, net


def plot_best_shape(g, title, png_path, npz_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs, ys, net = draw_doping(ax, grid_to_boxes(g), title)
    peak = "peak " if DOPING_MODEL == "pearson4" else ""
    handles = [plt.Rectangle((0, 0), 1, 1, color="#3b78d8", label=f"p ({peak}{P_DOPING:.1e} cm$^{{-3}}$)"),
               plt.Rectangle((0, 0), 1, 1, color="#e05a47", label=f"n ({peak}{N_DOPING:.1e} cm$^{{-3}}$)")]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    np.savez(npz_path, x=xs, y=ys, net_doping_cm3=net, grid_p_is_true=g)   # + = p-type, - = n-type


def plot_seed_shapes(path):
    """Grid layout (cell indices, ridge / slab rows) + the seed shapes drawn on it."""
    seeds = all_seeds()
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    xe, ye = X_EDGES * 1e9, Y_EDGES * 1e9
    cmap = plt.matplotlib.colors.ListedColormap(["#065ce6", "#e62208"])        # n = red, p = blue
    panels = [("Grid layout (row, column)", None)] + list(seeds.items())
    for ax, (name, g) in zip(axs.flat, panels):
        if g is None:
            ax.pcolormesh(xe, ye, np.ones((NROWS, NX)), cmap="Greys", vmin=0, vmax=3, edgecolors="w", linewidth=1)
            for r in range(NROWS):
                for c in range(NX):
                    ax.text(0.5 * (xe[c] + xe[c + 1]), 0.5 * (ROW_Y[r][0] + ROW_Y[r][1]) * 1e9,
                            f"{r},{c}", ha="center", va="center", fontsize=6)
            ax.text(-HC * 1e9 - 12, 0.5 * (Y_SLAB + Y_TOP) * 1e9, "ridge\nrows", ha="right", va="center", fontsize=7)
            ax.text(-HC * 1e9 - 12, 0.5 * (Y_BOT + Y_SLAB) * 1e9, "slab\nrows", ha="right", va="center", fontsize=7)
        else:
            ax.pcolormesh(xe, ye, np.flipud(g.astype(float)), cmap=cmap, vmin=0, vmax=1, edgecolors="w", linewidth=1)
        ax.axhline(Y_SLAB * 1e9, color="k", ls="--", lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x [nm]")
        ax.set_ylabel("y [nm]")
    for ax in list(axs.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{NX} x {NROWS} cells (50 nm wide; red = p, blue = n; left/right slab outside the core is p / n)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)


def plot_implant_profiles(path):
    d = np.linspace(0, CORE_HEIGHT, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d * 1e9, PROFILES["p"](d), color="#3b78d8", lw=2, label="p implant (Pearson IV)")
    ax.plot(d * 1e9, PROFILES["n"](d), color="#e05a47", lw=2, label="n implant (Pearson IV)")
    ax.axhline(P_DOPING, color="#3b78d8", ls=":", lw=1, label="p peak / old constant")
    ax.axhline(N_DOPING, color="#e05a47", ls=":", lw=1, label="n peak / old constant")
    ax.axvline(SLAB_HEIGHT * 1e9, color="k", ls="--", lw=0.8)
    ax.text(SLAB_HEIGHT * 1e9 + 3, 0.02 * P_DOPING, "slab bottom", fontsize=7)
    ax.set_xlabel("depth below the top silicon surface [nm]")
    ax.set_ylabel("concentration [cm$^{-3}$]")
    ax.set_title("Implant doping profiles used by CHARGE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def plot_gallery(records, path, n=6):
    ok = sorted([r for r in records if r["ok"]], key=lambda r: r["cost"])[:n]
    if not ok:
        return
    cols = 2
    fig, axs = plt.subplots((len(ok) + 1) // cols, cols, figsize=(11, 2.9 * ((len(ok) + 1) // cols)), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, r in zip(axs.flat, ok):
        ax.axis("on")
        lab = f" [{r['label']}]" if r["label"] else ""
        draw_doping(ax, grid_to_boxes(np.array(r["grid"], dtype=bool)),
                    f"VpiLpi={r['vpilpi']:.2f} V.cm  loss={r['avg_loss']:.1f} dB/cm  cost={r['cost']:.1f}{lab}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)


def plot_scatter(records, best, path):
    ok = [r for r in records if r["ok"]]
    if not ok:
        print("No successful simulation to plot.")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    sc = a1.scatter([r["avg_loss"] for r in ok], [r["vpilpi"] for r in ok], c=[r["gen"] for r in ok],
                    cmap="viridis", s=55, edgecolor="k", linewidth=0.4)
    for r in ok:
        if r["label"]:
            a1.annotate(r["label"], (r["avg_loss"], r["vpilpi"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    a1.scatter(best["avg_loss"], best["vpilpi"], marker="*", s=380, c="red", edgecolor="k", label="best", zorder=5)
    a1.set_xlabel("Average loss [dB/cm]")
    a1.set_ylabel("Vpi*Lpi [V.cm]")
    a1.set_title("All evaluated shapes (color = generation)")
    a1.grid(alpha=0.3)
    a1.legend()
    fig.colorbar(sc, ax=a1, label="generation")

    costs = [r["cost"] for r in records]
    a2.plot(np.arange(1, len(costs) + 1), np.minimum.accumulate(costs), "r-", lw=2, label="best so far")
    a2.scatter(np.arange(1, len(costs) + 1), costs, s=18, c="gray", label="each simulation")
    a2.set_yscale("log")
    a2.set_xlabel("Simulation #")
    a2.set_ylabel("cost = Vpi*Lpi x loss [V.dB]")
    a2.set_title("Convergence")
    a2.grid(alpha=0.3, which="both")
    a2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)


# =============================================================================
if __name__ == "__main__":
    lum = get_lumapi()
    rng = np.random.default_rng(SEED)
    charge_app = mode_app = None
    records, cache = [], {}

    # never overwrite an earlier run: keep its history.json under a time-stamped name
    old_hist = os.path.join(RES_DIR, f"history_nx{NX}.json")
    if os.path.exists(old_hist):
        backup = os.path.join(PROJECT_DIR, f"history_backup_{time.strftime('%Y%m%d_%H%M%S')}.json")
        os.replace(old_hist, backup)
        print(f"Previous history saved as {backup}")

    print("Launching Lumerical CHARGE and MODE...")
    try:
        charge_app = lum.DEVICE(hide=not SHOW_GUI)
        mode_app = lum.MODE(hide=not SHOW_GUI)

        def fitness(g, label, gen):
            k = key_of(g)
            if k in cache:
                return cache[k]["cost"]
            print(f"\n=== Simulation {len(records) + 1} (gen {gen}) {label} ===")
            print("\n".join("     " + "".join("p" if v else "n" for v in row) for row in g))
            rec = {"grid": g.astype(int).tolist(), "label": label, "gen": gen, "doping": DOPING_MODEL}
            try:
                r = evaluate(charge_app, mode_app, g, "eval")
                cost = r["vpilpi"] * abs(r["avg_loss"])
                ok = bool(np.isfinite(cost))
                rec.update(r, ok=ok, cost=float(cost) if ok else PENALTY)
                print(f"  RESULT  Vpi*Lpi = {r['vpilpi']:.3f} V.cm   loss = {r['avg_loss']:.3f} dB/cm   "
                      f"cost = {rec['cost']:.3f} V.dB")
            except Exception as e:
                print(f"  FAILED: {e}")
                rec.update(ok=False, cost=PENALTY)
            cache[k] = rec
            records.append(rec)
            with open(os.path.join(PROJECT_DIR, "history.json"), "w") as f:
                json.dump(records, f, indent=2)
            return rec["cost"]

        if DOPING_MODEL == "pearson4" and USE_LUMERICAL_MOMENTS:
            print("Implant moments from Lumerical:")
            apply_lumerical_moments(charge_app)
        plot_seed_shapes(os.path.join(RES_DIR, "grid_and_seed_shapes.png"))
        if DOPING_MODEL == "pearson4":
            plot_implant_profiles(os.path.join(PROJECT_DIR, "implant_profiles.png"))
        old_seeds = load_old_seeds(OLD_HISTORY_FILES, N_OLD_SEEDS) if OLD_HISTORY_FILES else []

        run_ga(fitness, cache, rng, old_seeds)

        good = [r for r in records if r["ok"]]
        if not good:
            raise RuntimeError("No simulation succeeded.")
        best_rec = min(good, key=lambda r: r["cost"])
        best_g = np.array(best_rec["grid"], dtype=bool)
        print("\n##### BEST SHAPE (p = left/p-type, n = right/n-type) #####")
        print("\n".join("     " + "".join("p" if v else "n" for v in row) for row in best_g))
        print(f"cost = {best_rec['cost']:.3f} V.dB  {best_rec['label']}")

        # ---- final CHARGE + FDE run of the best shape (files are kept) -------------------
        print("\n=== Final CHARGE + FDE simulation of the best shape ===")
        final = evaluate(charge_app, mode_app, best_g, "best")
        final.update(cost=final["vpilpi"] * abs(final["avg_loss"]), label=best_rec["label"])
        print(f"  Vpi*Lpi = {final['vpilpi']:.3f} V.cm | loss(0V) = {final['loss0']:.3f}, "
              f"loss(-2V) = {final['loss2']:.3f} dB/cm | dneff = {final['dneff']:.3e}")

        # ---- save everything -----------------------------------------------------------
        with open(os.path.join(PROJECT_DIR, "best_shape.json"), "w") as f:
            json.dump({"doping_model": DOPING_MODEL, "p_peak_cm3": P_DOPING, "n_peak_cm3": N_DOPING,
                       "p_implants": P_IMPLANTS, "n_implants": N_IMPLANTS,
                       "grid_rows_top_to_bottom_p_is_1": best_g.astype(int).tolist(),
                       "x_edges_nm": (X_EDGES * 1e9).tolist(), "y_edges_nm": (Y_EDGES * 1e9).tolist(),
                       "metrics": final, "n_simulations": len(records)}, f, indent=2)
        plot_best_shape(best_g, f"Best evolved junction   cost = {final['cost']:.2f} V.dB",
                        os.path.join(PROJECT_DIR, "best_shape_doping.png"),
                        os.path.join(PROJECT_DIR, "best_shape_doping.npz"))
        plot_gallery(records, os.path.join(PROJECT_DIR, "top_shapes.png"))
        plot_scatter(records, final, os.path.join(PROJECT_DIR, "scatter_and_convergence.png"))
        print(f"\nSaved results, plots and Lumerical projects in: {PROJECT_DIR}")

        plt.show(block=False)
        plt.pause(0.5)
        if SHOW_GUI:
            input("--> Lumerical is still open. Press Enter to close it and exit...")
    finally:
        for app in (charge_app, mode_app):
            if app is not None:
                try:
                    app.close()
                except Exception:
                    pass