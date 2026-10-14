import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Automatically find the absolute path based on where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "results", "metrics", "history_bandwidth_run.json")
OUTPUT_PNG = os.path.join(BASE_DIR, "results", "plots", "final_scatter_plot.png")

# Load the data
if not os.path.exists(HISTORY_FILE):
    raise FileNotFoundError(f"Could not find the file at the absolute path:\n{HISTORY_FILE}\nCheck if the 'results' folder is in the same directory as this script.")

with open(HISTORY_FILE) as f:
    records = json.load(f)

# Filter for successful runs under the penalty threshold
ok = [r for r in records if r.get("ok", False) and r.get("cost", 1e6) < 1e6]

if not ok:
    print("No successful records found under the penalty threshold.")
else:
    # Find the current best
    best = min(ok, key=lambda r: r["cost"])
    has_bw = all("f3db_ghz" in r for r in ok)

    # Recreate the 3-panel plot
    fig, axs = plt.subplots(1, 3 if has_bw else 2, figsize=(17 if has_bw else 12, 5))
    a1, a_last = axs[0], axs[-1]
    
    sc = a1.scatter([r["avg_loss"] for r in ok], [r["vpilpi"] for r in ok], 
                    c=[r.get("gen", 0) for r in ok], cmap="viridis", s=55, edgecolor="k", linewidth=0.4)
    a1.scatter(best["avg_loss"], best["vpilpi"], marker="*", s=380, c="red", edgecolor="k", label="best", zorder=5)
    a1.set_xlabel("Average loss [dB/cm]")
    a1.set_ylabel("Vpi*Lpi [V.cm]")
    a1.set_title("Efficiency vs loss (color = generation)")
    a1.grid(alpha=0.3)
    a1.legend()
    fig.colorbar(sc, ax=a1, label="generation")

    if has_bw:
        a2 = axs[1]
        sc2 = a2.scatter([r["f3db_ghz"] for r in ok], [r["vpilpi"] for r in ok], 
                         c=[r["avg_loss"] for r in ok], cmap="plasma", s=55, edgecolor="k", linewidth=0.4)
        a2.scatter(best["f3db_ghz"], best["vpilpi"], marker="*", s=380, c="red", edgecolor="k", zorder=5)
        a2.set_xlabel("RC bandwidth f3dB [GHz]")
        a2.set_ylabel("Vpi*Lpi [V.cm]")
        a2.set_title("Efficiency vs speed (color = loss)")
        a2.grid(alpha=0.3)
        fig.colorbar(sc2, ax=a2, label="loss [dB/cm]")

    # Convergence plot
    costs = [r["cost"] for r in records if r.get("cost", 1e6) < 1e6]
    if costs:
        a_last.plot(np.arange(1, len(costs) + 1), np.minimum.accumulate(costs), "r-", lw=2, label="best so far")
        a_last.scatter(np.arange(1, len(costs) + 1), costs, s=18, c="gray", label="each simulation")
        a_last.set_yscale("log")
        a_last.set_xlabel("Valid Simulation #")
        a_last.set_ylabel("cost")
        a_last.set_title("Convergence")
        a_last.grid(alpha=0.3, which="both")
        a_last.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"Plot saved successfully to {OUTPUT_PNG}")
    plt.show()