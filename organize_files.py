import os
import shutil

# Target directory (current folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder rules based on extension and filename
DESTINATIONS = {
    "simulations/charge": [".ldev", ".log", ".data"],
    "simulations/mode": [".lms"],
    "data/carriers": [".mat", ".npz"],
    "results/plots": [".png"],
    "results/metrics": [".json"],
}

for folder in DESTINATIONS.keys():
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

for filename in os.listdir(BASE_DIR):
    filepath = os.path.join(BASE_DIR, filename)

    # Skip directories and our python scripts
    if os.path.isdir(filepath) or filename.endswith(".py"):
        continue

    # Delete junk/lock files created during crashes
    if filename.startswith("~#") or filename.endswith(".log"):
        try:
            os.remove(filepath)
            print(f"Deleted junk file: {filename}")
        except Exception:
            pass
        continue

    # Move files to their categorized homes
    moved = False
    for folder, extensions in DESTINATIONS.items():
        if any(filename.endswith(ext) for ext in extensions):
            dest_path = os.path.join(BASE_DIR, folder, filename)
            shutil.move(filepath, dest_path)
            moved = True
            break

print("All files have been cleanly sorted!")