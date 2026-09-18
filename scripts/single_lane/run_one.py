import sys
import itertools
import pandas as pd
import importlib
from functions import print_control as prc

# Usage:
# python run_one.py <algo_module> <task_id>
# Example:
#   python run_one.py fifo_250810 0
#   python run_one.py rm_250813 5
#   python run_one.py mpgc_250810 12

if len(sys.argv) != 3:
    print("Usage: python run_one.py <algo_module> <task_id>")
    sys.exit(1)

algo_module = sys.argv[1]  # Algorithm module name (without .py)
task_id = int(sys.argv[2])  # Task index

# Define all parameter combinations
ls_av_p = [0]  # AV penetration rates
ls_seed = list(range(1, 11))  # Random seeds
ls_r_fr = [180, 270, 360, 450, 540, 630, 720, 810, 900, 990]  # Ramp flow rates (veh/h)
all_combos = list(itertools.product(ls_av_p, ls_seed, ls_r_fr))

# Check task_id validity
if task_id < 0 or task_id >= len(all_combos):
    print(f"task_id {task_id} out of range")
    sys.exit(1)

# Get parameters for this task
av_p, seed, r_fr = all_combos[task_id]

# Default parameters
prc.PRINT_ENABLED = False  # Disable console output
st = 1200  # Simulation time in seconds
m_fr = 1080  # Mainline flow rate (veh/h)
gui = False  # SUMO GUI
plot = False  # Disable plotting

# Dynamically import the chosen algorithm module
exp = importlib.import_module(algo_module)

# Run the simulation
dic_id_speed, _, _, tp = exp.main(av_p, r_fr, m_fr, seed, gui, plot, st)

# Calculate average speed
all_speeds = sum(dic_id_speed.values(), [])
average_v = sum(all_speeds) / len(all_speeds) if all_speeds else 0

# Save results to CSV
df = pd.DataFrame([[m_fr, r_fr, av_p, seed, average_v, tp]],
                  columns=["m_fr", "r_fr", "av_p", "seed", "average_v", "throughput"])
df.to_csv(f"data/result_{algo_module}_{r_fr}_{av_p}_{seed}_{task_id}.csv", index=False)

# Print task summary
print(f"{algo_module} Task {task_id} done: m_fr={m_fr}, r_fr={r_fr}, av_p={av_p}, seed={seed}, "
      f"average_v={average_v}, tp={tp}")
