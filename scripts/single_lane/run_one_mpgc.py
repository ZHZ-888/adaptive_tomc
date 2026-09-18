import os
import sys
import time
import itertools
import pandas as pd
import importlib
from functions import print_control as prc
from functions import calc_ttc_sd_exposure


# Usage:
# python run_one_mpgc.py <algo_module> <exp_id> <task_id>
# Example:
#   python run_one_mpgc.py mpgc_250810 3 12
#
# Backward compatible:
#   python run_one_mpgc.py <algo_module> <task_id>   # default exp_id=1

# ---------- Argument parsing ----------
if len(sys.argv) == 3:
    # Old usage: <algo_module> <task_id>, assume exp_id=1
    algo_module = sys.argv[1]
    exp_id = 1
    task_id = int(sys.argv[2])
elif len(sys.argv) == 4:
    # New usage: <algo_module> <exp_id> <task_id>
    algo_module = sys.argv[1]
    exp_id = int(sys.argv[2])
    task_id = int(sys.argv[3])
else:
    print("Usage:\n"
          "  python run_one_mpgc.py <algo_module> <exp_id> <task_id>\n"
          "  python run_one_mpgc.py <algo_module> <task_id>   # (default exp_id=1)")
    sys.exit(1)

# ---------- Experiment parameter grids ----------
ls_r_fr = [180, 270, 360, 450, 540, 630, 720, 810, 900, 990]  # Ramp flow rates (veh/h)
ls_seed = list(range(1, 11))  # Random seeds
ls_interval_sec = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # mpc interval (seconds)
ls_interval_step = [int(v * 10) for v in ls_interval_sec]

def exp_grid(exp_id):
    """Return parameter grids for a given experiment ID."""
    if exp_id == 1:  # ideal
        return (ls_r_fr, [0.1, 0.2, 0.3], ls_seed, [1], [1], [0], [70])
    elif exp_id == 2:  # acc_hv
        return (ls_r_fr, [0.1], ls_seed, [0, 0.5], [1], [0], [70])
    elif exp_id == 3:  # platoon_hv
        return (ls_r_fr, [0.1], ls_seed, [1], [0.4, 0.6, 0.8], [0], [70])
    elif exp_id == 4:  # v2x_disturbance
        return (ls_r_fr, [0.1], ls_seed, [1], [1], [0.05, 0.1, 0.15], [70])
    elif exp_id == 5: # for find the best mpc_interval
        return ([720, 810, 900, 990], [0.1], ls_seed, [1], [1], [0], ls_interval_step)
    else:
        raise ValueError(f"Unknown exp_id={exp_id}. Use 1|2|3|4|5.")

# Select grids for this experiment
ls_r_fr, ls_av_p, ls_seed, ls_p_autoFollow, ls_platoon_p, ls_loss_rate, ls_interval = exp_grid(exp_id)

# Generate all parameter combinations
all_combos = list(itertools.product(ls_r_fr, ls_av_p, ls_seed,
                                    ls_p_autoFollow, ls_platoon_p, ls_loss_rate,
                                    ls_interval))

# Check task_id validity
if task_id < 0 or task_id >= len(all_combos):
    print(f"task_id {task_id} out of range for exp {exp_id} (0..{len(all_combos)-1})")
    sys.exit(1)

# Get parameters for this task
r_fr, av_p, seed,  p_autoFollow, platoon_p, loss_rate, mpc_interval = all_combos[task_id]

# Default parameters
prc.PRINT_ENABLED = False  # Disable console output
st = 1200  # Simulation time in seconds
m_fr = 1080  # Mainline flow rate (veh/h)
gui = False  # SUMO GUI
plot = False  # Disable plotting

# Dynamically import the chosen algorithm module
exp = importlib.import_module(algo_module)

# Run the simulation
delta_t = 12
display = False
start = time.time()
dic_id_speed, _, _, tp, _, _, xml_path = exp.main(av_p, r_fr, m_fr, seed, mpc_interval, delta_t,
                                            p_autoFollow, platoon_p, loss_rate,
                                            gui, plot, display, st)
end = time.time()
# Get execution_time
exe_time = end - start
# Calculate ttc_ratio and avg_speed_std
ttc_ratio, avg_speed_std = calc_ttc_sd_exposure.calc_ttc_and_speed_std(xml_path)
# Calculate average speed
all_speeds = sum(dic_id_speed.values(), [])
average_v = sum(all_speeds) / len(all_speeds) if all_speeds else 0

# Save results to CSV
# all line in one csv.  file
save_dir = "../../data"
os.makedirs(save_dir, exist_ok=True)
result_path = f"{save_dir}/all_results_{algo_module}_exp{exp_id}.csv"
df = pd.DataFrame([[m_fr, r_fr, av_p, seed, mpc_interval, p_autoFollow, platoon_p,
                    loss_rate, average_v, tp, ttc_ratio, avg_speed_std, exe_time]],
                  columns=["m_fr", "r_fr", "av_p", "seed", "mpc_interval", "p_autoFollow", "platoon_p", "loss_rate",
                           "average_v", "throughput", "ttc_ratio", "avg_speed_std", "execution_time"])
df.to_csv(result_path, mode="a", header=not os.path.exists(result_path), index=False)

# Print task summary
print(f"{algo_module} Task {task_id} done: m_fr={m_fr}, r_fr={r_fr}, av_p={av_p}, seed={seed}, "
      f"mpc_interval={mpc_interval}, p_autoFollow={p_autoFollow}, platoon_p={platoon_p}, loss_rate={loss_rate}, "
      f"average_v={average_v}, tp={tp}, ttc_ratio={ttc_ratio}, avg_speed_std={avg_speed_std}, "
      f"execution_time={exe_time}")
