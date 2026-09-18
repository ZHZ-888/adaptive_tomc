# run_one_RFdata.py
"""
Run one simulation for RL training dataset generation.

This script is designed to call the scripts simulation function
(run_mpgc_multi_lane.py) with specific parameters
(av_p, r_fr, seed, st), and save the resulting feature dataset
with a parameter-specific file name.
"""

import argparse
import pandas as pd
import functions.print_control as prc
from scripts.multi_lane.run_mpgc_multi_lane_collect_RF_state_data import main


if __name__ == "__main__":
    # ---------------- Argument Parser ----------------
    parser = argparse.ArgumentParser(description="Run one RL simulation instance")

    parser.add_argument("--av_p", type=float, required=True,
                        help="AV penetration rate (e.g., 0.1, 0.2, 0.3)")
    parser.add_argument("--r_fr", type=int, required=True,
                        help="Ramp flow rate (veh/h), e.g., 810")
    parser.add_argument("--m_fr", type=int, default=1080,
                        help="Mainline flow rate (veh/h), default = 1080")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed")
    parser.add_argument("--st", type=int, default=6000,
                        help="Simulation duration in seconds, default = 6000")
    parser.add_argument("--nogui", action="store_true",
                        help="Disable SUMO GUI (used for HPC batch runs)")

    args = parser.parse_args()

    # ---------------- Settings ----------------
    prc.PRINT_ENABLED = False  # disable printing during batch runs

    gui = not args.nogui       # GUI only if user does not specify --nogui
    gui = False                # force GUI off for data generation
    plot = False
    display = False
    lc_hv = False
    lc_av = False
    loss_rate = 0

    print(f"\n=== Running one simulation ===")
    print(f"AV penetration: {args.av_p}")
    print(f"Ramp flow:      {args.r_fr}")
    print(f"Main flow:      {args.m_fr}")
    print(f"Seed:           {args.seed}")
    print(f"Duration:       {args.st}\n")

    # ---------------- Run Simulation ----------------
    dic_follower_state, his_dic_platoon_size, dic_id_features = main(
        av_p=args.av_p,
        r_fr=args.r_fr,
        m_fr=args.m_fr,
        seed=args.seed,
        loss_rate=loss_rate,
        gui=gui,
        plot=plot,
        display=display,
        lc_hv=lc_hv,
        lc_av=lc_av,
        st=args.st
    )

    # ---------------- Construct Feature Table ----------------
    feature_target = [
        features + dic_follower_state.get(key, ['unknown'])
        for key, features in dic_id_features.items()
    ]

    columns = [
        'id', 'dis_to_pv', 'v_pv', 'dis_leaderAV',
        'pos_this', 'size', 'leaderAV_id_start',
        'state', 'leaderAV_id_end'
    ]

    df = pd.DataFrame(feature_target, columns=columns)

    # ---------------- Output File Path ----------------
    out_path = (
        f"data/features/"
        f"df_RLdata_av{args.av_p}_r{args.r_fr}_seed{args.seed}_st{args.st}.csv"
    )

    df.to_csv(out_path, index=False)

    print(f"Saved dataset to: {out_path}\n")
    print("=== Simulation completed ===\n")
