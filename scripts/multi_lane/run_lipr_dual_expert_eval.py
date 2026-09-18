"""Evaluate one paired CE/SE model configuration without merging control."""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from functions import data_recording as dr
from functions import formation_controller as fc
from functions import hpc_utils
from functions import print_control as prc
from functions import vehicle_generation3 as vg


def set_global_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def summarise_rewards(handler):
    rewards = [
        values[1]
        for values in handler.dic_score_reward.values()
        if len(values) >= 2
    ]
    count = len(rewards)
    total = float(sum(rewards))
    return count, total, total / count if count else 0.0


def run_evaluation(args):
    ce_model = Path(args.ce_model_path).resolve()
    se_model = Path(args.se_model_path).resolve()
    if not ce_model.is_file():
        raise FileNotFoundError(f"CE model not found: {ce_model}")
    if not se_model.is_file():
        raise FileNotFoundError(f"SE model not found: {se_model}")

    set_global_seed(args.seed)
    root = Path(__file__).resolve().parents[2]
    sumo_config = (
        root / "road_network" / "multi_lane_motorway" / "real"
        / "cfg_multi_lane_merge.sumocfg"
    )
    sumo_binary = "sumo-gui" if args.gui else "sumo"
    sumo_command = [
        sumo_binary, "-c", str(sumo_config),
        "--step-length", "0.1",
        "--seed", str(args.seed),
        "--no-warnings",
    ]

    lane_0_schedule = vg.generate_entry_arrivals_shifted_exp(
        args.st, args.av_p, args.m_fr, args.seed
    )
    lane_1_schedule = vg.generate_entry_arrivals_shifted_exp(
        args.st, args.av_p, args.m_fr, 100 - args.seed
    )

    if args.gui:
        import traci
    else:
        import libsumo as traci

    formation = None
    traci.start(sumo_command)
    started_at = time.time()
    try:
        recorder = dr.DataRecording(traci)
        recorder.max_platoon_size = args.max_team_size
        recorder.get_avhid_ptype(r_dpt_type={})
        generator = vg.VehGen(traci, args.seed)

        formation = fc.FormationController(
            recorder,
            traci,
            sa_mode="predict",
            ca_mode="predict",
            tsg_mode="off",
            exp_name=f"PAIR{args.pair_id}_S{args.seed}",
            expert_hidden_dims=args.hidden_layer,
            max_team_size=args.max_team_size,
            fc_mode="full",
            se_model_path=str(se_model),
            ce_model_path=str(ce_model),
        )

        step = 0
        final_state = ({}, {}, {})
        while step < args.st * 10:
            traci.simulationStep()
            generator.veh_gen_homo(
                step, lane_1_schedule, "m", "route_m", 27.5, "1"
            )
            generator.veh_gen_homo(
                step, lane_0_schedule, "m", "route_m", 27.5, "0"
            )
            final_state = formation.step(args.st, step, lc=False)
            recorder.record_tail_arrival(step)
            step += 1

        # Finish all generated trips so the final SPR excludes unfinished traffic.
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            final_state = formation.step(args.st, step, lc=False)
            recorder.record_tail_arrival(step)
            step += 1

        follower_state, platoon_size_history, _ = final_state
        formation_result = hpc_utils.get_fc_detail(
            follower_state,
            platoon_size_history,
            max_size=args.max_team_size,
        )
        se_count, se_sum, se_avg = summarise_rewards(formation.split_agent)
        ce_count, ce_sum, ce_avg = summarise_rewards(formation.collect_agent)

        row = {
            "pair_id": args.pair_id,
            "learning_rate": args.lr,
            "train_interval": args.train_interval,
            "seed": args.seed,
            "hv_type": "homogeneous",
            "av_p": args.av_p,
            "mainline_demand_per_lane": args.m_fr,
            "simulation_time": args.st,
            "ce_model_path": str(ce_model),
            "se_model_path": str(se_model),
            "SPR": formation_result["spr"],
            "CFR": formation_result["cfr"],
            "over_pltn": formation_result["over_pltn"],
            "std_pltn": formation_result["std_pltn"],
            "sparse_pltn": formation_result["sparse_pltn"],
            "avg_pltn_size": formation_result["avg_pltn_size"],
            "se_count": se_count,
            "se_reward_sum": se_sum,
            "se_reward_avg": se_avg,
            "ce_count": ce_count,
            "ce_reward_sum": ce_sum,
            "ce_reward_avg": ce_avg,
            "runtime": time.time() - started_at,
        }
        hpc_utils.write_one_row_csv(args.out_csv, row)
        print(f"[LIPR Eval] Saved: {args.out_csv}")
        return row
    finally:
        if formation is not None:
            for name in ("split_agent", "collect_agent"):
                handler = getattr(formation, name, None)
                writer = getattr(getattr(handler, "agent", None), "writer", None)
                if writer is not None:
                    writer.close()
        traci.close()


def main(args=None, root=None):
    del root
    prc.PRINT_ENABLED = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce_model_path", required=True)
    parser.add_argument("--se_model_path", required=True)
    parser.add_argument("--pair_id", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--train_interval", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--av_p", type=float, default=0.1)
    parser.add_argument("--m_fr", type=float, default=1000)
    parser.add_argument("--st", type=int, default=1500)
    parser.add_argument("--max_team_size", type=int, default=12)
    parser.add_argument("--hidden_layer", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--gui", action="store_true")
    parsed = parser.parse_args(args=args)
    run_evaluation(parsed)


if __name__ == "__main__":
    main()
