"""Evaluate one LIPR Dispatch model in audit mode."""

import argparse
import csv
import os
import random
import time
from collections import defaultdict, deque
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


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _decision_key(row):
    return (
        row["decision_step"],
        row["category"].lower(),
        row["cand_av_id"],
        row["target_platoon_leader_id"],
    )


def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def match_audit_records(decision_csv, reward_csv):
    decisions = _read_csv(decision_csv)
    rewards = _read_csv(reward_csv)
    decision_queues = defaultdict(deque)

    for decision in decisions:
        decision_queues[_decision_key(decision)].append(decision)

    duplicate_decision_keys = sum(
        max(0, len(queue) - 1)
        for queue in decision_queues.values()
    )
    matched = []
    unmatched_rewards = 0

    for reward_row in rewards:
        queue = decision_queues.get(_decision_key(reward_row))
        if not queue:
            unmatched_rewards += 1
            continue

        decision = queue.popleft()
        reward = float(reward_row["final_reward"])
        tsg_execute = _as_bool(decision["tsg_execute"])
        matched.append({
            "category": reward_row["category"].lower(),
            "reward": reward,
            "tsg_execute": tsg_execute,
            "gated_reward": reward if tsg_execute else 0.0,
        })

    unmatched_decisions = sum(len(queue) for queue in decision_queues.values())
    return {
        "decision_count": len(decisions),
        "reward_count": len(rewards),
        "matched": matched,
        "matched_count": len(matched),
        "unmatched_decision_count": unmatched_decisions,
        "unmatched_reward_count": unmatched_rewards,
        "duplicate_decision_key_count": duplicate_decision_keys,
    }


def summarise_category(records, category=None):
    if category is not None:
        records = [row for row in records if row["category"] == category]

    count = len(records)
    rewards = [row["reward"] for row in records]
    executed = [row for row in records if row["tsg_execute"]]
    positive = [row for row in records if row["reward"] > 0]
    nonpositive = [row for row in records if row["reward"] <= 0]

    reward_sum = float(sum(rewards))
    gated_reward_sum = float(sum(row["gated_reward"] for row in records))
    accepted_reward_sum = float(sum(row["reward"] for row in executed))
    positive_executed = sum(row["tsg_execute"] for row in positive)
    nonpositive_rejected = sum(not row["tsg_execute"] for row in nonpositive)
    correct = sum(
        row["tsg_execute"] == (row["reward"] > 0)
        for row in records
    )

    return {
        "count": count,
        "reward_sum": reward_sum,
        "reward_avg": reward_sum / count if count else 0.0,
        "gated_reward_sum": gated_reward_sum,
        "gated_reward_avg": gated_reward_sum / count if count else 0.0,
        "execution_count": len(executed),
        "execution_rate": len(executed) / count if count else 0.0,
        "accepted_reward_avg": (
            accepted_reward_sum / len(executed) if executed else 0.0
        ),
        "positive_count": len(positive),
        "positive_retention_rate": (
            positive_executed / len(positive) if positive else 0.0
        ),
        "nonpositive_count": len(nonpositive),
        "failure_rejection_rate": (
            nonpositive_rejected / len(nonpositive) if nonpositive else 0.0
        ),
        "decision_accuracy": correct / count if count else 0.0,
    }


def _add_prefixed(row, prefix, values):
    row.update({f"{prefix}_{key}": value for key, value in values.items()})


def run_evaluation(args):
    dispatch_model = Path(args.dispatch_model_path).resolve()
    if not dispatch_model.is_file():
        raise FileNotFoundError(
            f"Dispatch model not found: {dispatch_model}"
        )

    set_global_seed(args.seed)
    root = Path(__file__).resolve().parents[2]
    run_dir = Path(os.environ.get(
        "RUN_DIR",
        root / "rl_model" / "rl_logs" / "LIPR_dispatch_eval",
    ))
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RUN_DIR"] = str(run_dir)

    decision_csv = run_dir / "task_self_gate_audit_decision_log.csv"
    reward_csv = run_dir / "task_self_gate_audit_reward_log.csv"
    sumo_config = (
        root / "road_network" / "multi_lane_motorway" / "real"
        / "cfg_multi_lane_merge.sumocfg"
    )
    sumo_binary = "sumo-gui" if args.gui else "sumo"
    sumo_command = [
        sumo_binary,
        "-c",
        str(sumo_config),
        "--step-length",
        "0.1",
        "--seed",
        str(args.seed),
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
            tsg_mode="audit",
            exp_name=f"DISPATCH{args.model_id}_S{args.seed}",
            gate_hidden_dims=(16, 16),
            max_team_size=args.max_team_size,
            fc_mode="full",
            tsg_model_path=str(dispatch_model),
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

        # Complete generated trips so late insertion outcomes receive feedback.
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
        audit = match_audit_records(decision_csv, reward_csv)
        overall = summarise_category(audit["matched"])
        ce = summarise_category(audit["matched"], "ce")
        se = summarise_category(audit["matched"], "se")
        available_task_scores = [
            values["gated_reward_avg"]
            for values in (ce, se)
            if values["count"] > 0
        ]

        row = {
            "model_id": args.model_id,
            "learning_rate": args.lr,
            "train_interval": args.train_interval,
            "seed": args.seed,
            "av_p": args.av_p,
            "mainline_demand_per_lane": args.m_fr,
            "simulation_time": args.st,
            "dispatch_model_path": str(dispatch_model),
            "primary_macro_gated_reward": (
                sum(available_task_scores) / len(available_task_scores)
                if available_task_scores else 0.0
            ),
            "SPR": formation_result["spr"],
            "CFR": formation_result["cfr"],
            "over_pltn": formation_result["over_pltn"],
            "std_pltn": formation_result["std_pltn"],
            "sparse_pltn": formation_result["sparse_pltn"],
            "avg_pltn_size": formation_result["avg_pltn_size"],
            "audit_decision_count": audit["decision_count"],
            "audit_reward_count": audit["reward_count"],
            "audit_matched_count": audit["matched_count"],
            "audit_unmatched_decision_count": audit[
                "unmatched_decision_count"
            ],
            "audit_unmatched_reward_count": audit[
                "unmatched_reward_count"
            ],
            "audit_duplicate_decision_key_count": audit[
                "duplicate_decision_key_count"
            ],
            "runtime": time.time() - started_at,
        }
        _add_prefixed(row, "overall", overall)
        _add_prefixed(row, "ce", ce)
        _add_prefixed(row, "se", se)
        hpc_utils.write_one_row_csv(args.out_csv, row)
        print(f"[Dispatch Eval] Saved: {args.out_csv}")
        return row
    finally:
        if formation is not None:
            for name in ("split_agent", "collect_agent"):
                handler = getattr(formation, name, None)
                writer = getattr(getattr(handler, "agent", None), "writer", None)
                if writer is not None:
                    writer.close()
            gate_agent = getattr(formation.tsg_manager, "gate_agent", None)
            if gate_agent is not None:
                gate_agent.writer.close()
        traci.close()


def main(args=None, root=None):
    del root
    prc.PRINT_ENABLED = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch_model_path", required=True)
    parser.add_argument("--model_id", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--train_interval", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--av_p", type=float, default=0.1)
    parser.add_argument("--m_fr", type=float, default=1500)
    parser.add_argument("--st", type=int, default=1500)
    parser.add_argument("--max_team_size", type=int, default=12)
    parser.add_argument("--gui", action="store_true")
    parsed = parser.parse_args(args=args)
    run_evaluation(parsed)


if __name__ == "__main__":
    main()
