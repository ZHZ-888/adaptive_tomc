'''
hpc_utils.py

Shared utility functions for HPC simulation tasks.
Includes:
- Standard argument parsing
- Performance indicator calculations
- CSV logging
'''

import csv
import argparse
from pathlib import Path
from functions import calc_ttc_sd_exposure
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np


DEFAULT_WARMUP_TIME = 180


def standard_arg_parser():
    """Returns a parser with the standard arguments for your experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--r_fr", type=float, default=800)
    parser.add_argument("--m_fr", type=float, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--st", type=int, default=1500)
    parser.add_argument('--av_p', type=float, default=0.1, help='AV Penetration Rate')
    parser.add_argument(
        "--max_team_size",
        type=int,
        default=12,
        help="Maximum platoon size used for tagging and evaluation",
    )
    parser.add_argument(
        "--tsg_mode",
        type=str,
        default="predict",
        choices=["off", "train", "predict", "audit", "fix"],
    )
    parser.add_argument(
        "--r_platoon_p",
        type=float,
        default=0.7,
        help="Proportion of standard ramp platoons",
    )
    return parser

def training_arg_parser():
    """Parser for RL Training"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--av_p", type=float, default=0.1)
    parser.add_argument("--r_fr", type=float, default=0)
    parser.add_argument("--m_fr", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--st", type=int, default=1200*100)
    parser.add_argument("--train_agent", type=str, choices=['SA', 'CA', 'TSG', None], default=None)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--train_interval", type=int, default=32) # update_interval = batch_size*2
    # Use nargs='+' to allow multiple integers: --batch_epoch 16 5
    parser.add_argument("--hidden_layer", type=int, nargs='+', default=[32, 32])
    # Note: Training usually doesn't need out_csv for traffic KPIs
    return parser

def get_mc_indicator(speed_log, tp, ssm_path, runtime,
                     max_time=None, warmup_time=DEFAULT_WARMUP_TIME):
    """Calculates performance indicators after simulation."""
    duration = (max_time if max_time is not None else 1500) - warmup_time
    total_vehicle_num = int(tp * max(duration, 0) / 3600)
    ttc_metrics_3 = calc_ttc_sd_exposure.calc_ttc_conflict_metrics(
        ssm_path, total_vehicle_num, ttc_threshold=3, min_time=warmup_time, max_time=max_time)
    ttc_metrics_2 = calc_ttc_sd_exposure.calc_ttc_conflict_metrics(
        ssm_path, total_vehicle_num, ttc_threshold=2, min_time=warmup_time, max_time=max_time)
    ttc_metrics_1 = calc_ttc_sd_exposure.calc_ttc_conflict_metrics(
        ssm_path, total_vehicle_num, ttc_threshold=1.5, min_time=warmup_time, max_time=max_time) # 1.5
    ttc_ratio_3 = ttc_metrics_3[2]
    ttc_ratio_2 = ttc_metrics_2[2]
    ttc_ratio_1 = ttc_metrics_1[2]
    warmup_step = int(warmup_time * 10)
    avg_speeds = [item[1] for item in speed_log if item[0] >= warmup_step]
    clean_avg_speeds = [v for v in avg_speeds if v is not None]
    average_v = sum(clean_avg_speeds) / len(clean_avg_speeds)
    print("\n           ---merging control performance indicators---")
    print(f'tp: {tp:.0f} veh/h, average_v:{average_v:.2f} m/s, '
          f'ttc_ratio_3: {ttc_ratio_3:.8f}, '
          f'mr_confilct_cnt_3: {ttc_metrics_3[3]}, '
          f'ttc_ratio_2: {ttc_ratio_2:.8f}, '
          f'ttc_ratio_1.5: {ttc_ratio_1:.8f}, '
          f'execution_time:{runtime:.2f} s')
    return tp, average_v, ttc_ratio_3, ttc_ratio_2, ttc_ratio_1, runtime


def get_mr_ttc_ratios(ssm_path, ramp_entry_ids, max_time=None,
                      warmup_time=DEFAULT_WARMUP_TIME):
    """Calculate M-R TTC exposure ratios among ramp entries."""
    if not ramp_entry_ids:
        return 0.0, 0.0, 0.0

    ratios = []
    for threshold in (3, 2, 1.5):
        metrics = calc_ttc_sd_exposure.calc_ttc_conflict_metrics(
            ssm_path,
            ttc_threshold=threshold,
            min_time=warmup_time,
            max_time=max_time,
        )
        ratios.append(len(metrics[3] & ramp_entry_ids) / len(ramp_entry_ids))
    print(f'mr_ttc: {ratios}')
    return tuple(ratios)


def get_delay_indicator(tripinfo_path, warmup_time=DEFAULT_WARMUP_TIME):
    """
    Calculate delay indicators from tripinfo XML file.
    Parameters
    ----------
    tripinfo_path : str
        "/home/zzha/PycharmProjects/RampMerging_2026/data
        /multi_lane/algo/tripinfo_0_0.1_9_0_mts12_local.xml"
        Path to the tripinfo XML file.

    Returns
    -------
    dict
        Dictionary containing delay indicators.
    """
    tree = ET.parse(tripinfo_path)
    root = tree.getroot()

    overall = []
    mainline = []
    ramp = []

    for trip in root.findall("tripinfo"):
        depart = float(trip.attrib.get("depart", 0))
        if depart < warmup_time:
            continue

        veh_id = trip.attrib["id"]
        time_loss = float(trip.attrib["timeLoss"])

        overall.append(time_loss)

        if veh_id.startswith("r"):
            ramp.append(time_loss)
        else:
            if time_loss > 10:
                pass
            mainline.append(time_loss)

    result = {
        "avg_time_loss": np.mean(overall) if overall else np.nan,
        "mainline_time_loss": np.mean(mainline) if mainline else np.nan,
        "ramp_time_loss": np.mean(ramp) if ramp else np.nan,
        "completed_veh": len(overall),
        "completed_mainline": len(mainline),
        "completed_ramp": len(ramp),
    }
    print("\n           ---delay indicators---")
    print(result)
    return result


def get_mrm_insertion_counts(fcd_path, warmup_time=DEFAULT_WARMUP_TIME, st=1500):
    """Count ramp entries into ws_1 and entries forming an M-R-M sequence."""
    ramp_vehicles_entered_ws_1 = set()
    ramp_entry_count = 0
    mrm_insertion_count = 0
    ramp_entry_ids = set()

    for _, timestep in ET.iterparse(fcd_path, events=("end",)):
        if timestep.tag != "timestep":
            continue

        sim_time = float(timestep.attrib["time"])
        if sim_time > st:
            break

        vehicles = {
            vehicle.attrib["id"]: (
                vehicle.attrib["lane"],
                float(vehicle.attrib["pos"]),
            )
            for vehicle in timestep
        }
        new_ramp_entries = [
            vehicle_id
            for vehicle_id, (lane_id, _) in vehicles.items()
            if (vehicle_id.startswith("r")
                and lane_id == "ws_1"
                and vehicle_id not in ramp_vehicles_entered_ws_1)
        ]
        ramp_vehicles_entered_ws_1.update(new_ramp_entries)

        if sim_time >= warmup_time:
            ramp_entry_ids.update(new_ramp_entries)
            ws_1_vehicles = sorted(
                (state[1], vehicle_id)
                for vehicle_id, state in vehicles.items()
                if state[0] == "ws_1"
            )
            ws_1_indices = {
                vehicle_id: index
                for index, (_, vehicle_id) in enumerate(ws_1_vehicles)
            }

            for vehicle_id in new_ramp_entries:
                ramp_entry_count += 1

                index = ws_1_indices[vehicle_id]
                if 0 < index < len(ws_1_vehicles) - 1:
                    rear_id = ws_1_vehicles[index - 1][1]
                    front_id = ws_1_vehicles[index + 1][1]
                    if rear_id.startswith("m") and front_id.startswith("m"):
                        mrm_insertion_count += 1

        timestep.clear()
    print(f"Ramp entries: {ramp_entry_count}, M-R-M insertions: {mrm_insertion_count}")
    return ramp_entry_count, mrm_insertion_count, ramp_entry_ids

def get_fc_detail(dic_follower_state, his_dic_platoon_size, max_size=12):
    """
    Get formation control statistics.

    Parameters
    ----------
    dic_follower_state : dict
        Format:
        {'mhv48': ['following_mode', 'mav38'], 'mhv65': ['free_mode', 'mav38']}
    his_dic_platoon_size : dict
        Format:
        {'mav38': 11, 'mav278': 8, 'mav754': 15}
    max_size : int, optional
        Maximum allowed platoon size.
    Returns
    -------
    dict
        {'over_pltn': int,
        'non_over_pltn': int,
        'std_pltn': int,
        'sparse_pltn': int,
        'avg_pltn_size': float}
    """
    ls_oversized_leader = []
    ls_sparse_leader = []

    over_pltn = 0
    non_over_pltn = 0
    std_pltn = 0
    sparse_pltn = 0

    # Store follower states for each leader
    leader_follower_states = defaultdict(list)

    for follower_id, (state, leader_id) in dic_follower_state.items():
        leader_follower_states[leader_id].append(state)

    # Calculate average platoon size
    total_size = sum(his_dic_platoon_size.values())

    for leader_id, platoon_size in his_dic_platoon_size.items():

        # Count oversized platoons
        if platoon_size > max_size:
            over_pltn += 1
            ls_oversized_leader.append(leader_id)
            continue

        # Count non-oversized platoons
        non_over_pltn += 1

        follower_states = leader_follower_states.get(leader_id, [])

        # A non-oversized platoon is considered sparse
        # if any follower is not in following_mode.
        if any(state != "following_mode" for state in follower_states):
            ls_sparse_leader.append(leader_id)
            sparse_pltn += 1
        else:
            std_pltn += 1

    avg_pltn_size = (
        total_size / len(his_dic_platoon_size)
        if his_dic_platoon_size
        else 0
    )

    sum_platoon_count = over_pltn + non_over_pltn


    print("\nNon standard platoon leaders:")
    print(f"Oversized leaders: {ls_oversized_leader}")
    print(f"Sparse leaders: {ls_sparse_leader}")

    print("\nFormation Control Statistics:")
    print(f'Sum (Over, NonOver): {sum_platoon_count} ({over_pltn}, {non_over_pltn})')
    print(f'NonOver (Sparse, Standard): {non_over_pltn} ({sparse_pltn}, {std_pltn})')
    print(f"Avg Platoon Size  : {avg_pltn_size:.2f}")

    num_follower = len(dic_follower_state)
    num_platoon_follower = len([k for k, v in dic_follower_state.items() if v[0] == 'following_mode'])
    cfr = round(num_platoon_follower / num_follower, 3)
    spr = round(std_pltn / sum_platoon_count, 3)
    print("\n           ---formation control performance indicators---")
    print(f"CFR: {num_platoon_follower} platoon_followers, {num_follower} followers, "
          f"ratio: {cfr * 100:.2f}%")
    print(f"SPR: {std_pltn} standard_platoon, {sum_platoon_count} platoon, "
          f"ratio: {spr * 100:.2f}%, ")

    result = {
        "over_pltn": over_pltn,
        "non_over_pltn": non_over_pltn,
        "std_pltn": std_pltn,
        "sparse_pltn": sparse_pltn,
        "avg_pltn_size": avg_pltn_size,
        "cfr": cfr,
        "spr": spr
    }
    return result

def write_one_row_csv(path: str, row: dict):
    """Writes a single row to a CSV file safely."""
    # Safety check: do nothing if no path is provided
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    file_exists = p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def write_dataframe_csv(path: str, df):
    """
    Save a DataFrame to CSV.
    """
    if path is None:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(p, index=False)

if __name__ == "__main__":
    # fcd_path = ("/home/zzha/PycharmProjects/RampMerging_2026/data/multi_lane/algo/"
    #             "trj_1400_0.1_4_0_mts12_local.xml")
    # fcd_path = ("/home/zzha/PycharmProjects/RampMerging_2026/data/multi_lane/algo/"
    #             "trj_rm_1400_0_3_20260808_142224.xml")
    fcd_path = ("/home/zzha/PycharmProjects/RampMerging_2026/data/multi_lane/algo/"
                "trj_fifo_1400_0_3_20260807_210226.xml")
    ramp_entry_count, mrm_insertion_count, _ = get_mrm_insertion_counts(fcd_path=fcd_path, st=1500)
    print(ramp_entry_count, mrm_insertion_count)

