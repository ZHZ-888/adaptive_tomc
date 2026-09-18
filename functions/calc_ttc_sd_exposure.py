# tree = ET.parse("data/Traj/trj_720_0.1_3.xml")
# tree = ET.parse("data/Traj/trajectory_fifo_test.xml")
import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import math
from tqdm import tqdm   # pip install tqdm
from pathlib import Path


def _parse_xy(pos):
    if pos is None or pos == "NA":
        return None
    try:
        x_str, y_str = pos.split(",")
        return float(x_str), float(y_str)
    except (ValueError, AttributeError):
        return None


def _in_ms_lane(lane_id):
    return lane_id is not None and lane_id.startswith("ws_")


def calc_ttc_conflict_metrics(ssm_file, total_vehicles=None, ttc_threshold=3.0,
                              measure="TTC", min_time=None, max_time=None):
    """Calculate TTC or DRAC conflict metrics within the merging section.
        Return: avg_value, conflict_count, conflict_rate, risky_mr_ramp_ids
    """

    ssm_file = Path(ssm_file)
    if not ssm_file.exists():
        return 0.0, 0, 0.0, set()

    root = ET.parse(ssm_file).getroot()
    measure = measure.upper()

    span_name = "TTCSpan" if measure == "TTC" else "DRACSpan"
    min_name = "minTTC" if measure == "TTC" else "maxDRAC"

    pair_values = {}

    for conflict in root.findall(".//conflict"):
        ego = conflict.get("ego")
        foe = conflict.get("foe")
        pair = tuple(sorted((ego, foe)))

        time_span = conflict.find("timeSpan")
        measure_span = conflict.find(span_name)
        ego_lanes = conflict.find("egoLane")
        foe_lanes = conflict.find("foeLane")

        if (time_span is None or measure_span is None
                or ego_lanes is None or foe_lanes is None):
            continue

        times = time_span.get("values", "").split()
        values = measure_span.get("values", "").split()
        ego_lane = ego_lanes.get("values", "").split()
        foe_lane = foe_lanes.get("values", "").split()

        for t, value, e_lane, f_lane in zip(times, values, ego_lane, foe_lane):
            if value == "NA":
                continue

            try:
                t = float(t)
                value = float(value)
            except ValueError:
                continue

            if min_time is not None and t < min_time:
                continue
            if max_time is not None and t > max_time:
                continue

            if not (_in_ms_lane(e_lane) or _in_ms_lane(f_lane)):
                continue

            if measure == "TTC":
                if pair not in pair_values or value < pair_values[pair]:
                    pair_values[pair] = value
            else:
                if pair not in pair_values or value > pair_values[pair]:
                    pair_values[pair] = value

    mr_ramp_values = {}
    for pair, value in pair_values.items():
        first, second = pair
        is_mr_pair = (
            (first.startswith("m") and second.startswith("r"))
            or (first.startswith("r") and second.startswith("m"))
        )
        if not is_mr_pair:
            continue

        ramp_id = first if first.startswith("r") else second
        if (ramp_id not in mr_ramp_values
                or (measure == "TTC" and value < mr_ramp_values[ramp_id])
                or (measure != "TTC" and value > mr_ramp_values[ramp_id])):
            mr_ramp_values[ramp_id] = value

    if measure == "TTC":
        risky_values = [v for v in pair_values.values() if v < ttc_threshold]
        risky_mr_ramps = {rid for rid, v in mr_ramp_values.items() if v < ttc_threshold}
    else:
        risky_values = [v for v in pair_values.values() if v > ttc_threshold]
        risky_mr_ramps = {rid for rid, v in mr_ramp_values.items() if v > ttc_threshold}

    conflict_count = len(risky_values)
    avg_value = float(np.mean(risky_values)) if risky_values else 0.0

    conflict_rate = (
        conflict_count / total_vehicles
        if total_vehicles is not None and total_vehicles > 0
        else 0.0
    )

    return avg_value, conflict_count, conflict_rate, risky_mr_ramps


if __name__ == "__main__":
    ssm_file = (
        "/home/zzha/PycharmProjects/RampMerging_2026/"
        "data/multi_lane/algo/ssm_rm_1400_0_0_20260807_194334.xml"
    )
    avg_min_ttc, conflict_count, conflict_rate, mr_ramp_ids = calc_ttc_conflict_metrics(
        ssm_file, total_vehicles=1235, ttc_threshold=3)
    print(f"Avg min TTC: {avg_min_ttc:.2f}, Conflict Count: {conflict_count}, "
          f"Conflict Rate: {conflict_rate:.4f}, M-R Ramp Count: {len(mr_ramp_ids)}")