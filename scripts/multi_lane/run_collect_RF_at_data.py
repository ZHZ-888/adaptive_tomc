'''
run_collect_RF_at_data.py
random forest arrival time feature/target data collection
multi-lane simulation
'''
import os
import random
import time
import pandas as pd
from pathlib import Path

from functions import vehicle_generation3 as vg
from functions import print_control as prc  # the shared fuction of print control
from functions import formation_controller as fc
from functions import merging_controller as mc
from functions import data_recording as dr
# Shared tools for arguments, KPIs, and CSV logging.// CLI and HPC
from functions import hpc_utils


def mpgc_main(av_p, r_fr, m_fr, seed, r_autoFollow_p=0,
              r_platoon_p=1, loss_rate=0, sumo_mode='no_gui', plot=False,
              display=False, lc=False, st=1500, jam_mode_core='legacy',
              yield_ratio_ranges=None):
    # SUMO SETTING
    ROOT = Path(__file__).resolve().parents[2]
    sumo_config_path = (ROOT
            / "road_network"
            / "multi_lane_motorway"
            / "real"
            / "cfg_multi_lane_merge.sumocfg")
    # Simulation step length
    sim_step = 0.1
    sumo_home = "/home/zzha/opt/sumo-1.19.0-src"
    os.environ["SUMO_HOME"] = sumo_home
    gui_enabled = sumo_mode != "no_gui"
    sumo_bin = os.path.join(sumo_home, "bin", "sumo-gui" if gui_enabled else "sumo",)
    # Construct the SUMO command and options
    sumo_cmd = [sumo_bin, "-c", str(sumo_config_path),
                "--seed", str(seed),
                "--no-warnings"]  # , '-S' start auto, and quit auto
    sumo_options = ["--step-length", str(sim_step)]

    # Use libsumo-gui to avoid TraCI socket overhead (traci_gui).
    if sumo_mode == "traci_gui":
        os.environ.pop("LIBSUMO_GUI", None)
        import traci
    elif sumo_mode == "libsumo_gui":
        os.environ["LIBSUMO_GUI"] = "1"
        import libsumo as traci
    elif sumo_mode == "no_gui":
        os.environ.pop("LIBSUMO_GUI", None)
        import libsumo as traci
    traci.start(sumo_cmd + sumo_options)
    print("SUMO mode:", sumo_mode)
    print("SUMO engine:", traci.getVersion())
    print("SUMO binary:", sumo_bin)

    try:
        # VEHICLE GENERATOR
        max_attempts = 7
        av_p0, av_p1 = av_p, av_p
        m0_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p0, m_fr, seed)
        m1_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p1, m_fr, 100 - seed)
        r_dpt_type = vg.get_schedule2(st, av_p, r_fr, r_platoon_p,
                                      max_attempts, plot, seed, display)
        # ramp road vehicle depature schedule
        veh_gen = vg.VehGen(traci, seed)  # function related to veh generation
        data_recorder = dr.DataRecording(traci)

        formation_controller = fc.FormationController(data_recorder, traci,
                                                      fc_mode = 'dla_tsc_lhr',
                                                      jam_mode_core=jam_mode_core)
        action_rng = random.Random(seed + 10000)
        merging_controller = mc.MergingController(data_recorder, traci,
                                                  platoon_formation=True,
                                                  jam_mode_core=jam_mode_core,
                                                  action_rng=action_rng,
                                                  yield_ratio_ranges=yield_ratio_ranges)

        dic_targets, ls_features = \
            loop(traci, st, data_recorder, veh_gen, formation_controller, merging_controller, lc,
                 r_autoFollow_p, m0_dpt_type, m1_dpt_type, r_dpt_type,
                 jam_mode_core)
    finally:
        traci.close()
    return (dic_targets, ls_features)

def loop(traci, st, data_recorder,
         veh_gen, formation_controller, merging_controller,
         lc, r_autoFollow_p, m0_dpt_type=None, m1_dpt_type=None, r_dpt_type=None,
         jam_mode_core='legacy'):
    # START SIMULATION
    step = 0
    # scripts loop
    while step < st * 10:
        if step > 1128 * 10:
            pass
        traci.simulationStep()  # start simulation
        # vehicle generation without Heterogeneous
        veh_gen.veh_gen_homo(step, m1_dpt_type, 'm', 'route_m', 27.5, '0')

        # control
        formation_controller.step(st, step, lc)
        merging_controller.step(st, step)

        # record targets
        data_recorder.record_tail_arrival(step)
        step += 1

    # Stop generating vehicles and let the remaining platoons complete their trips.
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        formation_controller.step(st, step, lc)
        merging_controller.step(st, step)
        data_recorder.record_tail_arrival(step)
        step += 1

    dic_targets = data_recorder.dic_tail_arrived_ms
    ls_features = (data_recorder.ls_action_features
                   if jam_mode_core == 'sampling' else data_recorder.ls_features)
    return (dic_targets, ls_features)

def main(args=None, root=None):
    """
    Unified entry point for CLI (command line interface) / HPC.
    This function is called by run.py.
    It parses command-line arguments and calls mpgc_main().
    """
    prc.PRINT_ENABLED = False

    # 1. Parse Args (HPC/CLI Mode)
    parser = hpc_utils.standard_arg_parser()
    parser.add_argument(
        "--jam-mode-core",
        choices=("legacy", "dp", "sampling"),
        default="legacy")
    parsed_args = parser.parse_args(args=args)
    sumo_mode = "libsumo_gui" if parsed_args.gui else "no_gui"

    # 2. Run simulation
    # Call the original algorithm
    dic_targets, ls_features = mpgc_main(
        av_p=parsed_args.av_p,
        r_fr=parsed_args.r_fr,
        m_fr=parsed_args.m_fr,
        seed=parsed_args.seed,
        sumo_mode=sumo_mode,
        st=parsed_args.st,
        jam_mode_core=parsed_args.jam_mode_core)

    # 3. Save results
    if parsed_args.out_csv:
        collect_RF_data(dic_targets, ls_features, parsed_args.out_csv)

def collect_RF_data(dic_targets, ls_features, output_path):
    platoon_columns = [
        "platoon_type",
        "leader_to_pv_dis", # dis_to_pv
        "leader_speed", # speed leader
        "leader_left_dis", # remain_dis_leader
        "m"]
    free_flow_columns = [
        "leader_id",
        "record_index",
        "prediction_ts"] + platoon_columns
    action_columns = [
        "leader_id",
        "record_index",
        "action_start_ts"] + platoon_columns + [
        "est_head_dur",
        "yield_dur",
        "max_delta_t",
        "yield_ratio",
        "target_head_ts"]
    is_action_data = bool(
        ls_features and len(ls_features[0]) == len(action_columns))
    columns = action_columns if is_action_data else free_flow_columns
    df = pd.DataFrame(ls_features, columns=columns)
    tail_ts_column = "actual_tail_ts" if is_action_data else "arrival_ts"
    if is_action_data:
        df["actual_head_ts"] = df["leader_id"].map(
            lambda x: dic_targets[x][1] if x in dic_targets else None)
    df[tail_ts_column] = df["leader_id"].map(
        lambda x: dic_targets[x][2] if x in dic_targets else None)
    if is_action_data:
        df["head_to_tail_duration"] = (
            df["actual_tail_ts"] - df["actual_head_ts"])

    if output_path:
        # Ensure directory exists (optional but safe)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)


if __name__ == '__main__':
    prc.PRINT_ENABLED = False
    start = time.time()
    dic_targets, ls_features = mpgc_main(
        av_p = 0.3,
        r_fr = 0,
        m_fr = 1500, # 1500
        seed = 16,
        r_autoFollow_p = 0,
        r_platoon_p = 1,
        loss_rate = 0,
        sumo_mode = "libsumo_gui", # libsumo_gui/traci_gui/no_gui
        plot = False,
        display = False,
        lc = False,
        st = 1000,
        jam_mode_core = 'sampling') # dp, legacy, sampling (training data collection with yield action)
    end = time.time()

    # orgnise data
    # output_path = "/home/zzha/PycharmProjects/adaptive_tomc/data/features/test_261110.csv"
    # output_path = None
    collect_RF_data(dic_targets, ls_features, output_path=output_path)

