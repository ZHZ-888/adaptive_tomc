'''
run_mpgc_multi_lane_test_PF_performance.py
multi-lane simulation
test the performance of platoon formation
'''
import os
import time
from pathlib import Path
import itertools

import torch
import numpy as np
import random

from functions import vehicle_generation3 as vg
from functions import print_control as prc  # the shared fuction of print control
from functions import formation_controller as fc
from functions import merging_controller as mc
from functions import data_recording as dr
# Shared tools for arguments, KPIs, and CSV logging.// CLI and HPC
from functions import hpc_utils


def mpgc_main(av_p, r_fr, m_fr, seed, r_autoFollow_p=0, r_platoon_p=1, loss_rate=0,
              deploy_agent = None,
              gui=False, plot=False, display=False, lc=True, st=1200,
              model_name=None):
    set_global_seed(seed, enable=True)
    # SUMO SETTING
    ROOT = Path(__file__).resolve().parents[2]
    sumo_config_path = (ROOT
                        / "road_network"
                        / "multi_lane_motorway"
                        / "real"
                        / "cfg_multi_lane_merge.sumocfg"
    )
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    '''
    Look for a system setting named TRAJ_DIR. If it exists, use it. If not, use this default folder.
    '''
    traj_dir = Path(os.environ.get("TRAJ_DIR", ROOT / "data" / "multi_lane" / "algo"))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
    file_name = f'trj_{r_fr}_{av_p}_{seed}_{loss_rate}.xml'
    xml_path = os.path.join(traj_dir, file_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sumo_cmd = [sumo_bin, "-c", str(sumo_config_path),
                "--seed", str(seed),
                "--fcd-output", str(xml_path), # save path
                "--no-warnings"]  #
    sumo_options = ["--step-length", str(sim_step)]

    # If GUI is enabled, set the GUI view schema
    if gui:
        import traci
        traci.start(sumo_cmd + sumo_options)
        available_views = traci.gui.getIDList()
        print("Available Views:", available_views)
    else:
        import libsumo as traci
        traci.start(sumo_cmd + sumo_options)
    try:
        # VEHICLE GENERATOR
        # scripts road veh depature schedule (lane0 and lane1)
        max_attempts = 7
        av_p0, av_p1 = av_p, av_p
        m0_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p0, m_fr, seed)
        m1_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p1, m_fr, 100 - seed)
        r_dpt_type = vg.get_schedule2(st, av_p, r_fr, r_platoon_p,
                                      max_attempts, plot, seed, display)
        # ramp road veh depature schedule
        veh_gen = vg.VehGen(traci)  # function related to veh generation
        data_recorder = dr.DataRecording(traci)
        data_recorder.get_avhid_ptype(r_dpt_type = r_dpt_type)  # here only have r_dpt_type

        if deploy_agent == 'SA':
            SA_mode, CA_mode = 'predict', None
        elif deploy_agent == 'CA':
            SA_mode, CA_mode = None, 'predict'
        elif deploy_agent is None: # If None, both agents remain in 'predict'
            SA_mode, CA_mode = None, None
        elif deploy_agent == 'both':
            SA_mode, CA_mode = 'predict', 'predict'
        print(f'***** evaluate {deploy_agent} *****')
        formation_controller = fc.FormationController(data_recorder, traci, splitting_agent=SA_mode,
                                                      collecting_agent=CA_mode, model_name=model_name)
        merging_controller = mc.MergingController(data_recorder, traci, av_p,
                                                  platoon_formation=True, ml=True)

        (dic_score_reward, dic_follower_state, his_dic_platoon_size,
         dic_id_features, tp, speed_log, queue_log) = \
            loop(traci, st, data_recorder, veh_gen, formation_controller, merging_controller, lc,
                 r_autoFollow_p, m0_dpt_type, m1_dpt_type, r_dpt_type)
    finally:
        traci.close()
    return (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
            tp, speed_log, queue_log, xml_path)

def loop(traci, st, data_recorder, veh_gen, formation_controller, merging_controller, lc,
         r_autoFollow_p, m0_dpt_type=None, m1_dpt_type=None, r_dpt_type=None):
    # START SIMULATION
    step = 0
    # scripts loop
    while step < st * 10:
        # checkpoint
        if step > 1100 * 10:
             pass
        traci.simulationStep()  # start simulation

        c_ts = traci.simulation.getTime()  # current_timestep
        if c_ts % 1 == 0:
            prc.print_message(f'************current_time, step:{c_ts, step}************')

        # main vehicle generation (1 => inner lane; 0 => outer lane)
        veh_gen.veh_gen_homo(step, m1_dpt_type, 'm', 'route_m', 27.5, '1')  # 30m/s => 110km/h
        veh_gen.veh_gen_homo(step, m0_dpt_type, 'm', 'route_m', 27.5, '0')  # 25m/s => 90km/h; ori 29.5

        # ramp vehicle generation
        veh_gen.platoon_gen(step, r_dpt_type, 'r', r_autoFollow_p)

        (dic_score_reward, dic_follower_state, his_dic_platoon_size,
         dic_id_features) = formation_controller.step(st, step, lc)
        tp, speed_log, queue_log = merging_controller.step(st, step, r_dpt_type)

        data_recorder.record_tail_arrival(step)
        step += 1
    return (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
            tp, speed_log, queue_log)

def main(args=None, root=None):
    """
    Unified entry point for CLI (command line interface) / HPC.
    This function is called by run.py.
    It parses command-line arguments and calls mpgc_main().
    """
    prc.PRINT_ENABLED = False

    # 1. Parse Args (HPC/CLI Mode)
    parser = hpc_utils.standard_arg_parser()
    parsed_args = parser.parse_args(args=args)

    # 2. Run simulation
    # Call the original algorithm
    start = time.time()
    (dic_score_reward, dic_follower_state, his_dic_platoon_size,
     dic_id_features, tp, speed_log, queue_log, xml_path) = mpgc_main(
        av_p=parsed_args.av_p,
        r_fr=parsed_args.r_fr,
        m_fr=parsed_args.m_fr,
        seed=parsed_args.seed,
        gui=parsed_args.gui
    )
    end = time.time()
    runtime = end - start

    hpc_utils.get_fc_indicator(dic_follower_state, his_dic_platoon_size)
    tp, average_v, ttc_ratio, avg_speed_std, runtime = (
        hpc_utils.get_mc_indicator(speed_log, tp, xml_path, runtime))

    # 3. Save results
    if parsed_args.out_csv:
        row = {
            "algo": "mpgc_multi_lane",
            "av_p": parsed_args.av_p,
            "ramp_demand": parsed_args.r_fr,
            "mainline_demand": parsed_args.m_fr,
            "seed": parsed_args.seed,
            "throughput": tp,
            "avg_speed": average_v,
            "ttc_ratio": ttc_ratio,
            "avg_speed_std": avg_speed_std,
            "runtime": runtime
        }
        hpc_utils.write_one_row_csv(parsed_args.out_csv, row)

def set_global_seed(seed, enable=True):
    """Fix all sources of randomness globally
    (external traffic-environment constraints + internal neural-network constraints)"""
    if not enable:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def _set_dynamic_traffic(step, start_t, r_dpt_type, dynamic=True):
    '''
    set dynamic traffic. For example: default r_demands=720 veh/h; start_t=5 min, new_fr=180 veh/h;
                                      after 5 min simulation, 720 veh/h => 180 veh/h
    '''
    # 100324update: new ramp flow rate after 350
    start_t = 300
    dynamic = False
    if step == start_t * 10 and dynamic:
        p = 0.3
        new_fr = 180
        seed = 1
        new_r_dpt_type = vg.get_schedule_startT(start_t, p, new_fr, max_attempts=1, seed=seed)  # after start_t
        r_dpt_type = {key: value for key, value in r_dpt_type.items() if key <= start_t}  # before start_t
        r_dpt_type.update(new_r_dpt_type)  # merge together
    return r_dpt_type

if __name__ == '__main__':
    prc.PRINT_ENABLED = False

    dic_res = {} # {model_name: [ca_indicator_avg, sa_indicator_avg]}

    update_intervals = [8, 16, 32, 64]
    lrs = [0.0005, 0.001, 0.005, 0.01]
    seeds = [0, 1, 2, 3, 4] # for each model, run 5 times with different seeds and calculate the average performance

    for ui, lr in itertools.product(update_intervals, lrs):
        model_name = f'sa_i{ui}_{lr}.pt' # example: sa_i16_0.005

        ls_ca_idc = []
        ls_sa_idc = []
        for seed in seeds:
            (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
             tp, speed_log, queue_log, xml_path) = mpgc_main(
                av_p = 0.2, # 0.3
                r_fr = 0, # 1000
                m_fr = 1200, # 1500
                seed = seed,
                r_autoFollow_p = 1,  # auto follow proportion
                r_platoon_p = 1, # percentage of platoon vehicles
                loss_rate = 0,
                deploy_agent = None, # 'CA', 'SA', 'both', None
                gui = False,
                plot = False,
                display = True,
                lc = False, # if consider HV lane-changing; True
                st = 1200,
                model_name = model_name)
            ca_idc, sa_idc = hpc_utils.get_fc_indicator(dic_follower_state, his_dic_platoon_size)
            ls_ca_idc.append(ca_idc)
            ls_sa_idc.append(sa_idc)

        print(f"\n***********************{model_name}************************")
        ca_indicator_avg = round(sum(ls_ca_idc) / len(ls_ca_idc), 3) if ls_ca_idc else 0.0
        sa_indicator_avg = round(sum(ls_sa_idc) / len(ls_sa_idc), 3) if ls_sa_idc else 0.0
        print(f"ca_indicator_detail: {ls_ca_idc}, sa_indicator_detail: {ls_sa_idc}")
        print(f"ca_indicator_avg: {ca_indicator_avg:.3f}, sa_indicator_avg: {sa_indicator_avg:.3f}")
        dic_res[model_name] = [ca_indicator_avg, sa_indicator_avg]
        print(dic_res)