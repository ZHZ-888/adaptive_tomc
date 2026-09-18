'''
run_mpgc_multi_lane.py
multi-lane simulation
test longer platoon
'''
import argparse
import os
import time
from pathlib import Path
from datetime import datetime

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
from functions import merging_traffic_calibrator as mtc
from functions.computation_logger import ComputationLogger, measure_cycle
from functions import accident_simulation

def mpgc_main(av_p, r_fr, m_fr, seed, r_autoFollow_p=0, r_platoon_p=1,
              loss_rate=0, gui=False, plot=False, display=False,
              lc=True, st=1500, tsg_mode='predict', max_team_size=12,
              fc_mode='full'):
    set_global_seed(seed, enable=True)  # set global random seed (especially for RL training)
    comm_rng = random.Random(seed + 100)
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
    # sumo_bin = 'sumo-gui' if gui else 'sumo'

    sumo_home = "/home/zzha/opt/sumo-1.19.0-src"
    os.environ["SUMO_HOME"] = sumo_home
    sumo_bin = os.path.join(sumo_home, "bin", "sumo-gui" if gui else "sumo",)

    # Construct the SUMO command and options
    '''
    Look for a system setting named TRAJ_DIR. If it exists, use it. If not, use this default folder.
    '''
    traj_dir = Path(os.environ.get("TRAJ_DIR", ROOT / "data" / "multi_lane" / "algo"))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
    size_tag = f"mts{max_team_size}"
    file_name = f"trj_{r_fr}_{av_p}_{seed}_{loss_rate}_{size_tag}_{fc_mode}_{task_id}.xml"
    xml_path = os.path.join(traj_dir, file_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    lc_file_name = f"lc_{r_fr}_{av_p}_{seed}_{loss_rate}_{size_tag}_{fc_mode}_{task_id}.xml"
    lc_path = os.path.join(traj_dir, lc_file_name)
    ssm_file_name = f"ssm_{r_fr}_{av_p}_{seed}_{loss_rate}_{size_tag}_{fc_mode}_{task_id}.xml"
    ssm_path = traj_dir / ssm_file_name
    # delay indicators
    trip_file_name = f'tripinfo_{r_fr}_{av_p}_{seed}_{loss_rate}_{size_tag}_{fc_mode}_{task_id}.xml'
    tripinfo_path = os.path.join(traj_dir, trip_file_name)
    # computation save location
    comp_save_dir = Path(os.environ.get("COMP_DIR", ROOT / "data" / "computation_record"))
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    comp_file_name = f"comp_{r_fr}_{av_p}_{seed}_{loss_rate}_{size_tag}_{fc_mode}_{task_id}_{run_stamp}.csv"
    comp_save_dir.mkdir(parents=True, exist_ok=True)
    comp_logger = ComputationLogger(comp_save_dir, filename=comp_file_name)

    sumo_cmd = [sumo_bin, "-c", str(sumo_config_path),
                "--seed", str(seed),
                # fcd trajectory path
                "--fcd-output", str(xml_path),
                # lane-change output
                "--lanechange-output", str(lc_path),
                # tripinfo output/delay indicators
                "--tripinfo-output", str(tripinfo_path),
                # SSM output for TTC conflicts
                "--device.ssm.probability", "1",
                "--device.ssm.file", str(ssm_path),
                "--device.ssm.measures", "TTC",
                "--device.ssm.thresholds", "3", # record 3s; output (default) 1.5s, see calc_ttc_sd_exposure.py
                "--device.ssm.trajectories", "true",
                "--device.ssm.write-lane-positions", "true",
                "--no-warnings"]  #
    sumo_options = ["--step-length", str(sim_step)]

    # If GUI is enabled, set the GUI view schema
    if gui:
        import traci
        traci.start(sumo_cmd + sumo_options)

        print("SUMO engine:", traci.getVersion())
        print("SUMO binary:", sumo_bin)

        available_views = traci.gui.getIDList()
        print("Available Views:", available_views)
    else:
        import libsumo as traci
        # import traci
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
        veh_gen = vg.VehGen(traci, seed)  # function related to veh generation
        data_recorder = dr.DataRecording(traci)
        data_recorder.max_platoon_size = max_team_size
        data_recorder.get_avhid_ptype(r_dpt_type = r_dpt_type)  # here only have r_dpt_type



        formation_controller = fc.FormationController(data_recorder, traci,
                                                      loss_rate=loss_rate, tsg_mode=tsg_mode,
                                                      max_team_size=max_team_size,
                                                      fc_mode=fc_mode,
                                                      comm_rng=comm_rng) # fix/off/predict/train/audit
        merging_controller = mc.MergingController(data_recorder, traci, av_p,
                                                  platoon_formation=True, ml=True,
                                                  loss_rate=loss_rate,
                                                  warmup_time=hpc_utils.DEFAULT_WARMUP_TIME,
                                                  comm_rng=comm_rng, comp_logger=comp_logger)

        (dic_follower_state, his_dic_platoon_size,
         dic_id_features, tp, speed_log, queue_log, ts_first_jam, ts_first_back_to_regular) = \
            loop(traci, st, data_recorder, veh_gen, formation_controller, merging_controller,
                 lc, r_autoFollow_p, m0_dpt_type, m1_dpt_type, r_dpt_type, comp_logger,
                 sim_step=sim_step)

        split_reward_log = (
            {av_id: list(v) for av_id, v in formation_controller.split_agent.dic_score_reward.items()}
            if formation_controller.split_agent else {})
        collect_reward_log = (
            {av_id: list(v) for av_id, v in formation_controller.collect_agent.dic_score_reward.items()}
            if formation_controller.collect_agent else {})

        se_rewards = [v[1] for v in split_reward_log.values() if len(v) == 2]
        se_counts = len(se_rewards)
        se_reward_sum = sum(se_rewards)
        se_reward_avg = se_reward_sum / se_counts if se_counts else 0
        se_result = [se_counts, se_reward_sum, se_reward_avg]

        ce_rewards = [v[1] for v in collect_reward_log.values() if len(v) == 2]
        ce_counts = len(ce_rewards)
        ce_reward_sum = sum(ce_rewards)
        ce_reward_avg = ce_reward_sum / ce_counts if ce_counts else 0
        ce_result = [ce_counts, ce_reward_sum, ce_reward_avg]
        output_file_path = {'xml_path': xml_path, 'ssm_path': ssm_path, 'tripinfo_path': tripinfo_path}

    finally:
        traci.close()
    return (dic_follower_state, his_dic_platoon_size, dic_id_features,
            tp, speed_log, queue_log, output_file_path,
            se_result, ce_result, ts_first_jam, ts_first_back_to_regular)

def loop(traci, st, data_recorder,
         veh_gen, formation_controller,
         merging_controller, lc, r_autoFollow_p,
         m0_dpt_type=None, m1_dpt_type=None, r_dpt_type=None,
         comp_logger=None, sim_step=0.1):
    # START SIMULATION
    step = 0
    horizon_steps = st * 10
    traffic_calibrator = mtc.MergingTrafficCalibrator(traci)
    # scripts loop
    while step < horizon_steps:
        # checkpoint
        if step > 600 * 10:
             pass
        traci.simulationStep()  # start simulation

        c_ts = traci.simulation.getTime()  # current_timestep
        if c_ts % 1 == 0:
            prc.print_message(f'************current_time, step:{c_ts, step}************')

        # main vehicle generation (1 => inner lane; 0 => outer lane)
        veh_gen.veh_gen_hetero(step, m0_dpt_type, 'm', 'route_m', 27.5, '0')  # 25m/s => 90km/h; ori 29.5
        veh_gen.veh_gen_hetero(step, m1_dpt_type, 'm', 'route_m', 27.5, '1')  # 30m/s => 110km/h # veh_gen_homo
        # veh_gen.veh_gen_homo(step, m0_dpt_type, 'm', 'route_m', 27.5, '0')  # 25m/s => 90km/h; ori 29.5
        # veh_gen.veh_gen_homo(step, m1_dpt_type, 'm', 'route_m', 27.5, '1')

        # ramp vehicle generation
        veh_gen.platoon_gen(step, r_dpt_type, 'r', r_autoFollow_p)

        # traffic_calibrator.update()
        with measure_cycle(
                comp_logger,
                step=step,
                interval=10,
                module="PF",
                sim_time_s=step * sim_step,
        ):
            (dic_follower_state, his_dic_platoon_size,
             dic_id_features) = formation_controller.step(st, step, lc)

        with measure_cycle(
                comp_logger,
                step=step,
                interval=merging_controller.mpc_interval,
                module="MC",
                sim_time_s=step * sim_step,
        ):
            (tp, queue_log, ts_first_jam,
             ts_first_back_to_regular) = merging_controller.step(st, step, r_dpt_type)

        # accident_simulation.sudden_accident(traci, step, data_recorder)

        data_recorder.record_tail_arrival(step)
        step += 1

    # Flush-out stage: stop generating vehicles, but keep merging control active
    # so stopped ramp vehicles can be released and complete their trips.
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        traffic_calibrator.update()
        data_recorder.record_multi_lane_info()
        formation_controller.step(st, step, lc)
        merging_controller.step(st, step, r_dpt_type)
        step += 1

    del merging_controller.speed_log[st*10+1:]
    speed_log = merging_controller.speed_log

    return (dic_follower_state, his_dic_platoon_size, dic_id_features,
            tp, speed_log, queue_log, ts_first_jam, ts_first_back_to_regular)

def main(args=None, root=None):
    """
    Unified entry point for CLI (command line interface) / HPC.
    This function is called by run.py.
    It parses command-line arguments and calls mpgc_main().
    """
    prc.PRINT_ENABLED = False

    # 1. Parse Args (HPC/CLI Mode)
    parser = hpc_utils.standard_arg_parser()
    parser.add_argument('--fc_mode', choices=fc.FC_MODES,
                        default='full', help='Platoon formation ablation mode')
    parser.add_argument('--lc', action=argparse.BooleanOptionalAction,
                        default=True, help='Enable background lane-changing control')
    parser.add_argument('--loss_rate', type=float, default=0.0,
                        help='Packet loss rate for MPGC communication')
    parsed_args = parser.parse_args(args=args)

    # 2. Run simulation
    # Call the original algorithm
    start = time.time()
    (dic_follower_state, his_dic_platoon_size, dic_id_features,
     tp, speed_log, queue_log, output_file_path,
     se_result, ce_result, ts_first_jam, ts_first_back_to_regular) = mpgc_main(
        av_p=parsed_args.av_p, # default 0.1
        r_fr=parsed_args.r_fr, # default 800
        m_fr=parsed_args.m_fr, # default 1500; parsed_args.m_fr
        seed=parsed_args.seed,
        r_platoon_p=parsed_args.r_platoon_p,
        loss_rate=parsed_args.loss_rate,
        gui=parsed_args.gui,
        lc=parsed_args.lc,
        fc_mode=parsed_args.fc_mode,
        st=parsed_args.st,
        tsg_mode=parsed_args.tsg_mode, # default 'predict'
        max_team_size=parsed_args.max_team_size, # default 12
    )
    end = time.time()
    runtime = end - start
    # CFR: coupled_following_ratio; SPR: standard_size_platoon_ratio
    res = hpc_utils.get_fc_detail(dic_follower_state, his_dic_platoon_size,
                                  max_size=parsed_args.max_team_size)
    tp, average_v, ttc_ratio_3, ttc_ratio_2, ttc_ratio_1, runtime = (
        hpc_utils.get_mc_indicator(speed_log, tp, output_file_path['ssm_path'],
                                   runtime, max_time=parsed_args.st))
    delay_res = hpc_utils.get_delay_indicator(output_file_path['tripinfo_path'])
    ramp_entry_count, mrm_insertion_count, ramp_entry_ids = hpc_utils.get_mrm_insertion_counts(
        output_file_path['xml_path'], hpc_utils.DEFAULT_WARMUP_TIME, parsed_args.st)
    mr_ttc_ratio_3, mr_ttc_ratio_2, mr_ttc_ratio_1_5 = hpc_utils.get_mr_ttc_ratios(
        output_file_path['ssm_path'], ramp_entry_ids, max_time=parsed_args.st)

    # 3. Save results
    if parsed_args.out_csv:
        row = {
            "algo": "mpgc_multi_lane",
            'fc_mode': parsed_args.fc_mode,
            "tsg_mode": parsed_args.tsg_mode,
            "av_p": parsed_args.av_p,
            "ramp_demand": parsed_args.r_fr,
            "mainline_demand": parsed_args.m_fr,
            "seed": parsed_args.seed,
            "max_team_size": parsed_args.max_team_size,
            "r_platoon_p": parsed_args.r_platoon_p,
            "loss_rate": parsed_args.loss_rate,
            "CFR": res["cfr"],
            "SPR": res["spr"],
            "over_pltn": res["over_pltn"],
            "std_pltn": res["std_pltn"],
            "sparse_pltn": res["sparse_pltn"],
            "avg_pltn_size": res["avg_pltn_size"],
            "throughput": tp,
            "avg_speed": average_v,
            "ttc_ratio_3": ttc_ratio_3,
            "ttc_ratio_2": ttc_ratio_2,
            "ttc_ratio_1": ttc_ratio_1,
            "mr_ttc_ratio_3": mr_ttc_ratio_3,
            "mr_ttc_ratio_2": mr_ttc_ratio_2,
            "mr_ttc_ratio_1.5": mr_ttc_ratio_1_5,
            "se_cnt": se_result[0],
            "se_reward_sum": se_result[1],
            "se_reward_avg": se_result[2],
            "ce_cnt": ce_result[0],
            "ce_reward_sum": ce_result[1],
            "ce_reward_avg": ce_result[2],
            'avg_time_loss': delay_res["avg_time_loss"],
            'mainline_time_loss': delay_res["mainline_time_loss"],
            'ramp_time_loss': delay_res["ramp_time_loss"],
            'completed_mainline': delay_res["completed_mainline"],
            'completed_ramp': delay_res["completed_ramp"],
            "ramp_entry_count": ramp_entry_count,
            "mrm_insertion_count": mrm_insertion_count,
            "ts_first_jam": ts_first_jam,
            "ts_first_back_to_regular": ts_first_back_to_regular,
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
    start = time.time()
    max_team_size = 12
    st = 1500 # 1500
    (dic_follower_state, his_dic_platoon_size, dic_id_features,
     tp, speed_log, queue_log, output_file_path,
     se_result, ce_result, ts_first_jam, ts_first_back_to_regular) = mpgc_main(
        av_p = 0.1, # 0.1
        r_fr = 1400, # 1300
        m_fr = 1500, # 1500
        seed = 2, # 2 analysis
        r_autoFollow_p = 0,  # auto follow proportion
        r_platoon_p = 0.7, # percentage of rplatoon vehicles on ramp
        loss_rate = 0, # 0.15
        gui = False,
        plot = False,
        display = False,
        lc = True, # if allow HV lane-changing; True
        fc_mode = 'full', # dla_only/dla_tsc/dla_tsc_lhr/dla_tsc_lhr_ce/full
        st = st, # 1200
        tsg_mode = 'predict', # off/fix/predict/train/audit
        max_team_size = max_team_size
    )
    end = time.time()
    runtime = end - start

    print(f'\nse_result: {se_result}, ce_result: {ce_result}')
    hpc_utils.get_fc_detail(dic_follower_state, his_dic_platoon_size, max_size=max_team_size)

    tp, average_v, ttc_ratio_3, ttc_ratio_2, ttc_ratio_1, runtime = (
        hpc_utils.get_mc_indicator(speed_log, tp,
                                   output_file_path['ssm_path'], runtime, max_time=st))
    hpc_utils.get_delay_indicator(output_file_path['tripinfo_path'])
    ramp_entry_count, _, ramp_entry_ids = (
        hpc_utils.get_mrm_insertion_counts(output_file_path['xml_path'],
                                           hpc_utils.DEFAULT_WARMUP_TIME, st))
    mr_ttc_ratio_3, mr_ttc_ratio_2, mr_ttc_ratio_1_5 = hpc_utils.get_mr_ttc_ratios(
        output_file_path['ssm_path'], ramp_entry_ids, max_time=st)
    print(f'ts_first_jam: {ts_first_jam}, ts_first_back_to_regular: {ts_first_back_to_regular}')
