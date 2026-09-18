# run_lipr_dual_expert_train.py

import os
import time
from pathlib import Path

# for random seed setting
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

def mpgc_main(av_p=0.3, r_fr=0, m_fr=1000, seed=21, r_platoon_p=1,
              loss_rate=0, gui=False, plot=False, display=False,
              lc=False, st=1000, train_agent=None, lr=0.0005,
              train_interval=32, hidden_layer=[64, 64]):
    '''
    SA: splitting agent; CA: collecting agent
    LR: learning rate; batch_epoch: B, E; hidden_layer: HA, HB; seed: S
    '''
    exp_name = f"{train_agent or 'EVAL_ONLY'}_LR{lr}_I{train_interval}_HA{hidden_layer[0]}HB{hidden_layer[1]}_S{seed}"

    set_global_seed(seed, enable=True) # set global random seed (especially for RL training)
    print(f"global random seed: {seed}")

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
    traj_dir = Path(os.environ.get("TRAJ_DIR",
                                   ROOT / "data" / "multi_lane" / "algo"))
    file_name = f'trj_{r_fr}_{av_p}_{seed}_{loss_rate}.xml'
    xml_path = os.path.join(traj_dir, file_name)
    sumo_cmd = [sumo_bin, "-c", str(sumo_config_path),
                # "--fcd-output", str(xml_path), # save path (no need to save xml)
                "--no-warnings"]
    sumo_options = ["--step-length", str(sim_step)]

    formation_controller = None

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

        # Configure isolation training logic
        SA_mode, CA_mode = 'predict', 'predict'
        if train_agent == 'SA':
            SA_mode, CA_mode = 'train', 'off'
        elif train_agent == 'CA':
            SA_mode, CA_mode = 'off', 'train'
        elif train_agent is None: # If None, both agents remain in 'predict'
            pass
        else:
            raise ValueError(f"[Error] Unknown train_model parameter: {train_agent}")

        formation_controller = fc.FormationController(
            data_recorder, traci, sa_mode=SA_mode,
            ca_mode=CA_mode, exp_name=exp_name, learning_rate=lr,
            train_interval=train_interval,
            expert_hidden_dims=hidden_layer)

        dic_follower_state, his_dic_platoon_size, dic_id_features = \
            loop(traci, st, data_recorder, veh_gen, formation_controller, lc,
                 m0_dpt_type, m1_dpt_type)
    finally:
        # Close TensorBoard writers for any active agents
        for attr in ['split_agent', 'collect_agent']:
            handler = getattr(formation_controller, attr, None)
            scoring_agent = getattr(handler, 'agent', None)

            if scoring_agent is not None and hasattr(scoring_agent, 'writer'):
                scoring_agent.writer.close()

        traci.close()
    return (dic_follower_state, his_dic_platoon_size, dic_id_features,
            xml_path)

def loop(traci, st, data_recorder,
         veh_gen, formation_controller, lc,
         m0_dpt_type=None, m1_dpt_type=None):
    # START SIMULATION
    step = 0
    history_cleanup_interval = 30000  # 3000 simulation seconds
    # scripts loop
    while step < st * 10:
        # checkpoint
        if step > 14452 * 10:
             pass
        traci.simulationStep()  # start simulation

        c_ts = traci.simulation.getTime()  # current_timestep
        if c_ts % 1 == 0:
            prc.print_message(f'************current_time, step:{c_ts, step}************')

        # main vehicle generation
        veh_gen.veh_gen_homo(step, m1_dpt_type, 'm', 'route_m', 27.5, '1')  # 30m/s => 110km/h
        veh_gen.veh_gen_homo(step, m0_dpt_type, 'm', 'route_m', 27.5, '0')  # 25m/s => 90km/h; ori 29.5

        # cleaning up the history of platoon size for speed up training process
        if step > 0 and step % history_cleanup_interval == 0:
            active_ids = set(traci.vehicle.getIDList())
            p_basic = formation_controller.p_basic
            p_basic.dic_platoon_size = {
                leader: size
                for leader, size in p_basic.dic_platoon_size.items()
                if leader in active_ids
            }
            p_basic.dic_platoon_members = {
                leader: members
                for leader, members in p_basic.dic_platoon_members.items()
                if leader in active_ids
            }
            data_recorder.ls_m_leader_his_asc = list(
                p_basic.dic_platoon_size.keys()
            )
            print(
                f"[History Cleanup] step={step}, "
                f"active_leaders={len(p_basic.dic_platoon_size)}"
            )

        dic_follower_state, his_dic_platoon_size, dic_id_features = (
            formation_controller.step(st, step, lc))

        data_recorder.record_tail_arrival(step)
        step += 1
    return (dic_follower_state, his_dic_platoon_size, dic_id_features)

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

def main(args=None, root=None):
    """
    Unified entry point for CLI (command line interface) / HPC.
    This function is called by run.py.
    It parses command-line arguments and calls mpgc_main().
    """
    prc.PRINT_ENABLED = False

    # 1. Parse Args (HPC/CLI Mode)
    parser = hpc_utils.training_arg_parser()
    parsed_args = parser.parse_args(args=args)

    # 2. Run simulation
    # Call the original algorithm
    start = time.time()
    _ = mpgc_main(
        av_p=parsed_args.av_p,
        r_fr=parsed_args.r_fr,
        m_fr=parsed_args.m_fr,
        seed=parsed_args.seed,
        gui=parsed_args.gui,
        st=parsed_args.st,
        train_agent=parsed_args.train_agent,
        lr=parsed_args.lr,
        train_interval=parsed_args.train_interval,
        hidden_layer=parsed_args.hidden_layer,
        )
    end = time.time()
    runtime = end - start
    print(runtime)

if __name__ == '__main__':
    prc.PRINT_ENABLED = False
    start = time.time()
    (dic_follower_state, his_dic_platoon_size, dic_id_features,
     xml_path) = mpgc_main(
        av_p = 0.1, # 0.3
        r_fr = 0,
        m_fr = 1000,
        seed = 21, # 1
        gui = True,
        st = 1200*5, # 50; 100
        train_agent = 'CA',
        lr = 0.0001, # 0.0005
        train_interval = 16, # default => train_interval (I): 32; batch_size (I/2): 16; epoch: 5
        hidden_layer = [64, 64] # default: [64, 64]
    )
    end = time.time()
    runtime = end - start

'''
train_agent = ['CA', 'SA']
av_p: if model = CA, av_p = 0.3; if model = SA, av_p = 0.1
lr = [0.0005, 0.0001, 0.001]
batch_epoch = [[16, 5], [32, 2], [64, 1]]
hidden_layer = [[32, 32], [64, 64], [128, 128]]
seed = [21, 22, 23]
'''