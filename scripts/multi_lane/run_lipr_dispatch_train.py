# run_lipr_dispatch_train.py

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
from functions import data_recording as dr
# Shared tools for arguments, KPIs, and CSV logging.// CLI and HPC
from functions import hpc_utils

TSG_TRAIN_SCENARIOS = [
    # av_p = 0.1, 5 runs
    (0.1, 21),
    (0.1, 22),
    (0.1, 23),
    (0.1, 24),
    (0.1, 25),

    # av_p = 0.2, 3 runs
    (0.2, 26),
    (0.2, 27),
    (0.2, 28),

    # av_p = 0.3, 2 runs
    (0.3, 29),
    (0.3, 30),
]

def ensure_local_tsg_run_dir():
    if "RUN_DIR" not in os.environ:
        root = Path(__file__).resolve().parents[2]
        # run_dir = root / "rl_model" / "rl_logs" / "TSG_local_training"
        # run_dir = root / "rl_model" / "rl_logs" / "TSG_single_6d_netreward"
        run_dir = (
                root / "rl_model" / "rl_logs"
                / "TSG_single_6d_reward_weighted"
        )
        os.environ["RUN_DIR"] = str(run_dir)
        print(f"[TSG] RUN_DIR={os.environ['RUN_DIR']}")

def mpgc_main(av_p=0.3, r_fr=0, m_fr=1200, seed=21, r_platoon_p=1,
              loss_rate=0, gui=False, plot=False, display=False,
              lc=False, st=1000, train_model=None,
              lr=0.0005, train_interval=32):
    '''
    SA: splitting agent; CA: collecting agent; TSG: target self-gating
    LR: learning rate; batch_epoch: B, E; seed: S
    '''

    param_tag = f"LR{lr}_I{train_interval}__S{seed}"
    if train_model == 'TSG':
        exp_name = param_tag
    else:
        exp_name = f"{train_model or 'EVAL_ONLY'}_{param_tag}"

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
        data_recorder.max_platoon_size = 12
        data_recorder.get_avhid_ptype(r_dpt_type = r_dpt_type)  # here only have r_dpt_type

        # Configure isolation training logic
        SA_mode, CA_mode, TSG_mode = 'predict', 'predict', 'predict' # 'predict', 'train', None
        if train_model == 'SA':
            SA_mode, CA_mode, TSG_mode = 'train', 'off', 'off'
        elif train_model == 'CA':
            SA_mode, CA_mode, TSG_mode = 'off', 'train', 'off'
        elif train_model == 'TSG': # TSG: target self-gating
            SA_mode, CA_mode, TSG_mode = 'predict', 'predict', 'train'
        elif train_model is None: # If None, both agents remain in 'predict'
            pass # pass
        else:
            raise ValueError(f"[Error] Unknown train_model parameter: {train_model}")
        formation_controller = fc.FormationController(data_recorder, traci, sa_mode=SA_mode,
        ca_mode=CA_mode, tsg_mode=TSG_mode, exp_name=exp_name,
        learning_rate=lr, train_interval=train_interval)  # Passes the unique folder name down

        (dic_follower_state, his_dic_platoon_size,
         dic_id_features) = \
            loop(traci, st, data_recorder, veh_gen, formation_controller, lc,
                 m0_dpt_type, m1_dpt_type)

    finally:
        # Each scenario owns a fresh SUMO session, but TSG training should be
        # continuous across scenarios.
        if ('formation_controller' in locals()
                and formation_controller.tsg_mode == 'train'):

            tsg_manager = formation_controller.tsg_manager
            gate_agent = tsg_manager.gate_agent

            # Train completed transitions left below train_interval before
            # starting the next simulation scenario.
            if gate_agent is not None and gate_agent.memory:
                remaining = len(gate_agent.memory)
                gate_agent.train_on_recorded(
                    current_step=int(st * 10),
                    epochs=5,
                    batch_size=max(1,
                        min(int(train_interval / 2), remaining)))

            # Save the updated model for the next scenario.
            tsg_manager.save_latest()

        traci.close()
    return (dic_follower_state, his_dic_platoon_size,
            dic_id_features, xml_path)

def loop(traci, st, data_recorder,
         veh_gen, formation_controller, lc,
         m0_dpt_type=None, m1_dpt_type=None):
    # START SIMULATION
    step = 0
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
        veh_gen.veh_gen_homo(step, m1_dpt_type, 'm', 'route_m', 27.5, '1')  # 30m/exp_names => 110km/h
        veh_gen.veh_gen_homo(step, m0_dpt_type, 'm', 'route_m', 27.5, '0')  # 25m/s => 90km/h; ori 29.5

        (dic_follower_state, his_dic_platoon_size,
         dic_id_features) = formation_controller.step(st, step, lc)

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

def run_tsg_training_plan(st=1500*10, lr=0.0001, train_interval=16,
                          gui=False):

    start = time.time()

    for idx, (av_p, seed) in enumerate(TSG_TRAIN_SCENARIOS, start=1):
        print(
            f"\n===== TSG training scenario {idx}/{len(TSG_TRAIN_SCENARIOS)}: "
            f"av_p={av_p}, seed={seed}, st={st} ====="
        )

        _ = mpgc_main(
            av_p=av_p,
            r_fr=0,
            m_fr=1500,
            seed=seed,
            r_platoon_p=1,
            loss_rate=0,
            gui=gui,
            plot=False,
            display=False,
            lc=False,
            st=st,
            train_model='TSG',
            lr=lr,
            train_interval=train_interval,
        )

    end = time.time()
    print(f"\nTotal TSG training runtime: {end - start:.1f} s")

def main(args=None, root=None):
    """
    Unified entry point for CLI / HPC.
    Called by run.py.
    Runs the shared TSG multi-scenario training plan.
    """
    prc.PRINT_ENABLED = False

    parser = hpc_utils.training_arg_parser()
    parsed_args = parser.parse_args(args=args)
    ensure_local_tsg_run_dir()
    run_tsg_training_plan(
        st=parsed_args.st,
        lr=parsed_args.lr,
        train_interval=parsed_args.train_interval,
        gui=parsed_args.gui,
    )

if __name__ == '__main__':
    prc.PRINT_ENABLED = False
    start = time.time()
    ensure_local_tsg_run_dir()
    (dic_follower_state, his_dic_platoon_size,
     dic_id_features, xml_path) = mpgc_main(
        av_p = 0.1, # 0.3
        r_fr = 0,
        m_fr = 1000,
        seed = 30, # 29, 1200*60
        gui = False,
        st = 1500*100,  # 50; 100
        train_model = 'TSG', # 'SA', 'CA', 'TSG', None
        lr = 0.0005,  # 0.0005
        train_interval = 16,  # default => train_interval (I): 32; batch_size (I/2): 16; epoch: 5
    )
    end = time.time()
    runtime = end - start
