'''
run_mpgc_multi_lane_collect_RF_state_data.py
this version is for collecting RF train data
multi-lane simulation
test longer platoon
'''

import os
import pandas as pd
from pathlib import Path

from functions import vehicle_generation3 as vg
from functions import print_control as prc # the shared fuction of print control
from functions import formation_controller as fc
from functions import data_recording as dr
from functions import hpc_utils

def mpgc_main(av_p=0.1, r_fr=0, m_fr=1000, seed=0,
              loss_rate=0, gui=False, st=1200):
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
    sumo_home = "/home/zzha/opt/sumo-1.19.0-src"
    os.environ["SUMO_HOME"] = sumo_home
    sumo_bin = os.path.join(sumo_home, "bin", "sumo-gui" if gui else "sumo",)
    sumo_cmd = [sumo_bin, "-c", str(sumo_config_path),
                "--seed", str(seed),
                "--no-warnings"] # , '-S' start auto, and quit auto
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
        av_p0 = av_p
        m0_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p0, m_fr, seed)
        m1_dpt_type = {}

        veh_gen = vg.VehGen(traci, seed) # function related to veh generation
        data_recorder = dr.DataRecording(traci)

        formation_controller = fc.FormationController(data_recorder, traci,
                                                      loss_rate=loss_rate)

        dic_follower_state, his_dic_platoon_size, dic_id_features = \
            loop(traci, st, data_recorder, veh_gen, formation_controller, m0_dpt_type, m1_dpt_type)
    finally:
        traci.close()
    return dic_follower_state, his_dic_platoon_size, dic_id_features

def loop(traci, st, data_recorder, veh_gen,
         formation_controller, m0_dpt_type=None, m1_dpt_type=None, lc=False):
    # START SIMULATION
    step = 0
    # scripts loop
    while step < st*10:
        # checkpoint
        if step > 160*10:
            pass
        if step % 10 == 0:
            pass
        traci.simulationStep()  # start simulation
        # vehicle generation
        # 25m/s => 90km/h; ori 29.5; 30m/s => 110km/h
        veh_gen.veh_gen_homo(step, m0_dpt_type, 'm', 'route_m', 27.5, '0', hv_type='hv_mean')
        veh_gen.veh_gen_homo(step, m1_dpt_type, 'm', 'route_m', 27.5, '1', hv_type='hv_mean')
        # veh_gen.veh_gen_hetero(step, m0_dpt_type, 'm', 'route_m', 27.5, '0')
        # veh_gen.veh_gen_hetero(step, m1_dpt_type, 'm', 'route_m', 27.5, '1')

        (dic_follower_state, his_dic_platoon_size,
         dic_id_features) = formation_controller.step(st, step, lc, rf_model=False)
        step += 1
    return dic_follower_state, his_dic_platoon_size, dic_id_features

def main(args=None, root=None):
    prc.PRINT_ENABLED = False
    parser = hpc_utils.standard_arg_parser()
    parsed_args = parser.parse_args(args=args)

    dic_follower_state, his_dic_platoon_size, dic_id_features = mpgc_main(
        av_p=parsed_args.av_p,
        r_fr=parsed_args.r_fr,
        m_fr=parsed_args.m_fr,
        seed=parsed_args.seed,
        gui=parsed_args.gui,
        st=parsed_args.st)
    df_fea_tar = organise_data(dic_follower_state, dic_id_features)
    print(len(dic_follower_state))
    print(len(dic_id_features))
    print(len(df_fea_tar))
    hpc_utils.write_dataframe_csv(parsed_args.out_csv, df_fea_tar)

def organise_data(dic_follower_state, dic_id_features):
    '''
    Record features and targets (final states)
    Parameters
        dic_follower_state: {follower: [state, leader],..., }
        dic_id_features: {follower: [f1, f2, f3, f4, leader],...}
    Returns
        df_fea_tar: dataframe

    '''
    feature_target = []
    for follower, features in dic_id_features.items():
        state, leader = dic_follower_state.get(follower, [None, None])
        row = [follower] + features + [state, leader]
        feature_target.append(row)
    columns = ['follower_id', 'v_leader', 'dis_leader_to_mcz', 'n_veh_between',
               'time_headway_to_leader', 'state', 'leader_id']
    df_fea_tar = pd.DataFrame(feature_target, columns=columns)
    return df_fea_tar


if __name__ == '__main__':
    prc.PRINT_ENABLED = False
    dic_follower_state, his_dic_platoon_size, dic_id_features = mpgc_main(
        av_p = 0.1,
        r_fr = 0,
        m_fr = 1000,
        seed = 5,
        gui = True,
        st = 600)

    # record features and targets (final states)
    df_fea_tar = organise_data(dic_follower_state, dic_id_features)
    df_fea_tar.to_csv("/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_state_test.csv", index=False)

