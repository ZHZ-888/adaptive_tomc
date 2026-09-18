'''
use merging_controller.py to contain all merging controller functions
250616 add non-platoon HV, possion distribution platoon generation
250604 add HV heterogeneity
'''

import os
import time
import pandas as pd

from functions import vehicle_generation3 as vg
from functions import merging_controller as mc

from functions import print_control as prc # the shared function of print control
from functions import calc_ttc_sd_exposure
from functions import data_recording as dr

def main(av_p, r_fr, m_fr, seed, mpc_interval=70, delta_t=12, p_autoFollow=1, platoon_p=1, loss_rate=0,
         gui=False, plot=False, display=False, st=1000):
    # SUMO SETTING
    # 250606 parameter updates (AV: tau=0.5, minGap=1.5; HV(default): tau=1, minGap=2.5)
    # path = 'road_network/single_lane_merge/cfg_merge_100acc.sumocfg'
    path = '../../road_network/single_lane_merge/cfg_merge.sumocfg'
    sumo_config_path = path
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    # traj_dir = 'data/mpgc'
    traj_dir = os.environ.get("TRAJ_DIR", "../../data/mpgc") # default 'data/mpgc'
    file_name = f'trj_{r_fr}_{av_p}_{seed}_{mpc_interval}_{p_autoFollow}_{platoon_p}_{loss_rate}.xml'
    xml_path = os.path.join(traj_dir, file_name)
    sumo_cmd = [sumo_bin, "-c", sumo_config_path,
                "--fcd-output", xml_path, # save path
                "--no-warnings"]
    sumo_options = ["--step-length", str(sim_step)]
    # If GUI is enabled, set the GUI view schema
    if gui:
        import traci
        traci.start(sumo_cmd + sumo_options)
        # traci.gui.setSchema('View #0', "real world")
    else:
        import libsumo as traci
        traci.start(sumo_cmd + sumo_options)
    try:
        # VEHICLE GENERATOR
        # generate departure sequence and corresponding fleet types
        data_recorder = dr.DataRecording(traci)
        max_attempts = 7
        # scripts road veh depature schedule
        m_dpt_type = vg.get_schedule2(st, av_p, m_fr, platoon_p, max_attempts, plot, seed, display)
        # ramp road veh depature schedule
        r_dpt_type = vg.get_schedule2(st, av_p, r_fr, platoon_p, max_attempts, plot, seed, display)
        vgvg = vg.VehGen(traci) # function related to veh generation
        data_recorder.get_avhid_ptype(m_dpt_type, r_dpt_type)
        merging_controller = mc.MergingController(data_recorder, traci, av_p, loss_rate=loss_rate)
        tp, speed_log, queue_log = (
            loop(traci, st, vgvg, merging_controller,
                 p_autoFollow, m_dpt_type, r_dpt_type))

        df_vsj = pd.DataFrame(speed_log, columns=['step', 'speed', 'jam_mode'])  # vsj => velocity, step, jam_state
    finally:
        traci.close() # Ensure SUMO simulation is properly closed to release memory and system resources
    return tp, speed_log, queue_log, xml_path

def loop(traci, st, vgvg, merging_controller, p_autoFollow, m_dpt_type=None, r_dpt_type=None):
    # START SIMULATION
    step = 0
    # scripts loop

    while step < st*10:
        if step > 69:
            pass
        if step >= 777 * 10:
            pass
        if step%10 == 0:
            pass

        traci.simulationStep()  # start simulation
        c_ts = traci.simulation.getTime()  # current_timestep
        if c_ts % 1 == 0:
            prc.print_message(f'************current_time, step:{c_ts, step}************')

        # 100324update: new ramp flow rate after 350
        change_t = 500
        dynamic = False
        if step == change_t*10 and dynamic:
            av_p = 0.1
            new_fr = 180
            seed = 1
            new_r_dpt_type = vg.get_schedule_dynamic(st, change_t, av_p, new_fr, platoon_p=1,
                                                     max_attempts=1, seed=seed) # after start_t
            r_dpt_type = {key: value for key, value in r_dpt_type.items() if key <= change_t} # before start_t
            r_dpt_type.update(new_r_dpt_type) # merge together

        vgvg.platoon_gen(step, m_dpt_type, 'm', p_autoFollow)
        vgvg.platoon_gen(step, r_dpt_type, 'r', p_autoFollow)

        tp, speed_log, queue_log \
            = merging_controller.step(st, step, m_dpt_type, r_dpt_type)

        step += 1
    return tp, speed_log, queue_log

if __name__ == '__main__':
    prc.PRINT_ENABLED = False
    start = time.time()
    tp, speed_log, queue_log, xml_path = (
        main(
            av_p = 0.1,
            r_fr = 720, # 990, 540, 360
            m_fr = 1080,
            seed = 1, # 4
            mpc_interval = 70, # step; 10step = 1s
            delta_t = 12,
            p_autoFollow = 1, # auto follow proportion (percentage = proportion × 100%); 0.8 ;default 1
            platoon_p = 1, # percentage of platoon vehicles; 0.8; default 1
            loss_rate = 0, # 0 (default), 0.05, 0.1, 0.15; packet loss rate
            gui = True,
            plot = False,
            display = False,
            st = 1200
        ))
    end = time.time()

    ttc_ratio, avg_speed_std = calc_ttc_sd_exposure.calc_ttc_and_speed_std(xml_path)

    avg_speeds = [item[1] for item in speed_log]
    clean_avg_speeds = [v for v in avg_speeds if v is not None]
    average_v = sum(clean_avg_speeds) / len(clean_avg_speeds)
    avg = (sum(q for _, q in queue_log) / len(queue_log)) if queue_log else None
    print(f'tp: {tp} veh/h, average_v:{average_v} m/s, queue_length_avg:{avg}, '
          f'ttc_ratio: {ttc_ratio}, avg_speed_std: {avg_speed_std}, '
          f'execution_time:{end - start:.1f} s')

