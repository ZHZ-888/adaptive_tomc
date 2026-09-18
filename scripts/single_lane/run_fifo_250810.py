# 250810: add HV heter
# no control strategy, following first in first out principle
import pandas as pd
import os
import time

from functions import vehicle_generation3 as vg
from functions import data_recording as dr
from functions import calc_ttc_sd_exposure


def main(av_p, r_fr, m_fr, seed, gui=False, plot=False, st=1000):
    # SUMO SETTING
    # path = '/home/zzha/PycharmProjects/RoadNetwork/merge_rode12_shapeChanged.sumocfg'
    # path = '/home/zzha/PycharmProjects/RoadNetwork/single_lane_merge/merge_road_250621_shapeChanged.sumocfg'
    # path = 'road_network/single_lane_merge/merge_road_250811_fifo.sumocfg'
    path = '../../road_network/single_lane_merge/cfg_merge_fifo.sumocfg'
    sumo_config_path = path
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    traj_dir = '../../data/fifo/traj_fifo'
    file_name = f'trj_{r_fr}_{av_p}_{seed}.xml'
    xml_path = os.path.join(traj_dir, file_name)
    # Construct the SUMO command and options
    sumo_cmd = [sumo_bin, "-c", sumo_config_path,
                "--fcd-output", xml_path,
                "--no-warnings"] # "data/Traj/trajectory_fifo_test.xml"
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
        # scripts road veh depature schedule
        max_attempts = 8
        m_dpt_type = vg.generate_depature_time(st, m_fr, seed)
        # ramp road veh depature schedule
        r_dpt_type = vg.generate_depature_time(st, r_fr, seed)
        drdr = dr.DataRecording(traci, sim_step)
        veh_gen = vg.VehGen(traci) # function related to veh generation
        # get dic_avhid_ptype
        speed_log, tp = loop(traci, st, veh_gen, drdr, m_dpt_type, r_dpt_type)
    finally:
        traci.close() # Ensure SUMO simulation is properly closed to release memory and system resources
    return speed_log, tp, xml_path


def loop(traci, st, veh_gen, drdr, m_dpt_type=None, r_dpt_type=None):
    # START SIMULATION
    step = 0
    speed_log = []
    # edge_id, inflow_merge, center
    ramp_entry_edge = 'inflow_merge'
    ramp_exit_edge = 'center'
    # scripts loop
    while step < st*10:
        if step == 200:
            pass
        traci.simulationStep()  # start simulation
        c_ts = traci.simulation.getTime()  # current_timestep
        # vehicle generation
        veh_gen.veh_gen_hv2(step, m_dpt_type, 'm', 'route1', 24.5)
        veh_gen.veh_gen_hv2(step, r_dpt_type, 'r', 'route2', 10)

        # performance indicator
        dic_vehinfo = drdr.record_vehinfo()
        ls_veh_id = dic_vehinfo['ls_vehid']
        step_speed = drdr.get_average_speed(step, ls_veh_id)
        # 1. speed_record
        speed_log.append(step_speed)
        # 2. throughput
        tp = drdr.record_throughput(st, ls_veh_id, 'center')  # throughput

        step += 1
    return speed_log, tp


if __name__ == '__main__':
    st = 1200
    av_p = 0
    r_fr = 810
    m_fr = 1080
    seed = 1
    gui = True
    plot = False
    start = time.time()
    speed_log, tp, xml_path = main(av_p, r_fr, m_fr, seed, gui, plot, st)
    end = time.time()

    ttc_ratio, avg_speed_std = calc_ttc_sd_exposure.calc_ttc_and_speed_std(xml_path)
    avg_speeds = [item[1] for item in speed_log]
    clean_avg_speeds = [v for v in avg_speeds if v is not None]
    average_v = sum(clean_avg_speeds) / len(clean_avg_speeds)
    print(f'tp: {tp} veh/h, average_v:{average_v} m/s, '
          f'ttc_ratio: {ttc_ratio}, avg_speed_std: {avg_speed_std}, '
          f'execution_time:{end - start:.1f} s')

