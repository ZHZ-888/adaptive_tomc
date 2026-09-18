import os
import time

from functions import vehicle_generation3 as vg
from functions import data_recording as dr
from functions import print_control as prc # the shared function of print control
from functions import calc_ttc_sd_exposure

from comparsion_algo import ramp_metering_algo2 as rm

def main(av_p, r_fr, m_fr, seed, gui=False, plot=False, st=1000):
    # SUMO SETTING
    # path = '/home/zzha/PycharmProjects/RampMerging4_250208/road_network/merge_rode13_rampmeter_acc.sumocfg'
    # path = 'road_network/single_lane_merge/merge_road13_rm_acc.sumocfg'
    path = '../../road_network/single_lane_merge/cfg_merge_rm.sumocfg'
    sumo_config_path = path
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    traj_dir = '../../data/rm/traj_rm'
    file_name = f'trj_{r_fr}_{av_p}_{seed}.xml'
    xml_path = os.path.join(traj_dir, file_name)
    # Construct the SUMO command and options
    sumo_cmd = [sumo_bin, "-c", sumo_config_path,
                "--fcd-output", xml_path,
                "--no-warnings"]
    sumo_options = ["--step-length", str(sim_step)]
    # If GUI is enabled, set the GUIcomparsion_algo' view schema
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
        # generate only hv
        ls_m_dpt = vg.generate_depature_time(st, m_fr, seed)
        ls_r_dpt = vg.generate_depature_time(st, r_fr, seed)
        veh_gen = vg.VehGen(traci) # function related to veh generation
        rm_algo = rm.Func(traci) # function related to ramp_metering algo
        data_recorder = dr.DataRecording(traci, sim_step)
        speed_log, tp = loop(traci, st, veh_gen, data_recorder, rm_algo, ls_m_dpt, ls_r_dpt)
    finally:
        traci.close()
    return speed_log, tp, xml_path


def loop(traci, st, veh_gen, data_recorder, rm_algo, ls_m_dpt=None, ls_r_dpt=None):
    # START SIMULATION
    step = 0
    speed_log = []
    # edge_id, inflow_merge, center
    ramp_entry_edge = 'inflow_merge'
    ramp_exit_edge = 'center'
    # rm related parameters
    target_density = 25 # veh_num/km
    gain = 20
    p_release_rate = 200 # veh_num/h

    accident_state = False
    # scripts loop
    while step < st*10:
        if step == 200:
            pass
        traci.simulationStep()  # start simulation
        prc.print_message("\n**************")
        c_ts = traci.simulation.getTime()  # current_timestep
        prc.print_message(f'current_time, step:{c_ts, step}')
        # only hv
        veh_gen.veh_gen_hv2(step, ls_m_dpt, 'm', 'route1', 24.5)
        veh_gen.veh_gen_hv2(step, ls_r_dpt, 'r', 'route2', 10)

        # get veh info
        dic_vehinfo = data_recorder.record_vehinfo()
        ls_veh_id = dic_vehinfo['ls_vehid']
        # dic_id_speed = data_recorder.record_vehSpeed(ls_vehid)
        step_speed = data_recorder.get_average_speed(step, ls_veh_id)
        # 1. speed_record
        speed_log.append(step_speed)

        # get current density
        main_id = 'inflow_highway_0'
        ramp_id = 'inflow_merge_0'
        # 2. throughput
        tp = data_recorder.record_throughput(st, ls_veh_id, 'center')  # throughput

        # simulate a sudden accident
        # accident_state = ac.sudden_accident(traci, ls_veh_id, 200, 100, accident_state)

        if c_ts % 90 == 0:
            des_center = rm_algo.get_current_density("center_0") # center_0; current center flow
            des_ramp = rm_algo.get_current_density("inflow_merge_0")

            c_release_rate = rm_algo.alinea_control(des_center, target_density, gain, p_release_rate)
            prc.print_message(f'c_release_rate: {c_release_rate}')

            # set the green time of the signal
            g_time, r_time = rm_algo.set_ramp_metering_signal('center', c_release_rate)

            # update last release_rate
            p_release_rate = c_release_rate  # p-previous, c-current
        try:
            prc.print_message(f'g_time: {g_time}, r_time:{r_time}')
            prc.print_message(f'center:{des_center},main:{des_ramp}')
        except:
            pass

        step += 1
    return speed_log, tp


if __name__ == '__main__':
    st = 1200
    av_p = 0
    r_fr = 720
    m_fr = 1080
    seed = 1

    prc.PRINT_ENABLED = True
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
