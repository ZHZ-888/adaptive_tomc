'''
run_rm_multi_lane.py
multi-lane simulation
'''
import os
import time
from datetime import datetime
from pathlib import Path

from functions import vehicle_generation3 as vg
from functions import data_recording as dr
from functions import print_control as prc  # the shared function of print control
# Shared tools for arguments, KPIs, and CSV logging.// CLI and HPC
from functions import hpc_utils
from functions import merging_traffic_calibrator as mtc

from comparsion_algo import ramp_metering_algo2 as rm

def rm_main(av_p, r_fr, m_fr, seed,
         gui=False, plot=False, display=False, st=1500):
    """
    Main simulation logic
    """
    # Project root directory
    ROOT = Path(__file__).resolve().parents[2]
    # SUMO SETTING
    sumo_config_path = (
            ROOT
            / "road_network"
            / "multi_lane_motorway"
            / "real"
            / "cfg_multi_lane_merge_rm.sumocfg"
    )
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    traj_dir = Path(os.environ.get(
                    "TRAJ_DIR",
                    ROOT / "data" / "multi_lane" / "algo"
                    ))  # default 'data/mpgc'
    file_name = f"trj_rm_{r_fr}_{av_p}_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
    xml_path = os.path.join(traj_dir, file_name)

    lc_file_name = f"lc_rm_{r_fr}_{av_p}_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
    lc_path = os.path.join(traj_dir, lc_file_name)

    ssm_file_name = f"ssm_rm_{r_fr}_{av_p}_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
    ssm_path = traj_dir / ssm_file_name
    # delay indicators
    trip_file_name = f"tripinfo_{r_fr}_{av_p}_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
    tripinfo_path = os.path.join(traj_dir, trip_file_name)
    rm_debug_file_name = f"rm_debug_{r_fr}_{av_p}_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rm_debug_path = os.path.join(traj_dir, rm_debug_file_name)
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
                "--device.ssm.thresholds", "3",
                "--device.ssm.trajectories", "true",
                "--device.ssm.write-lane-positions", "true",
                "--no-warnings"]  # , '-S' start auto, and quit auto
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
        av_p0, av_p1 = av_p, av_p
        m0_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p0, m_fr, seed)
        m1_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p1, m_fr, 100 - seed)
        r_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p, r_fr, seed)

        veh_gen = vg.VehGen(traci, seed)  # function related to veh generation
        rm_algo = rm.Func(traci)  # function related to ramp_metering algo
        data_recorder = dr.DataRecording(traci, sim_step)
        output_file_path = {'xml_path': xml_path, 'ssm_path': ssm_path, 'tripinfo_path': tripinfo_path}
        speed_log, tp = \
            loop(traci, st, veh_gen, data_recorder, rm_algo, rm_debug_path,
                 m0_dpt_type, m1_dpt_type, r_dpt_type)
    finally:
        traci.close()
    return (speed_log, tp, output_file_path)


def loop(traci, st, veh_gen,
         data_recorder, rm_algo, rm_debug_path,
         m0_dpt_type=None, m1_dpt_type=None, r_dpt_type=None):
    # START SIMULATION
    step = 0
    horizon_steps = st * 10
    traffic_calibrator = mtc.MergingTrafficCalibrator(traci)
    speed_log = []
    occupancy_sum = 0
    occupancy_samples = 0

    with open(rm_debug_path, "w") as f:
        f.write(
            "time,center_occupancy,"
            "release_rate,green_time,red_time\n"
        )

    # scripts loop
    while step < horizon_steps:
        # checkpoint
        traci.simulationStep()  # start simulation
        c_ts = traci.simulation.getTime()  # current_timestep

        if c_ts % 1 == 0:
            prc.print_message(f'************current_time, step:{c_ts, step}************')

        # mainlane vehicle generation
        veh_gen.veh_gen_hetero(step, m1_dpt_type, 'm', 'route_m', 27.5, '1')  # 30m/s => 110km/h
        veh_gen.veh_gen_hetero(step, m0_dpt_type, 'm', 'route_m', 27.5, '0')  # 25m/s => 90km/h; ori 29.5
        # ramp vehicle generation
        veh_gen.veh_gen_hetero(step, r_dpt_type, 'r', 'route_r', 10, '0')

        traffic_calibrator.update()

        # performance indicator
        dic_vehinfo = data_recorder.record_vehinfo()
        ls_veh_id = dic_vehinfo['ls_vehid']
        step_speed = data_recorder.get_average_speed(step, ls_veh_id)
        # 1. speed_record
        speed_log.append(step_speed)
        # 2. throughput
        tp = data_recorder.record_throughput(
            st, ls_veh_id, 'center',
            warmup_time=hpc_utils.DEFAULT_WARMUP_TIME)  # throughput

        # occupancy_sum += rm_algo.get_current_density("center_0")
        occupancy_sum += rm_algo.get_detector_occupancy("rm_occ_detector")
        occupancy_samples += 1

        if c_ts % 60 == 0: # 90
            des_center = occupancy_sum / occupancy_samples
            des_ramp = rm_algo.get_current_density("inflow_merge_0")

            c_release_rate = rm_algo.alinea_control(des_center)
            # prc.print_message(f'c_release_rate: {c_release_rate}')

            g_time, r_time = rm_algo.set_ramp_metering_signal('center', c_release_rate)

            with open(rm_debug_path, "a") as f:
                f.write(
                    f"{c_ts},"
                    f"{des_center:.4f},"
                    # f"{des_ramp:.4f},"
                    f"{c_release_rate:.2f},"
                    f"{g_time:.2f},"
                    f"{r_time:.2f}\n"
                )

            # set the green time of the signal
            occupancy_sum = 0
            occupancy_samples = 0

        try:
            prc.print_message(f'g_time: {g_time}, r_time:{r_time}')
            prc.print_message(f'center:{des_center},main:{des_ramp}')
        except:
            pass

        step += 1

    # Flush-out stage: stop generating vehicles and let all generated trips finish.
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        traffic_calibrator.update()
        step += 1

    return (speed_log, tp)

def main(args=None, root=None):
    """
    Unified entry point for CLI (command line interface) / HPC.
    This function is called by run.py.
    It parses command-line arguments and calls fifo_main().
    """

    # Algorithm parameters
    # fixed params
    prc.PRINT_ENABLED = False

    # 1. Parse Args (HPC/CLI Mode)
    parser = hpc_utils.standard_arg_parser()
    # Parse arguments passed from run.py
    parsed_args = parser.parse_args(args=args)

    # 2. Run simulation
    start = time.time()
    # Call the original algorithm
    speed_log, tp, output_file_path = rm_main(
        av_p=parsed_args.av_p,
        r_fr=parsed_args.r_fr,
        m_fr=parsed_args.m_fr,
        seed=parsed_args.seed,
        gui=parsed_args.gui,
        st=parsed_args.st
    )
    end = time.time()
    runtime = end - start

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
            "algo": "rm_multi_lane",
            "av_p": parsed_args.av_p,
            "ramp_demand": parsed_args.r_fr,
            "mainline_demand": parsed_args.m_fr,
            "seed": parsed_args.seed,
            "throughput": tp,
            "avg_speed": average_v,
            "ttc_ratio_3": ttc_ratio_3,
            "ttc_ratio_2": ttc_ratio_2,
            "ttc_ratio_1": ttc_ratio_1,
            "mr_ttc_ratio_3": mr_ttc_ratio_3,
            "mr_ttc_ratio_2": mr_ttc_ratio_2,
            "mr_ttc_ratio_1.5": mr_ttc_ratio_1_5,
            'avg_time_loss': delay_res["avg_time_loss"],
            'mainline_time_loss': delay_res["mainline_time_loss"],
            'ramp_time_loss': delay_res["ramp_time_loss"],
            'completed_mainline': delay_res["completed_mainline"],
            'completed_ramp': delay_res["completed_ramp"],
            "ramp_entry_count": ramp_entry_count,
            "mrm_insertion_count": mrm_insertion_count,
            "runtime": runtime
        }
        hpc_utils.write_one_row_csv(parsed_args.out_csv, row)

if __name__ == '__main__':
    print(">>> Running Local RM Test (Direct Function Call)...")
    prc.PRINT_ENABLED = False
    start = time.time()
    st = 1500
    speed_log, tp, output_file_path = rm_main(
        av_p = 0,
        r_fr = 800,
        m_fr = 1500,
        seed = 10,
        gui = True,
        plot = False,
        display = False,
        st = st
    )
    end = time.time()
    runtime = end - start
    hpc_utils.get_mc_indicator(speed_log, tp, output_file_path['ssm_path'], runtime, max_time=st)
    hpc_utils.get_delay_indicator(output_file_path['tripinfo_path'])
    hpc_utils.get_mrm_insertion_counts(output_file_path['xml_path'], hpc_utils.DEFAULT_WARMUP_TIME, st)
