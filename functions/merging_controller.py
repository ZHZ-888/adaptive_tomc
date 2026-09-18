# merging_controller.py
"""
High-level controller:
Combine regular / jam mode logic and three control modules into one place.

This class does NOT change any algorithm or behavior.
It only wraps existing logic in a cleaner interface.
"""

from functions import print_control as prc
from functions import action_manager as act_mgr
from functions import merging_control_regular as mcr
from functions import merging_control_jam as mcj


class MergingController:
    def __init__(self, data_recorder, traci, av_p, platoon_formation=False, ml=False,
                 loss_rate=0, mpc_interval=60, delta_t=15, warmup_time=0,
                 comm_rng=None, comp_logger=None):  # ml: multi-lane
        self.traci = traci
        self.data_recorder = data_recorder
        self.merge_regular = mcr.MergingControlRegular(traci, self.data_recorder, ml,
                                                       comp_logger=comp_logger)
        self.action_mgr = act_mgr.ActionManager(
            self.data_recorder, self.merge_regular, loss_rate, comm_rng)
        self.merge_jam = mcj.MergingControlJam(traci, self.data_recorder, self.merge_regular, loss_rate,
                                               ml, comm_rng)  # vehicle control during_jame3
        self.mode_switch = mcj.ShiftMode(traci, self.data_recorder, av_p)
        self.mpc_interval = mpc_interval
        self.delta_t = delta_t
        self.warmup_time = warmup_time
        self.ls_r_dep_times = []  # list of ramp veh depature time; [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 58, 59, 60]
        self.pf = platoon_formation
        self.speed_log = []  # [step, avg_speed, True/False], True/False Jam

        self.ts_first_jam = None
        self.ts_first_back_to_regular = None

    def step(self, st, step, r_dpt_type):
        '''

        :param st: total simulation time
        :param step: simulation step
        :param m_dpt_type: scripts lane veh departure schedule; {4: 'AHHHHHHHHH', 31: 'AHHHHHHHHHHH'}
        :param r_dpt_type: ramp lane veh departure schedule; {4: 'AHHHHHHHHH', 58: 'AHHHHHHHHHHH'}
        :param queue_log: record queue_length; [step, queue_length]

        :return:
        '''
        c_ts = round(step/10 + 0.1, 1)
        self.merge_regular.current_step = step

        if not self.ls_r_dep_times:
            for key, value in r_dpt_type.items():
                self.ls_r_dep_times.extend(range(key, key + len(value)))

        # === 1. Unpack veh info ===
        dic_vid_groups = (
            self.data_recorder.dic_vid_groups
            if self.pf
            else self.data_recorder.record_vehinfo()
        )

        ls_veh_id = dic_vid_groups['ls_vehid']
        ls_r_veh_net_asc = dic_vid_groups['ls_r_veh_net_asc']
        ls_r_veh_net_last_asc = dic_vid_groups['ls_r_veh_net_last_asc']
        # below leader lists all are before the MS (merging section)
        ls_r_leader_up = dic_vid_groups['ls_r_leader_up']
        ls_r_leader_up_asc = dic_vid_groups['ls_r_leader_up_asc']  # min => max
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']  # min => max

        ls_m_veh_up_asc = dic_vid_groups['ls_m_veh_up_asc']
        ls_wsA_asc = dic_vid_groups['ls_wsA_asc']
        ls_wsB_av_asc = dic_vid_groups['ls_wsB_av_asc']
        # ramp leader on MS_0
        ls_r_leader_wsA_asc = dic_vid_groups['ls_r_leader_wsA_asc']
        ls_wsB_asc = dic_vid_groups['ls_wsB_asc']


        # === 2. Platoon info (scripts + ramp) ===
        dic_platoon_info = self.merge_regular.get_platoon_info2()

        # get throughput on 'center'
        tp = self.data_recorder.record_throughput(
            st, ls_veh_id, 'center', warmup_time=self.warmup_time)  # throughput

        # Update mainline platoon ET
        dic_mplatoon_et, new_leader_flag = self.merge_regular.update_platoon_et(
            step,
            ls_m_leader_up_asc,
            m=True,
            interval=self.mpc_interval)

        # === 3. Determine mode: regular / jam ===
        # regular_mode, jam_mode = self.mode_switch.determine_mode4(
        #     ls_m_veh_up_asc,
        #     ls_r_veh_up,
        #     ls_r_leader_up
        # )

        # regular_mode, jam_mode = self.mode_switch.determine_mode_flexible_merge_point(
        #     ls_wsB_asc,
        #     ls_wsA_asc,
        #     ls_r_leader_up)

        regular_mode, jam_mode = self.mode_switch.determine_mode_low_sensor_reliance(
            ls_r_leader_wsA_asc,
            ls_wsB_av_asc)

        # jam_mode = True
        # regular_mode = False
        # self.merge_regular.set_veh_color()

        # === 4. Apply corresponding control logic ===
        if jam_mode:
            if self.ts_first_jam is None:
                self.ts_first_jam = c_ts
            # Jam mode control
            queue_log = self.merge_jam.jam_control(
                step,
                dic_platoon_info,
                ls_m_leader_up_asc,
                ls_m_veh_up_asc,
                dic_mplatoon_et,
                dic_vid_groups,
                self.ls_r_dep_times,
                self.mpc_interval,
                self.delta_t,
                self.pf
            )  # queue_length

        elif regular_mode:
            if self.ts_first_jam and self.ts_first_back_to_regular is None:
                self.ts_first_back_to_regular = c_ts

            # reset jam_mode params
            self.reset_jam_mode()

            # Regular mode control
            queue_log = []
            _, new_leader_flag = self.merge_regular.update_platoon_et(step, ls_r_leader_up_asc, m=False,
                                                     interval=self.mpc_interval) # get ramp platoon ET

            if c_ts % 1 == 0: prc.print_message('**in regular mode**')
            # find head rav; if can be ingored
            _ = self.merge_regular.find_r_leader(ls_r_veh_net_asc, ls_r_veh_net_last_asc)
            # get action information
            (dic_leader_action, dic_m_leader_followup_action) = self.action_mgr.get_action_info(
                step, new_leader_flag, interval=self.mpc_interval)

            # execute actions
            self.action_mgr.execute_action(step, dic_leader_action)
            self.action_mgr.execute_action(step, dic_m_leader_followup_action)

        # average_velocity of this step and its jam_state
        step_speed = self.data_recorder.get_average_speed(step, ls_veh_id, jam_mode)
        self.speed_log.append(step_speed)  # collect into one list

        # If neither regular nor jam_mode, jam_mode is False by default.
        return tp, queue_log, self.ts_first_jam, self.ts_first_back_to_regular


    def reset_jam_mode(self):
        # for cooldown function: reset jam_mode parameters
        if not self.merge_jam.r_leader_stop:
            return

        vid = self.merge_jam.r_leader_stop
        if vid in self.traci.vehicle.getIDList():
            try:
                if self.traci.vehicle.getStopState(vid) == 0:
                    # Pending stop: resume() is invalid, remove the stop.
                    self.traci.vehicle.replaceStop(vid, 0, "")
                    self.traci.vehicle.moveTo(vid, "ws_0", 5.0) # bugs bugs go die!!! fuck
                    self.traci.vehicle.setSpeed(vid, -1)
                else:
                    # reached stop: resume the stopped vehicle.
                    self.traci.vehicle.resume(vid)
                    self.traci.vehicle.setSpeed(vid, -1)
            except Exception as exc:
                print(f"Failed to release ramp leader {vid}: {exc}")
                return

        self.merge_jam.r_leader_stop = None
        self.merge_jam.stop_state = False
        self.merge_jam.r_leader_stop = None
        self.merge_jam.jam_mode_start_ts = None
        self.merge_jam.first_ramp_stop_ts = None
        self.merge_jam.stop_times = {}
