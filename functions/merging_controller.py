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
from functions import dp_jam as dpj
from functions import merging_control_jam as mcj


class MergingController:
    def __init__(self, data_recorder, traci, platoon_formation=False,
                 loss_rate=0, mpc_interval=60, delta_t=15,
                 warmup_time=0, comm_rng=None, comp_logger=None,
                 jam_mode_core='legacy'):  # jam_mode_core: 'legacy' (original Jam_mode) or 'dp' (dynamic programming)
        self.traci = traci
        self.data_recorder = data_recorder
        self.merge_regular = mcr.MergingControlRegular(traci, self.data_recorder,
                                                       comp_logger=comp_logger)
        self.action_mgr = act_mgr.ActionManager(
            self.data_recorder, self.merge_regular, loss_rate, comm_rng)

        # jam control/av-based ramp metering (original core or DP core)
        if jam_mode_core == 'legacy':
            self.av_rm = mcj.MergingControlJam(traci, self.data_recorder, self.merge_regular,
                                                   loss_rate, comm_rng)
        else:
            self.av_rm = dpj.DPJam(traci, self.data_recorder, self.merge_regular,
                                             loss_rate, comm_rng)

        self.mpc_interval = mpc_interval
        self.delta_t = delta_t
        self.warmup_time = warmup_time
        self.ls_r_dep_times = []  # list of ramp veh depature time; [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 58, 59, 60]
        self.pf = platoon_formation
        self.speed_log = []  # [step, avg_speed, True/False], True/False Jam

        self.ts_first_jam = None
        self.ts_first_back_to_regular = None

    def step(self, st, step):
        '''

        :param
            st: total simulation time
            step: simulation step
            m_dpt_type: scripts lane veh departure schedule; {4: 'AHHHHHHHHH', 31: 'AHHHHHHHHHHH'}
            r_dpt_type: ramp lane veh departure schedule; {4: 'AHHHHHHHHH', 58: 'AHHHHHHHHHHH'}
            queue_log: record queue_length; [step, queue_length]

        :return:
        '''
        c_ts = round(step/10 + 0.1, 1)
        self.merge_regular.current_step = step

        # === 1. Unpack veh info ===
        dic_vid_groups = (
            self.data_recorder.dic_vid_groups
            if self.pf
            else self.data_recorder.record_vehinfo())

        ls_veh_id = dic_vid_groups['ls_vehid']
        # below leader lists all are before the MS (merging section)
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']  # min => max

        # === 2. Platoon info (scripts + ramp) ===
        dic_platoon_info = self.merge_regular.get_platoon_info2()

        # get throughput on 'center'
        tp = self.data_recorder.record_throughput(
            st, ls_veh_id, 'center', warmup_time=self.warmup_time)  # throughput

        # Update mainline platoon ET
        pfz_leaders = dic_vid_groups.get('ls_pf_leader_asc', [])
        # Followers of the last MCZ platoon may still precede this leader in PFZ.
        first_pfz_leader = pfz_leaders[0] if pfz_leaders else None
        dic_mplatoon_et, new_leader_flag = self.merge_regular.update_platoon_et(
            step,
            ls_m_leader_up_asc,
            m=True,
            interval=self.mpc_interval,
            first_pfz_leader=first_pfz_leader)

        # === 4. Apply corresponding control logic ===
        queue_log = self.av_rm.jam_control(
            step,
            dic_platoon_info,
            dic_mplatoon_et,
            dic_vid_groups,
            self.ls_r_dep_times,
            self.mpc_interval,
            self.delta_t,
            self.pf)

        # average_velocity of this step and its jam_state
        step_speed = self.data_recorder.get_average_speed(step, ls_veh_id, jam_mode=True)
        self.speed_log.append(step_speed)  # collect into one list

        # If neither regular nor jam_mode, jam_mode is False by default.
        return tp, queue_log, self.ts_first_jam, self.ts_first_back_to_regular
