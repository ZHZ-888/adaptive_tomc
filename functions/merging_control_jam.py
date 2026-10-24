# merging_control_jam.py;
# ORIGINAL core for AV-based ramp metering

from functions import print_control as prc
from functions import v2x_disturbance as v2x

class MergingControlJam:
    def __init__(self, traci, instance_dr, merge_regular, loss_rate, comm_rng=None):
        self.traci = traci
        self.data_recorder = instance_dr
        self.merge_regular = merge_regular

        self.dic_id_speed = self.data_recorder.dic_speed
        self.timing = False
        self.stop_state = False
        self.resume_state = False
        self.r_leader_stop = None  # current stop r leader id; only one stop at the front is enough
        self.m_leader_acting = False

        self.dic_vid_groups = {}
        self.first_stop_recorded = [] # ramp av only stop once
        self.first_resume_recorded = []  # ramp av only resume once
        self.stop_times = {}
        self.resume_times = {}
        self.rp_type_resume_completed = {} # ramp platoon, format: {leader_id: [type, resume_time, tail_completed_time]}

        self.ls_skip_stop = [] # those r_leader can pass with it's preceding r_leader platoon doesn't need to stop
        self.m_leader_action_dic = {} # the action dic of m_leader
        self.dic_desire_reach_ts = {} # drt => desire reaching timestamp
        # m_leader and its action parameters; self.dic_mavh_actionP => self.dic_m_leader_action_params
        self.dic_m_leader_action_params = {'':[]}
        self.ls_r_proper = [] # list of ramp veh before acc
        self.dic_max_interval = {} # {m_leader: max_interval}
        self.dic_mainline_gaps = {} # {m_leader: gap_seconds}, None for the rear window

        self.loss_rate = loss_rate
        self.action_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate, rng=comm_rng)
        self.timing_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate, rng=comm_rng)
        self.last_action_payload = None
        self.last_timing_payload = None

        self.dic_mplatoon_et = {}  # dic_mplatoon_et: {m_leader:[platoon_type, ts_head, ts_tail, c_ts]}
        self.length_mcz = self.data_recorder.length_mcz
        self.max_speed = self.data_recorder.max_speed
        self.last_action_params = None
        self.m_action_params = None

        self.jam_mode_start_ts = None
        self.first_ramp_stop_ts = None
        self.delta_t = None
        self.last_m_leader = None

        self.cooldown_dur = 15 # 30
        self.buffer = 1.5 # platoon-to-platoon time buffer, 1.5 s
        self.stop_pos = 120
        self.speed_level3 = 25
        # the time needed for ramp AV leader moving from stop point to the merging section (weaving section)
        self.r_leader_acc_dur = 11.5 # 9.3; 11.5
        # Mapping from platoon size to total merge completion time (from leader start to tail completing merging)
        # after 13 may not that accurate, but only in case max interval is very large, then allow the ramp combined number exceeds 12
        # self.dic_platoon_merge_time_by_size = \
        #     {1: 2.32, 2: 4.41, 3: 6.30, 4: 8.20, 5: 9.86, 6: 11.70,
        #      7: 13.50, 8: 15.12, 9: 16.85, 10: 18.40, 11: 20.12,
        #      12: 21.81, 13: 23.49, 14: 25.17, 15: 26.85, 16: 28.53,
        #      17: 30.21, 18: 31.89, 19: 33.57, 20: 35.25}

    def _get_R_MTW_duration(self, n):
        '''
        Get the estimated Ramp Platoon MTW (merging time window) duration for size n.
        From this table (measured data)
        self.dic_platoon_merge_time_by_size = \
            {1: 2.32, 2: 4.41, 3: 6.30, 4: 8.20, 5: 9.86, 6: 11.70,
             7: 13.50, 8: 15.12, 9: 16.85, 10: 18.40, 11: 20.12,
             12: 21.81, ...}
        '''
        return 1.753 * n + 0.989

    def jam_control(self, step, dic_platoon_info, dic_mplatoon_et, dic_vid_groups,
                    ls_r_dep_times, mpc_interval, delta_t, pf):

        self.jam_mode_start_ts = round(step/10 + 0.1, 1) \
            if self.jam_mode_start_ts is None else self.jam_mode_start_ts
        self.dic_mplatoon_et = dic_mplatoon_et
        self.pf = pf
        self.delta_t = delta_t
        disturb = self.loss_rate != 0
        if disturb:
            return self._jam_control_disturbed(
                step, dic_platoon_info,
                dic_vid_groups, ls_r_dep_times, mpc_interval)
        else:
            return self._jam_control_clean(
                step, dic_platoon_info,
                dic_vid_groups, ls_r_dep_times, mpc_interval)

    def _monitor_ramp_leader_stop(self, leader_id):
        """
        Monitors the vehicle and records the time when its speed becomes zero.
        :param leader_id: The vehicle ID to monitor
        :return: The time when the vehicle stops (speed = 0), or None if not stopped yet
        """
        # Get the current speed of the vehicle
        current_speed = self.data_recorder.dic_speed[leader_id]
        current_pos = self.data_recorder.dic_pos[leader_id]
        # If the speed is zero, record the simulation time
        if (current_speed == 0 and leader_id not in self.stop_times and
                current_pos > self.stop_pos - 5): # 203 - 5
            stop_time = self.traci.simulation.getTime()  # Get the simulation time
            self.stop_times[leader_id] = stop_time  # Record the time the vehicle stopped
            if len(self.stop_times) == 1:
                self.first_ramp_stop_ts = next(iter(self.stop_times.values()))
            return stop_time  # Return the stop time
        else:
            return None  # If the speed is not zero, return None

    def _monitor_ramp_platoon_tail_completed(self, dic_platoon_info):
        """
        Only activate in need
        Monitors the ramp platoon tail and records the time it completed passes the merging section.
        :param
                self.resume_times: {leader_id: resume_time, ...}
                dic_platoon_info: {leader_id: [[platoon_type, tail_id], deque([])], ...}
                leader: The ramp platoon leader
        :return:
                add completed time to self.stop_times
                self.rp_type_resume_completed: {leader_id: [type, resume_ts, tail_completed_ts]}
        """

        if not self.resume_times:
            return

        # Only iterate through the leaders currently in the waiting pool
        for leader, resume_time in self.resume_times.items():
            if leader in self.rp_type_resume_completed:
                continue
            platoon_info = dic_platoon_info.get(leader)
            platoon_type, tail_id = platoon_info[0]
            tail_id = leader if platoon_type == 'A' else tail_id
            # Check if the tail vehicle has entered the target merging lane 'ms_1'
            if self.data_recorder.dic_lane.get(tail_id) == 'ms_1':
                completed_ts = self.traci.simulation.getTime()
                # Record the completion time in resume_times
                self.rp_type_resume_completed[leader] = [platoon_type, resume_time, completed_ts]  # Record the platoon type, resume time, and completion time
        return self.rp_type_resume_completed

    def _get_r_leader_pre_stop(self):
        '''
        get ls of ramp leaders before stop point
            (multi-lane: first 120 meters on ramp; single lane: first 203.5 meters on ramp)
        ls_r_leader_up_asc => vehicle on RAMP_PROPER (curved / straight approach)
        :param:
            ls_r_leader_up_asc: list of ramp leaders before merging, asc order
        :return:
            ls_r_leader_pre_stop (ls_ravhb_acc) // ['ravh70', 'ravh200']
            ls_r_leader_fixed // ['ravh70']
        '''
        ls_r_leader_up_asc = self.dic_vid_groups["ls_r_leader_up_asc"]  # ['ravh70', 'ravh200', 'ravh440']
        ls_r_leader_pre_stop = []

        for vid in ls_r_leader_up_asc:
            pos = self.data_recorder.dic_pos[vid] # pos: distance to the start of lane
            if pos < self.stop_pos:
                ls_r_leader_pre_stop.append(vid)

        # Filter out the last leader because it belongs to the open platoon
        ls_r_leader_fixed = ls_r_leader_pre_stop[:-1] if len(ls_r_leader_pre_stop) > 1 else ls_r_leader_pre_stop

        return ls_r_leader_fixed

    def _get_r_proper(self): # _get_rvb_acc => _get_r_proper(self)
        '''
        get ls of ramp veh on RAMP_PROPER (section before acc lane)
        :return: self.ls_rvb_acc => self.ls_r_proper
        '''
        ls_r_veh_up = self.dic_vid_groups["ls_r_veh_up"]
        self.ls_r_proper = []
        self.ls_r_proper = ls_r_veh_up
        return self.ls_r_proper

    def _stop_ramp_fleet3(self):
        '''
        stop the ramp fleet closest to the weaving section
        :return:
        '''
        ls_r_leader_pre_stop = self._get_r_leader_pre_stop()
        # first ramp leader (on ramp_proper) before acc space
        first_r_leader_proper = ls_r_leader_pre_stop[0] if ls_r_leader_pre_stop else None
        # only dis < 203
        if (self.r_leader_stop is None
                and first_r_leader_proper is not None
                and first_r_leader_proper not in self.ls_skip_stop
                and first_r_leader_proper not in self.first_stop_recorded):
            # stop the first ramp_proper leader and make sure it's not in skip list
            try:
                self.traci.vehicle.setStop(first_r_leader_proper, 'inflow_merge', self.stop_pos, laneIndex=0)  # duration
                self.first_stop_recorded.append(first_r_leader_proper)
                self.r_leader_stop = first_r_leader_proper # ravh_stop => r_leader_stop
            except:
                pass

        # Monitor when the vehicle fully stops (speed becomes 0)
        self._monitor_ramp_leader_stop(self.r_leader_stop) if self.r_leader_stop is not None else None
        return self.r_leader_stop

    def _check_resume_state4(self, dic_platoon_info):
        '''
        default: False
        PS: sometimes the just stopped r_leader had already released, but the next r_leader in the process of stop
        between new stop and non-stop there is a gap where the latest stop_r_leader out of ramp_proper (control area)
        under this condition, it's still in resume_state = True
        :param dic_platoon_info:
        :return: resume_state
        '''

        ls_stopped_r_leader = list(self.stop_times.keys())

        if len(ls_stopped_r_leader) > 0:
            newest_stop_r_leader = ls_stopped_r_leader[-1]
            # 10032024updated: make sure all stop vehicle are not in acc space
            pos = self.data_recorder.dic_pos[newest_stop_r_leader] \
                if newest_stop_r_leader in self.traci.vehicle.getIDList() else None
            lane_id = self.data_recorder.dic_lane[newest_stop_r_leader] \
                if newest_stop_r_leader in self.traci.vehicle.getIDList() else None
            speed = self.data_recorder.dic_speed[newest_stop_r_leader] \
                if newest_stop_r_leader in self.traci.vehicle.getIDList() else None  # velocity_newest_stop_r_leader

            # pos < 210;
            if (speed == 0 and pos < self.stop_pos+5 and lane_id == 'inflow_merge_0'
                    and newest_stop_r_leader in dic_platoon_info): # new added 250615
                self.resume_state = False
                self.stop_state = True
            else:
                self.resume_state = True
                self.stop_state = False
        return

    def _get_remaining_t2(self, step, id, type='follower'):
        '''
        use prediction model to get t
        get the remaining time to the weaving section

        :param  id:
                type: 'leader', if its leader, max speed is faster
                self.data_recorder.dic_platoon_info: {vid:[type, tail_id, length1, length2...]}
        :return: re_t (remaining time)
                 dis (remaining dis)
        '''
        c_ts = round(step/10 + 0.1, 1)
        veh_info = self.data_recorder.get_vid_states(id)
        dis = veh_info['dis']
        v = veh_info['v']
        v = 0.0000001 if v== 0 else v

        if type == 'leader':
            re_t2 = self.merge_regular.estimate_travel_time(v, dis)
        else:
            # use prediction model
            # get its leader id
            leader_id = self.data_recorder.get_hv_leader(id, m=True)
            platoon_size = len(self.dic_mplatoon_et.get(leader_id, ([None]*12,))[0])
            (_, tail_id), _ = self.data_recorder.dic_platoon_info.get(leader_id, ([None, None], None))
            if (leader_id == None or id == leader_id or leader_id not in self.dic_mplatoon_et
                or tail_id != id or platoon_size > 100):
                re_t2 = self.merge_regular.estimate_travel_time(v, dis)
            else:
                re_t2 = self.dic_mplatoon_et[leader_id][2]-c_ts
        return re_t2, dis

    def _find_timing6(self, step, m_leader, action_m_leader,
                      max_interval, rp_pass_time):
        """
        find the resume timing for ramp leader
        :param
                ls_m_veh_up_asc: all veh on mainline before merging, ascending order
                ls_m_speed_up: speed of above vehicles

                m_leader: mainline leader behind the selected gap
                action_m_leader: mainline leader receiving a control action
                max_interval: biggest time gap between mainline platoons
                rp_pass_time: ramp platoons passing time

        :return:
        """
        c_ts = round(step/10 + 0.1, 1)
        if action_m_leader in ['r_av20', 'r_av297']:
            pass

        ls_m_veh_up_asc = self.dic_vid_groups.get('ls_m_veh_up_asc', None)  # ['mavh680', 'mhv690', 'mhv700', ] all veh on inflow_highway
        self.timing = False
        # 10-s cooldown; in the first 10-s forbid to resume ramp stopped platoon
        if (self.stop_state and self.first_ramp_stop_ts is not None):
            if c_ts - self.first_ramp_stop_ts < self.cooldown_dur:
                self.timing = False
                return self.timing

        # S4
        if self.stop_state and len(ls_m_veh_up_asc) == 0:
            self.timing = True
            return self.timing

        # S3
        # only one fleet on the mainline, no follower of fleet's last veh
        if self.stop_state and len(ls_m_veh_up_asc) > 0 and m_leader is None:
            last_m_veh = ls_m_veh_up_asc[-1]  # last veh on the mainline
            remain_time_to_ms, dis = self._get_remaining_t2(step, last_m_veh)
            if remain_time_to_ms + self.buffer <= self.r_leader_acc_dur:  # condition 8
                self.timing = True
                return self.timing

        # S6: another common situation, m_leader need to take action, and m_leader has a leader on inflow_highway
        # if self.stop_state and len(ls_m_veh_up_asc) > 0 and m_leader and m_leader_acting is True:
        if self.stop_state and len(ls_m_veh_up_asc) > 0 and action_m_leader and self.m_leader_acting is True:
            if action_m_leader == 'm_av4936':
                pass
            pv_info = self.traci.vehicle.getLeader(action_m_leader, self.length_mcz)
            if pv_info is not None:  # condition 6

                pv_id = pv_info[0]
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                # what's the difference self.merge_regular.estimate_travel_time()
                # remain_time_pv_ori, _ = self._get_remaining_t2(step, pv_id)
                # prev_leader = list(self.dic_mplatoon_et.keys())[list(self.dic_mplatoon_et.keys()).index(action_m_leader) - 1]
                # pv_tail_reach_ts = self.dic_mplatoon_et[prev_leader][2]
                pv_tail_reach_ts = self._get_prev_platoon_tail_at_ts(c_ts, action_m_leader)
                remain_time_pv = pv_tail_reach_ts - c_ts

                # desire_reaching_time = self.dic_desire_reach_ts[action_m_leader]
                # rp_tail_reach_time = c_ts + rp_pass_time + self.r_leader_acc_dur  # last veh of ramp platoon reaching time

                if pv_lane_id == 'inflow_highway' and remain_time_pv + self.buffer <= self.r_leader_acc_dur:
                    self.timing = True
                    # if desire_reaching_time > rp_tail_reach_time: # no need double check
                    #     self.timing = True
                    # else:
                    #     self.ls_skip_stop = []
                    return self.timing

        # S2: Most common situation without m_leader action
        if self.stop_state and len(ls_m_veh_up_asc) > 0 and m_leader and self.m_leader_acting is False:
            pv_info = self.traci.vehicle.getLeader(m_leader, self.length_mcz)
            if pv_info is not None:  # condition 6
                pv_id = pv_info[0]
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)

                pv_tail_reach_ts = self._get_prev_platoon_tail_at_ts(c_ts, m_leader)
                remain_time_pv = pv_tail_reach_ts - c_ts

                diff = self.r_leader_acc_dur - remain_time_pv
                if (pv_lane_id == 'inflow_highway'  # codition 7
                        and remain_time_pv + self.buffer <= self.r_leader_acc_dur  # condition 8
                        and max_interval - diff - self.buffer > rp_pass_time):  # condition 9
                    self.timing = True  # S2
                else:
                    self.ls_skip_stop = []  # updated: 241203
                return self.timing

        # S5: 100624updated, m_leader has no leader on inflow_highway
        if self.stop_state and len(ls_m_veh_up_asc) > 0 and m_leader and self.m_leader_acting is False:
            pv_info = self.traci.vehicle.getLeader(m_leader, self.length_mcz)
            if pv_info is not None:  # condition 6
                pv_id = pv_info[0]
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                if (pv_lane_id != 'inflow_highway'
                        and max_interval - self.buffer > rp_pass_time + self.r_leader_acc_dur):  # condition 12
                    self.timing = True  # S5
                    return self.timing
        return self.timing

    def _get_mainline_gaps(self, step,
                           ls_m_leader_up_asc, ls_m_veh_up_asc):
        '''
        get gaps between mainline platoons

        :param ls_m_leader_up_asc: min => max
               ls_m_veh_up_asc: ['mhv693', 'mhv765', 'mhv784', 'mhv795', 'mhv811', 'mav839']
                            seems like all vehicles on the merging section (ascending order)

               self.r_leader_acc_dur: the time needed for r_leader moving from stop point to the merging section
               dic_mplatoon_et: {m_leader:[platoon_type, ts_head, ts_tail, c_ts]}
               thw: time headway window
        :return dic_mainline_gaps: {m_leader: [gap_seconds]} for all windows.
                None identifies the window behind the last mainline vehicle.
        '''
        if step % 10 != 0:
            return self.dic_mainline_gaps
        c_ts = round(step / 10 + 0.1, 1)

        pfz_vehicles = self.dic_vid_groups['ls_pf_veh_asc']
        pfz_leaders = self.dic_vid_groups['ls_pf_leader_asc']

        first_pfz_vehicle = pfz_vehicles[0] if pfz_vehicles else None
        first_pfz_is_leader = (
                first_pfz_vehicle is not None
                and bool(pfz_leaders)
                and first_pfz_vehicle == pfz_leaders[0])

        headway_differences = {}

        # 1. no vehicles on the MCZ
        if len(ls_m_veh_up_asc) == 0:
            thw = self.length_mcz / self.speed_level3 - self.r_leader_acc_dur
            if not pfz_vehicles:
                thw = ((self.data_recorder.length_pfz + self.length_mcz)
                       / self.speed_level3 - self.r_leader_acc_dur)
            elif (first_pfz_is_leader
                  and self.merge_regular.first_pfz_leader == first_pfz_vehicle
                  and self.merge_regular.first_pfz_leader_arrival_ts is not None):
                thw = (self.merge_regular.first_pfz_leader_arrival_ts
                       - c_ts - self.r_leader_acc_dur)
            headway_differences[None] = thw
        # 2. no leader on the merging control section, but there are followers
        elif len(ls_m_leader_up_asc) == 0:
            headway_differences[None] = 0
        # 3. both leaders and followers exists
        else:
            # Dictionary to store the time differences for each head vehicle
            first_m_leader = ls_m_leader_up_asc[0]
            first_veh = ls_m_veh_up_asc[0]

            # 3.1 between the last platoon tail and the first pfz-leader (if exists)
            last_mcz_leader = ls_m_leader_up_asc[-1]
            last_mcz_platoon_tail_ts = self.dic_mplatoon_et[last_mcz_leader][2]
            gap_start = last_mcz_platoon_tail_ts
            gap_end = self.merge_regular.first_pfz_leader_arrival_ts
            leader = pfz_leaders[0] if pfz_leaders else None
            if gap_end is None:
                gap_end = self.length_mcz / self.speed_level3 + c_ts
                leader = None
            thw = gap_end - max(gap_start, c_ts + self.r_leader_acc_dur)
            if thw > 0:
                headway_differences[leader] = thw

            # 3.2 between the first platoon leader and the MS (Merging section) entry
            if first_m_leader == first_veh:
                dis = self.data_recorder.get_vid_states(first_m_leader)['dis']
                # dic_mplatoon_et: {m_leader:[platoon_type, ts_head, ts_tail, c_ts]}
                first_veh_info = self.dic_mplatoon_et.get(first_m_leader, [None, None])
                arrive_time = first_veh_info[1] # ts_head
                # Calculate real headway using a ternary expression
                real_headway = dis / self.speed_level3 if arrive_time is None \
                    else arrive_time - c_ts
                headway_differences[first_m_leader] = real_headway - self.r_leader_acc_dur

            # 3.3 between platoons
            for i, leader in enumerate(ls_m_leader_up_asc):
                if leader == first_m_leader and first_m_leader == first_veh:
                    # This gap has already calculated in 3.2.
                    continue
                if leader not in self.dic_mplatoon_et:
                    headway_differences[leader] = 0
                elif i >= 0:
                    ts_head_current = self.dic_mplatoon_et[leader][1]
                    ts_prev_tail = self._get_prev_platoon_tail_at_ts(c_ts, leader)
                    ts_front_remaining = ts_prev_tail - c_ts
                    if ts_front_remaining >= self.r_leader_acc_dur:
                        headway_differences[leader] = ts_head_current - ts_prev_tail
                    else:
                        headway_differences[leader] = ts_head_current - ts_prev_tail - (
                                self.r_leader_acc_dur - ts_front_remaining)

        self.dic_mainline_gaps = {
            leader_id: gap_seconds
            for leader_id, gap_seconds in headway_differences.items()}
        return self.dic_mainline_gaps

    def _get_max_interval_ml(self, step,
                             ls_m_leader_up_asc, ls_m_veh_up_asc):
        '''
        ml - multi-lane version
        get the max interval on the mainline
        241203 updated, consider the acc time
        112624 updated, use prediction model

        :param ls_m_leader_up_asc: min => max
               ls_m_veh_up_asc: ['mhv693', 'mhv765', 'mhv784', 'mhv795', 'mhv811', 'mav839']
                            seems like all vehicles on the merging section (ascending order)

               self.r_leader_acc_dur: the time needed for r_leader moving from stop point to the merging section
               dic_mplatoon_et: {m_leader:[platoon_type, ts_head, ts_tail, c_ts]}
               thw: time headway window
        :return dic_max_interval: {m_leader: [max_thw]}
        '''

        if step % 10 != 0:
            return self.dic_max_interval
        c_ts = round(step / 10 + 0.1, 1)

        # 1. no vehicles on the merging section
        if len(ls_m_veh_up_asc) == 0:
            m_leader = None
            max_thw = self.length_mcz / self.speed_level3 - self.r_leader_acc_dur  # the acc time(consider)
        # 2. no leader on the merging control section, but there are followers
        elif len(ls_m_leader_up_asc) == 0:
            m_leader = None
            max_thw = 0
        # 3. both leaders and followers exists
        else:
            # Dictionary to store the time differences for each head vehicle
            headway_differences = {}
            first_m_leader = ls_m_leader_up_asc[0]
            first_veh = ls_m_veh_up_asc[0]

            # 3.1 between the last platoon tail and the start of merging control section
            last_mvb = ls_m_veh_up_asc[-1]
            veh_info = self.data_recorder.get_vid_states(last_mvb)
            dis = veh_info['dis'] # distance to the end of lane
            # (self.length_mcz - dis): the distance between last veh and start of merging control section
            thw = (self.length_mcz - dis) / self.speed_level3
            headway_differences[None] = thw

            # 3.2 between the first platoon and the merging(weaving) section
            if first_m_leader == first_veh:
                dis = self.data_recorder.get_vid_states(first_m_leader)['dis']
                first_veh_info = self.dic_mplatoon_et.get(first_m_leader, [None, None])
                arrive_time = first_veh_info[1]
                # Calculate real headway using a ternary expression
                real_headway = dis / self.speed_level3 if arrive_time is None \
                    else arrive_time - c_ts # self.r_leader_acc_dur
                headway_differences[first_m_leader] = real_headway - self.r_leader_acc_dur

            # 3.3 between platoons
            for i, head_id in enumerate(ls_m_leader_up_asc):
                if head_id == first_m_leader and first_m_leader == first_veh:
                    # This gap has already calculated in 3.2.
                    continue
                if head_id not in self.dic_mplatoon_et:
                    headway_differences[head_id] = 0
                elif i >= 0:
                    ts_head_current = self.dic_mplatoon_et[head_id][1]
                    ts_prev_tail = self._get_prev_platoon_tail_at_ts(c_ts, head_id)
                    ts_front_remaining = ts_prev_tail - c_ts
                    if ts_front_remaining >= self.r_leader_acc_dur:
                        headway_differences[head_id] = ts_head_current - ts_prev_tail
                    else:
                        headway_differences[head_id] = ts_head_current - ts_prev_tail - (
                                self.r_leader_acc_dur - ts_front_remaining)

            # get the max thw
            m_leader, max_thw = max(headway_differences.items(), key=lambda x: x[1])
        # last veh on inflow_highway
        self.dic_max_interval = {m_leader: [max_thw]} # dic_result (self.dic_max_interval)
        return self.dic_max_interval

    def _get_prev_platoon_tail_at_ts(self, c_ts, this_leader):
        prev_leader = list(self.dic_mplatoon_et.keys())[
            list(self.dic_mplatoon_et.keys()).index(this_leader) - 1]

        prev_platoon_size = len(self.dic_mplatoon_et[prev_leader][0])
        if prev_platoon_size > 100:  # if previous platoon size is too big, use on-board sensor value to predict
            p_veh_info = self.traci.vehicle.getLeader(this_leader, float('inf'))
            front_id = p_veh_info[0] if p_veh_info is not None else None
            front_veh_info = self.data_recorder.get_vid_states(front_id)
            ts_prev_tail = self.merge_regular.estimate_travel_time(
                front_veh_info['v'], front_veh_info['dis']) + c_ts
        else:  # if normal platoon size, use prediction model
            ts_prev_tail = self.dic_mplatoon_et[prev_leader][2]
        return ts_prev_tail

    def _get_ramp_platoon_merge_duration(self, dic_platoon_info):
        # _get_r_traveltime3
        """
        update: only r_leader before stop point // 0928.2024
        get corresponding ramp platoon merging completion time, from leader to tail merging completion
        :params:
            ls_r_leader_pre_stop: list of ramp leader before stop point
            dic_platoon_info: {leader_id: [[platoon_type, tail_id], deque([])], ...}
            dic_ramp_platoon_info: only keep ls_r_leader_pre_stop in
            dic_ramp_platoon_basic: delete length of platoon info, only keep type and tail_id
        :return: {'ravh880': ['AHHHHH', 'rhv930', 15.3], 'ravh680': ['AHH', 'rhv700', 8.8]}
        """
        ls_r_leader_pre_stop = self._get_r_leader_pre_stop()
        dic_ramp_platoon_info = \
            {r_leader: dic_platoon_info[r_leader] for r_leader in ls_r_leader_pre_stop if
             r_leader in dic_platoon_info}
        dic_ramp_platoon_basic = {key: value[0][:2] for key, value in dic_ramp_platoon_info.items()}  # 20240929update: fixed length platoon info
        # dic_r_platoon_travel_time = {key: value + [self.dic_platoon_merge_time_by_size[len(value[0])]]
        #                              for key, value in dic_ramp_platoon_basic.items()}
        dic_r_platoon_travel_time = {key: value + [self._get_R_MTW_duration(len(value[0]))]
                                     for key, value in dic_ramp_platoon_basic.items()}
        return dic_r_platoon_travel_time

    def _compare3(self, dic_max_interval, dic_r_platoon_travel_time):
        """
        filter dic_r_platoon_merge_duration after using _get_ramp_platoon_merge_duration //0928.2024
        updated from find_suitable_interval2
        Targets: 2. determing how many ramp fleets can pass at this interval
        :param  min_time_diff: minimal time difference
                dic_max_interval: {m_leader:[mpv_id, max_dis, max_thw]}
                dic_r_platoon_travel_time: {r_leader: [platoon_type, tail_id, rp_pass_time], ...}
        :return: final_rp_pass_time: final ramp passing time (accumulate ramp passing time)
        """
        if dic_max_interval and dic_r_platoon_travel_time and self.stop_state:
            # find the biggest interval and corresponding m_leader
            m_leader = list(dic_max_interval.keys())[0]
            max_interval = dic_max_interval[m_leader][-1]
            num_ramp_platoon = len(dic_r_platoon_travel_time) # number of ramp platoons
            cum_rp_pass_time = 0 # Cumulative time
            final_rp_type = '' # the final combined ramp platoon type
            ls_pass_rid = [] # all r_leader_id can pass
            for i in range(num_ramp_platoon): # 3; i=0, 1, 2
                rp_info = list(dic_r_platoon_travel_time.items())[i] # the ramp fleet info
                rp_leader = rp_info[0] # the leader if of ramp platoon
                rp_pass_time = rp_info[1][2] # the ramp fleet passing time
                rp_type = rp_info[1][0] # the ramp fleet type

                # judge if rp_leader is in stop state
                speed = self.data_recorder.dic_speed[rp_leader]
                if speed < 0.8:
                    final_rp_type += rp_type
                    final_rp_number = len(final_rp_type)
                    # accumulate ramp passing time
                    # cum_rp_pass_time = self.dic_platoon_merge_time_by_size.get(final_rp_number, float('inf'))
                    cum_rp_pass_time = self._get_R_MTW_duration(final_rp_number)
                    if cum_rp_pass_time < max_interval + self.delta_t:
                        ls_pass_rid.append(rp_leader)
                        if rp_leader not in self.ls_skip_stop and i != 0:
                            if rp_leader == 'r_av20' or rp_leader == 'r_av297':
                                pass
                            self.ls_skip_stop.append(rp_leader)
                        final_rp_pass_time = cum_rp_pass_time # final ramp passing time
                    else:
                        break  # jump out this loop
                else:
                    break

            if len(ls_pass_rid) > 0:
                # print(f'ramp platoons number passed this time:{len(ls_pass_rid)}')
                return m_leader, max_interval, final_rp_pass_time
            else:
                return m_leader, max_interval, cum_rp_pass_time
        else:
            return None, None, None

    def _restart_ramp_fleet(self, step, first_r_leader, timing):
        """
        250613 update, avoid r_leader stopped before stop point (self.stop_pos = 203.5)
        :param dic_mavhb_hinfo:
               m_leader:
               first_r_leader: first ramp platoon leader (head)
        :return:
        """
        c_ts = round(step/10 + 0.1, 1)
        pos = self.data_recorder.dic_pos[first_r_leader] if first_r_leader else 0
        if (timing
                and first_r_leader not in self.first_resume_recorded
                and self.stop_state
                and not self.resume_state
                and first_r_leader
                and pos >= self.stop_pos-1): # new
            # make sure stop_state=True, resume_state=False
            # (finished resume, whole fleet finished pass weaving section)
            v_r_leader_f = self.data_recorder.dic_speed[first_r_leader]
            if v_r_leader_f > 0:
                pass
            else:
                if first_r_leader == 'ravh5450':
                    pass
                self.traci.vehicle.resume(first_r_leader)
                self.first_resume_recorded.append(first_r_leader)
                self.r_leader_stop = None
                self.resume_times[first_r_leader] = c_ts

    def _get_m_leader_action(self, step, first_r_leader,
                             rp_pass_dur, m_leader, max_interval,
                             mpc_interval):
        """
        _get_mavh_action => _get_m_leader_action
        Decide whether a MAVH (mainline leader) should take action to match the desired merging time.

        Params:
            step: current simulation step
            first_r_leader: first ramp leader ID
            pv_m_leader: preceding vehicle of m_leader (short as pv)
            rp_pass_dur: ramp platoon passing duration
            m_leader: candidate mainline vehicle ID
            max_interval: max time gap between ramp platoon and MAVH
            dic_mplatoon_et: estimated arrival time dict for platoon
            delta_t: allowable timing error
            mpc_interval: frequency of evaluation
            ts: timestamp
            dur: duration (time period)
        Returns:
            self.dic_mavh_actionP: dict of MAVH (m_leader) and its action parameters
            => self.dic_m_leader_action_params = {m_leader: [, c_ts]}
        """
        c_ts = round(step / 10 + 0.1, 1)
        allowable_error = self.delta_t  # 0, 2, 4, 6, 8, 10
        last_stop_ts = list(self.stop_times.items())[-1][-1] if self.stop_times else None
        if self.first_ramp_stop_ts is not None:
            # in cooldown period, not action
            if c_ts - self.first_ramp_stop_ts < self.cooldown_dur:
                return self.dic_m_leader_action_params

        if not (step % mpc_interval == 0 or
                (last_stop_ts is not None and c_ts == last_stop_ts + 0.1) or
                self.merge_regular.new_leader_flag):
            # *10 because sim_step=0.1
            # move forward only if in mpc_interval or just after stop
            # or just new m_leader emerged in MCZ-M
            return self.dic_m_leader_action_params

        if not m_leader:
            return self.dic_m_leader_action_params

        if (self.data_recorder.ms_exit_speed is not None and
                self.data_recorder.ms_exit_speed < 19.44):
            # if latest veihcle's speed exit MS is too low, no action
            return self.dic_m_leader_action_params

        pv_m_leader_info = self.traci.vehicle.getLeader(m_leader)
        pv_m_leader, dis_m_leader_to_pv = pv_m_leader_info if pv_m_leader_info else (None, None)
        if not pv_m_leader:
            return self.dic_m_leader_action_params

        pv_m_leader_lane = self.data_recorder.dic_lane[pv_m_leader]
        if not self.stop_state or pv_m_leader_lane != 'inflow_highway_0':
            return self.dic_m_leader_action_params

        pv_tail_reach_ts = self._get_prev_platoon_tail_at_ts(c_ts, m_leader)
        pv_tail_reach_dur = max(0, pv_tail_reach_ts - c_ts)
        # dev_rleader_pmtail: time deviation between r_leader and previous m_tail
        dev_rleader_pmtail = max(0, self.r_leader_acc_dur - pv_tail_reach_dur)  # self.r_leader_acc_dur = 9,3 (ml) or 12
        des_interval = dev_rleader_pmtail + rp_pass_dur + self.buffer * 2
        interval_shortage = des_interval - max_interval
        if interval_shortage <= 0:  # no need to take action
            self.dic_m_leader_action_params = {m_leader: []}
            return self.dic_m_leader_action_params

        # interval_shortage > 0, m_leader needs to take action
        if self.stop_times[first_r_leader] == self.first_ramp_stop_ts:
            r_leader_waiting_dur = c_ts - self.stop_times[first_r_leader] - self.cooldown_dur
        else:
            r_leader_waiting_dur = c_ts - self.stop_times[first_r_leader]

        if m_leader == 'm_av5112':
            pass

        # Case 1: If r_leader has been waiting too long, allow looser error margin to avoid long waiting
        if r_leader_waiting_dur > 30 and interval_shortage <= allowable_error + 10:
            # Looser threshold due to long waiting time
            des_m_leader_reach_dur = des_interval + pv_tail_reach_dur
        # Case 2: Otherwise, allow only if within strict allowable error
        elif interval_shortage <= allowable_error:
            # Strict error control
            des_m_leader_reach_dur = des_interval + pv_tail_reach_dur
        else:
            return self.dic_m_leader_action_params

        dic_m_leader_info = self.data_recorder.get_vid_states(m_leader)
        m_dis = dic_m_leader_info['dis']  # m_leader distance to ms
        m_v0 = dic_m_leader_info['v']
        action_params = list(self.merge_regular.get_action_params(des_m_leader_reach_dur, m_dis, m_v0))
        action_params.append(c_ts)  # (t1, a1, t3, a3, v_reach, c_ts)
        self.m_leader_action_dic[m_leader] = action_params
        self.dic_m_leader_action_params = {m_leader: action_params}
        return self.dic_m_leader_action_params

    def _apply_m_leader_control(self, step, dic_m_leader_action_params):
        '''
        Apply m_leader action params
        :param dic_m_leader_action_params: action params,
            format => {'mavh1630': [14.3, -1.1, 14.6, 1.1, 24.9, 168.1]}
        :return: the list of action_m_leader
        '''
        ls_m_leader_up_asc = self.dic_vid_groups.get('ls_m_leader_up_asc', None)
        action_m_leader = next(iter(dic_m_leader_action_params or {}), None)
        self.m_leader_acting = False # should be m_leader_acting
        if (action_m_leader in self.m_leader_action_dic
            and action_m_leader in ls_m_leader_up_asc):
            # apply action
            self.merge_regular.apply_leader_action(step, dic_m_leader_action_params)
            # flash
            self.merge_regular.flashing_merging(step, [action_m_leader])
            self.m_leader_acting = True
        return action_m_leader

    def _jam_control_clean(self, step, dic_platoon_info,
                               dic_vid_groups, ls_r_dep_times,
                               mpc_interval):
        '''
        orignial version
        update 25.1.30
        :param step:
               dic_platoon_info: {'ravh40': [['AHHHHHHHHH', 'rhv130'], deque([p_length1, p_length2], maxlen=10)],
                                    'mavh40': [['AHHHHHHHHH', 'mhv130'], deque([p_length1, p_length2], maxlen=10)]}
               ls_m_leader_up_asc: ['mavh1920']
               ls_m_veh_up_asc: ['mhv693', 'mhv765', 'mhv784', 'mhv795', 'mhv811', 'mav839']
               dic_mplatoon_et: {'mavh40': ['AHHHHHHHHH', 44.57, 64.42],
                                'mavh310': ['AHHHHHHHHHHH', 63.33, 87.53, 63.10]}
               dic_vid_groups: all veh info
               dic_id_speed: {id1 : [speed1, speed2, ...], id2: [speed1, speed2, ...]}
               ls_r_dep_times: [4, 5, 6, ....]
        :return:
        '''
        prc.print_message('**in jam mode**')
        self.dic_vid_groups = dic_vid_groups
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']
        ls_m_veh_up_asc = dic_vid_groups['ls_m_veh_up_asc']
        first_r_leader = self._stop_ramp_fleet3()  # Stop first ramp leader
        self._check_resume_state4(dic_platoon_info) # 241003update
        # Get Ramp Platoon travel Time (from stop to pass intersection)
        dic_r_platoon_travel_time = self._get_ramp_platoon_merge_duration(dic_platoon_info)
        dic_max_interval = self._get_max_interval_ml(step, ls_m_leader_up_asc, ls_m_veh_up_asc)
        m_leader, max_interval, final_rp_pass_time = self._compare3(dic_max_interval,
                                                                    dic_r_platoon_travel_time)
        # decide if m_leader need to take action
        dic_m_leader_action_params = self._get_m_leader_action(step, first_r_leader,
                                                               final_rp_pass_time,
                                                               m_leader, max_interval,
                                                               mpc_interval)
        action_m_leader = self._apply_m_leader_control(step, dic_m_leader_action_params)
        timing = self._find_timing6(step, m_leader, action_m_leader,
                                    max_interval, final_rp_pass_time)
        ls_r_proper = self._get_r_proper()
        # record ramp queue length
        queue_log = self.data_recorder.get_queue_length(step, ls_r_proper, ls_r_dep_times)
        self._restart_ramp_fleet(step, first_r_leader, timing) # Resume r_leader
        return queue_log

    def _jam_control_disturbed(self, step, dic_platoon_info,
                               dic_vid_groups, ls_r_dep_times, mpc_interval):
        '''
        add v2x disturbance
        update 25.1.30
        param
            step:
            dic_platoon_info:
            dic_vid_groups: vehicle groups including ls_m_leader_up_asc and ls_m_veh_up_asc
            dic_id_speed:
            ls_r_dep_times:
            mpc_interval:

        return
            action_params: [t_dec, a_dec, t_acc, a_acc, reach_v, t_recorded]

        '''
        c_ts = round(step / 10 + 0.1, 1)
        c_ts % 1 == 0 and prc.print_message('**in jam mode**')
        self.dic_vid_groups = dic_vid_groups
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']
        ls_m_veh_up_asc = dic_vid_groups['ls_m_veh_up_asc']
        # Stop r_leader (the first one)
        first_r_leader = self._stop_ramp_fleet3()  # first r_leader id
        self._check_resume_state4(dic_platoon_info) # 241003update
        # Get ramp fleet travel time (from stop to pass intersection)
        dic_r_platoon_travel_time = self._get_ramp_platoon_merge_duration(dic_platoon_info)
        dic_max_interval = self._get_max_interval_ml(step, self.ml,
                                                        ls_m_leader_up_asc, ls_m_veh_up_asc)
        m_leader, max_interval, final_rp_pass_time = self._compare3(dic_max_interval,
                                                                    dic_r_platoon_travel_time)

        # m_leader take action; m_leader_acting = True/False
        action_params = self._get_m_leader_action(step, first_r_leader, final_rp_pass_time,
                                                  m_leader, max_interval, mpc_interval)  # MPC interval = 6s
        if action_params and any(action_params.values()) and action_params != self.last_action_params:
            self.action_buffer.push(step, action_params)
        action_pay_load = self.action_buffer.maybe_release(step)
        if action_pay_load and any(action_pay_load.values()):
            self.m_action_params = action_pay_load
            pass
        action_m_leader = self._apply_m_leader_control(step, self.m_action_params) # pay_load = dic_m_leader_action_params
        timing = self._find_timing6(step, m_leader, action_m_leader, max_interval, final_rp_pass_time)
        if timing:
            self.timing_buffer.push(step, timing)
        timing = self.timing_buffer.maybe_release(step)
        # record ramp queue length
        ls_r_proper = self._get_r_proper()
        queue_log = self.data_recorder.get_queue_length(step, ls_r_proper, ls_r_dep_times)
        # Resume r_leader
        self._restart_ramp_fleet(step, first_r_leader, timing)
        self.last_action_params = action_params
        return queue_log
