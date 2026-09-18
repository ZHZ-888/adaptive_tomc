'''
merging_control_jam.py
TODO: call traci to get time many times, maybe once is enough
'''

from functions import print_control as prc
from functions import v2x_disturbance as v2x

class MergingControlJam:
    def __init__(self, traci, instance_dr, merge_regular, loss_rate, ml, comm_rng=None):
        self.traci = traci
        self.data_recorder = instance_dr
        self.merge_regular = merge_regular
        self.ml = ml # multi-lane?

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
        self.ls_r_leader_pre_stop = [] # list of ramp leader before stop point
        self.ls_r_proper = [] # list of ramp veh before acc
        self.dic_max_interval = {} # {m_leader: max_interval}

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
        self.cooldown_dur = 30
        self.last_m_leader = None

        self.buffer = 1.5 # platoon-to-platoon time buffer, 1.5 s

        if ml:
            self.speed_level3 = 25
            # the time needed for ramp AV leader moving from stop point to the merging section (weaving section)
            self.r_leader_acc_dur = 11.5 # 9.3; 11.5
            # Mapping from platoon size to total merge completion time (from leader start to tail completing merging)
            # after 13 may not that accurate, but only in case max interval is very large, then allow the ramp combined number exceeds 12
            self.dic_platoon_merge_time_by_size = {1: 2.32, 2: 4.41, 3: 6.30, 4: 8.20, 5: 9.86, 6: 11.70, 7: 13.50,
                                                   8: 15.12, 9: 16.85, 10: 18.40, 11: 20.12, 12: 21.81,
                                                   13: 23.49, 14: 25.17, 15: 26.85, 16: 28.53, 17: 30.21, 18: 31.89,
                                                   19: 33.57, 20: 35.25}
            self.stop_pos = 120
        else: # single lane
            self.r_leader_acc_dur = 12 # single lane 12 seconds
            self.dic_platoon_merge_time_by_size = {1: 3.75, 2: 6.17, 3: 8.3, 4: 10.6, 5: 12.67, 6: 14.73, 7: 16.91,
                                                   8: 19.0, 9: 21.02, 10: 23.98, 11: 26.3, 12: 28.99}
            self.stop_pos = 203.5 # stop pos of ramp platoon


    def jam_control(self, step, dic_platoon_info, ls_m_leader_up_asc, ls_m_veh_up_asc,
                          dic_mplatoon_et, dic_vid_groups, ls_r_dep_times, mpc_interval, delta_t, pf):
        self.jam_mode_start_ts = round(step/10 + 0.1, 1) if self.jam_mode_start_ts is None else self.jam_mode_start_ts

        disturb = self.loss_rate != 0
        self.dic_mplatoon_et = dic_mplatoon_et
        self.pf = pf
        self.delta_t = delta_t
        if disturb:
            return self._jam_control_disturbed(step, dic_platoon_info, ls_m_leader_up_asc, ls_m_veh_up_asc,
                          dic_vid_groups, ls_r_dep_times, mpc_interval)
        else:
            return self._jam_control_clean(step, dic_platoon_info, ls_m_leader_up_asc, ls_m_veh_up_asc,
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
            # Check if the tail vehicle has entered the target merging lane 'ws_1'
            if self.data_recorder.dic_lane.get(tail_id) == 'ws_1':
                completed_ts = self.traci.simulation.getTime()
                # Record the completion time in resume_times
                self.rp_type_resume_completed[leader] = [platoon_type, resume_time, completed_ts]  # Record the platoon type, resume time, and completion time
        return self.rp_type_resume_completed

    def _get_r_leader_pre_stop(self): # _get_ravhb_acc()
        '''
        get ls of ramp leaders before stop point
            (multi-lane: first 120 meters on ramp; single lane: first 203.5 meters on ramp)
        ls_r_leader_up (ls_ravhb) => vehicle on RAMP_PROPER (curved / straight approach)
        :param:
            ls_r_leader_up: list of ramp leaders before merging, descending order
        :return:
            ls_r_leader_pre_stop (ls_ravhb_acc)//
            ls_ravhb_acc_r//['ravh1050', 'ravh920', 'ravh680', 'ravh440', 'ravh200', 'ravh70']
        '''
        ls_r_leader_up = self.dic_vid_groups["ls_r_leader_up"]  # ['ravh200', 'ravh70']
        self.ls_r_leader_pre_stop = []

        for id in ls_r_leader_up[::-1]:
            pos = self.data_recorder.dic_pos[id]
            if pos < self.stop_pos:
                self.ls_r_leader_pre_stop.append(id)

        return self.ls_r_leader_pre_stop

    def _get_r_proper(self): # _get_rvb_acc => _get_r_proper(self)
        '''
        get ls of ramp veh on RAMP_PROPER (section before acc lane)
        :return: self.ls_rvb_acc => self.ls_r_proper
        '''
        ls_r_veh_up = self.dic_vid_groups["ls_r_veh_up"]
        self.ls_r_proper = []
        if self.ml:
            self.ls_r_proper = ls_r_veh_up
            return self.ls_r_proper
        for id in ls_r_veh_up[::-1]:
            pos = self.data_recorder.dic_pos[id]
            if pos < self.stop_pos:
                self.ls_r_proper.append(id)
        return self.ls_r_proper

    def _stop_ramp_fleet3(self, dic_platoon_info):
        '''
        stop the ramp fleet closest to the weaving section
        :return:
        '''
        ls_r_leader_pre_stop = self._get_r_leader_pre_stop()
        first_r_leader_proper = ls_r_leader_pre_stop[0] if ls_r_leader_pre_stop else None # first ramp leader (on ramp_proper) before acc space
        # only dis < 203
        if first_r_leader_proper == 'ravh5930':
            pass

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
        # rp_resume_completed = self._monitor_ramp_platoon_tail_completed(dic_platoon_info)
        return self.r_leader_stop

    def _check_resume_state4(self, dic_platoon_info):
        '''
        default: False
        notation: sometimes newest stop_r_leader already gone, however the next r_leader still in the process of stop
        between new stop and non-stop there is a gap where latest stop_r_leader out of ramp_proper (control area)
        under this condition, it still under resume_state = True
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

    def _find_timing6(self, step, m_leader, action_m_leader, max_interval, rp_pass_time):
        """
        find the resume timing for ramp leader
        :param
                ls_m_veh_up_asc: all veh on mainline before merging, ascending order
                ls_m_speed_up: speed of above vehicles

                m_leader: choosing m_leader
                action_m_leader: ????
                max_interval: biggest time gap between mainline platoons
                rp_pass_time: ramp platoons passing time

        :return:
        """
        c_ts = round(step/10 + 0.1, 1)
        if action_m_leader == 'mb_av5634':
            pass

        ls_m_veh_up_asc = self.dic_vid_groups.get('ls_m_veh_up_asc', None)  # ['mavh680', 'mhv690', 'mhv700', ] all veh on inflow_highway
        ls_m_speed_up = [self.dic_id_speed[id] for id in ls_m_veh_up_asc] if ls_m_veh_up_asc else None  # velocity of every veh
        self.timing = False
        # 10-s cooldown; in the first 10-s forbid to resume ramp stopped platoon
        if (self.stop_state and self.first_ramp_stop_ts is not None):
            if c_ts - self.first_ramp_stop_ts < self.cooldown_dur:
                self.timing = False
                return self.timing
        # S1
        # if (self.stop_state
        #         and len(ls_m_veh_up_asc) > 0
        #         and len(self.first_resume_recorded) == 0  # condition 5
        #         and min(ls_m_speed_up) < 5):  # condition 11
        #     self.timing = False
        #     return self.timing

        # S4
        if self.stop_state and len(ls_m_veh_up_asc) == 0:
            self.timing = True
            return self.timing

        # S3
        # only one fleet on the mainline, no follower of fleet's last veh
        if self.stop_state and len(ls_m_veh_up_asc) > 0 and m_leader is None:
            last_m_veh = ls_m_veh_up_asc[-1]  # last veh on the mainline
            remain_time_to_ws, dis = self._get_remaining_t2(step, last_m_veh)
            if remain_time_to_ws + self.buffer <= self.r_leader_acc_dur:  # condition 8
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

    def _get_max_interval_upper(self, step, ml, ls_m_leader_up_asc, ls_m_veh_up_asc):
        '''get the max interval on the mainline, call different function according to ml or not'''
        if ml:
            return self._get_max_interval_ml(step, ls_m_leader_up_asc, ls_m_veh_up_asc)
        else:
            return self._get_max_interval_single(step, ls_m_leader_up_asc, ls_m_veh_up_asc)

    def _get_max_interval_single(self, step, ls_m_leader_up_asc, ls_m_veh_up_asc):
        '''
        get the max interval on the mainline
        241203 updated, consider the acc time
        112624 updated, use prediction model

        :param ls_m_leader_up_asc: min => max
               ls_m_veh_up_asc: ['mhv693', 'mhv765', 'mhv784', 'mhv795', 'mhv811', 'mav839']
                            seems like all vehicles on the merging section (ascending order)

               self.r_leader_acc_dur: the time needed for r_leader moving from stop point to the merging section
               dic_mplatoon_et: {m_leader:[platoon_type, ts_head, ts_tail, c_ts]}
               thw: time headway window
        :return

        '''
        c_ts = self.traci.simulation.getTime()

        # 1. no vehicles on the merging section
        if len(ls_m_veh_up_asc) == 0:
            m_leader = None
            max_thw = self.length_mcz / self.max_speed - self.r_leader_acc_dur  # the acc time(consider)
        # 2. no leader on the merging section, but there are followers
        elif len(ls_m_leader_up_asc) == 0:
            m_leader = None
            max_thw = 0
        # 3. both leaders and followers exist
        else:
            # Dictionary to store the time differences for each head vehicle
            headway_differences = {}
            first_m_leader = ls_m_leader_up_asc[0]
            first_veh = ls_m_veh_up_asc[0]

            # 3.1 between the last platoon and the start point inflow_highway
            last_mvb = ls_m_veh_up_asc[-1]
            veh_info = self.data_recorder.get_vid_states(last_mvb)

            dis = veh_info['dis']
            # (self.length_mcz - dis) => the distance between last veh and start point of merging control section
            thw = (self.length_mcz - dis) / self.max_speed
            headway_differences[None] = thw

            # 3.2 between the first platoon and the weaving section
            if first_m_leader == first_veh:
                dis = self.data_recorder.get_vid_states(first_m_leader)['dis']
                first_veh_info = self.dic_mplatoon_et.get(first_m_leader, [None, None])  # first vehicle arrival time
                arrive_time = first_veh_info[1]
                # Calculate real headway using a ternary expression
                real_headway = dis / self.max_speed - self.r_leader_acc_dur if arrive_time is None \
                    else arrive_time - c_ts - self.r_leader_acc_dur
                headway_differences[first_m_leader] = real_headway

            # 3.3 between platoons
            for i, head_id in enumerate(ls_m_leader_up_asc):
                if head_id == 'mbav11196':
                    pass
                if head_id not in self.dic_mplatoon_et:
                    headway_differences[head_id] = 0
                elif i == 0 and head_id != first_veh:
                    ts_head_current = self.dic_mplatoon_et[head_id][1]
                    keys = list(self.dic_mplatoon_et.keys())
                    this_index = keys.index(head_id)
                    previous_index = this_index - 1
                    previous_m_leader = keys[previous_index]
                    ts_tail_previous = self.dic_mplatoon_et[previous_m_leader][2]
                    headway_differences[head_id] = ts_head_current - ts_tail_previous
                elif i > 0:
                    # Get the arrival time of the current head vehicle
                    ts_head_current = self.dic_mplatoon_et[head_id][1]
                    # Get the arrival time of the preceding vehicle's tail
                    if ls_m_leader_up_asc[i-1] == 'mbav11196':
                        pass
                    ts_tail_previous = self.dic_mplatoon_et[ls_m_leader_up_asc[i - 1]][2]
                    # Get the remaining time of the preceding vehicle
                    ts_tail_remaining = ts_tail_previous - c_ts
                    if ts_tail_remaining >= self.r_leader_acc_dur:
                        # Calculate the time difference
                        headway_differences[head_id] = ts_head_current - ts_tail_previous
                    else:
                        headway_differences[head_id] = ts_head_current - ts_tail_previous - (
                                self.r_leader_acc_dur - ts_tail_remaining)

            # get the max thw
            m_leader, max_thw = max(headway_differences.items(), key=lambda x: x[1])

        # last veh on inflow_highway
        dic_result = {m_leader: [max_thw]}
        return dic_result


    def _get_max_interval_ml(self, step, ls_m_leader_up_asc, ls_m_veh_up_asc):
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
        :return

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
                if head_id == 'm_av3171':
                    pass
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
        dic_r_platoon_travel_time = {key: value + [self.dic_platoon_merge_time_by_size[len(value[0])]]
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
                    cum_rp_pass_time = self.dic_platoon_merge_time_by_size.get(final_rp_number, float('inf'))  # accumulate ramp passing time
                    if cum_rp_pass_time < max_interval + self.delta_t:
                        ls_pass_rid.append(rp_leader)
                        if rp_leader not in self.ls_skip_stop and i != 0:
                            if rp_leader == 'ravh3720' or rp_leader == 'ravh2610':
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

        if self.data_recorder.ms_exit_speed < 19.44:
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
        m_dis = dic_m_leader_info['dis']  # m_leader distance to ws
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


    def _push_if_not_redundant(self, step, value, update_queue, last_value_attr: str):
        """
        Push a value into the specified delay update_queue if it's not redundant.

        :param step: current time step
        :param value: payload to be pushed
        :param update_queue: which delay buffer to use (e.g., self.action_buffer or self.timing_buffer)
        :param last_value_attr: name of attribute to store the last pushed value (e.g., 'last_action_payload')
        :return: maybe_released value from the update_queue
        """
        if value == True:
            pass
        last_value = getattr(self, last_value_attr, None) # last_value_attr = last_value
        # is_redundant = (value == last_value or all(not x for x in value))
        if isinstance(value, (list, tuple)):
            is_redundant = (value == last_value or all(not x for x in value))
        else:
            last_push_step = update_queue.buffer[0][1] if update_queue.buffer else None
            is_redundant = (value is False or step == last_push_step)

        if not is_redundant: # only push when is not redundant => is_redundant = False
            setattr(self, last_value_attr, value) # last_value_attr = value
            update_queue.push(step, value)
        return update_queue.maybe_release(step)

    def _jam_control_clean(self, step, dic_platoon_info, ls_m_leader_up_asc, ls_m_veh_up_asc,
                           dic_vid_groups, ls_r_dep_times, mpc_interval):
        '''
        update 25.1.30
        :param step:
               dic_platoon_info: {'ravh40': [['AHHHHHHHHH', 'rhv130'], deque([p_length1, p_length2], maxlen=10)],
                                    'mavh40': [['AHHHHHHHHH', 'mhv130'], deque([p_length1, p_length2], maxlen=10)]}
               ls_m_leader_up_asc: ['mavh1920']
               ls_m_veh_up_asc: ['mhv693', 'mhv765', 'mhv784', 'mhv795', 'mhv811', 'mav839']
               dic_mplatoon_et: {'mavh40': ['AHHHHHHHHH', 44.57387499999983, 64.42628849831135],
                                'mavh310': ['AHHHHHHHHHHH', 63.3397847329901, 87.53000000000002, 63.1]}
               dic_vid_groups: all veh info
               dic_id_speed: {id1 : [speed1, speed2, ...], id2: [speed1, speed2, ...]}
               ls_r_dep_times: [4, 5, 6, ....]
        :return:
        '''
        self.dic_vid_groups = dic_vid_groups
        prc.print_message('**in jam mode**')
        first_r_leader = self._stop_ramp_fleet3(dic_platoon_info)  # Stop first ramp leader
        self._check_resume_state4(dic_platoon_info) # 241003update
        # Get Ramp Platoon travel Time (from stop to pass intersection)
        dic_r_platoon_travel_time = self._get_ramp_platoon_merge_duration(dic_platoon_info)
        dic_max_interval = self._get_max_interval_upper(step, self.ml, ls_m_leader_up_asc, ls_m_veh_up_asc)
        m_leader, max_interval, final_rp_pass_time = self._compare3(dic_max_interval, dic_r_platoon_travel_time)
        # decide if m_leader need to take action
        dic_m_leader_action_params = self._get_m_leader_action(step, first_r_leader, final_rp_pass_time,
                                                               m_leader, max_interval,
                                                               mpc_interval)
        action_m_leader = self._apply_m_leader_control(step, dic_m_leader_action_params)
        timing = self._find_timing6(step, m_leader, action_m_leader, max_interval, final_rp_pass_time)
        if timing:
            pass
        ls_r_proper = self._get_r_proper()
        # record ramp queue length
        queue_log = self.data_recorder.get_queue_length(step, ls_r_proper, ls_r_dep_times)
        self._restart_ramp_fleet(step, first_r_leader, timing) # Resume r_leader
        return queue_log

    def _jam_control_disturbed(self, step, dic_platoon_info, ls_m_leader_up_asc, ls_m_veh_up_asc,
                               dic_vid_groups, ls_r_dep_times, mpc_interval):
        '''
        add v2x disturbance
        update 25.1.30
        param
            step:
            dic_platoon_info:
            ls_m_leader_up_asc: dic_m_leader_action_params
            ls_m_veh_up_asc:
            dic_vid_groups:
            dic_id_speed:
            ls_r_dep_times:
            mpc_interval:

        return
            action_params: [t_dec, a_dec, t_acc, a_acc, reach_v, t_recorded]

        '''
        c_ts = round(step / 10 + 0.1, 1)
        c_ts % 1 == 0 and prc.print_message('**in jam mode**')
        self.dic_vid_groups = dic_vid_groups
        # Stop r_leader (the first one)
        first_r_leader = self._stop_ramp_fleet3(dic_platoon_info)  # first r_leader id
        self._check_resume_state4(dic_platoon_info) # 241003update
        # Get ramp fleet travel time (from stop to pass intersection)
        dic_r_platoon_travel_time = self._get_ramp_platoon_merge_duration(dic_platoon_info)
        dic_max_interval = self._get_max_interval_upper(step, self.ml, ls_m_leader_up_asc, ls_m_veh_up_asc)
        m_leader, max_interval, final_rp_pass_time = self._compare3(dic_max_interval, dic_r_platoon_travel_time)

        # m_leader take action; m_leader_acting = True/False
        action_params = self._get_m_leader_action(step, first_r_leader, final_rp_pass_time, m_leader,
                                                  max_interval, mpc_interval)  # MPC interval = 6s
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

class ShiftMode:
    def __init__(self, traci, instance_dr, av_p):
        self.regular_mode = True
        self.jam_mode = False
        self.traci = traci
        self.data_recorder = instance_dr
        self.av_p = av_p

        self.length_ih = self.traci.lane.getLength('inflow_highway_0')
        self.length_ws0 = self.traci.lane.getLength('ws_0')

        self.ws1_slow_ts = None
        self.r_leader_past_half_acc_ts = None
        self.mode_trigger_window = 5.0

    def determine_mode4(self, ls_m_veh_up_asc, ls_r_veh_up, ls_r_leader_up):
        '''
        determine_mode_fixed_merge_point

        for fixed merging point scenario
        241122 update: as platoon become longer, 1 lead 7
        params:
            ls_m_veh_up_asc: mainline veh before merging
            ls_r_veh_up: ramp veh before merging
            ls_r_leader_up: list of rav leader (head) before merging

            min_plength:

        :return:
        '''
        rho_jam = 90 # 90 veh/km.lane
        check_length = 100 # the last 100m on mainline
        jam_threshold = rho_jam * check_length / 1000  # → 9 vehicles

        length_ih = self.traci.lane.getLength('inflow_highway_0') # ok
        length_ramp = self.traci.lane.getLength('inflow_merge_0')

        # Count vehicles near the end of each lane
        ls_Mjam_veh = [
            vid for vid in ls_m_veh_up_asc
            if self.data_recorder.dic_pos[vid] >= length_ih - check_length
        ]
        ls_Rjam_veh = [
            vid for vid in ls_r_veh_up
            if self.data_recorder.dic_pos[vid] >= length_ramp - check_length
        ]

        if self.regular_mode and (len(ls_Mjam_veh) >= jam_threshold or len(ls_Rjam_veh) >= jam_threshold):
            # on jam condition
            self.regular_mode = False
            self.jam_mode = True

        if self.jam_mode and len(ls_r_leader_up) < 1 and len(ls_Mjam_veh) < jam_threshold and len(ls_Rjam_veh) < jam_threshold: # new condtion: len(ls_veh_f) < max_jam_vnum
            self.regular_mode = True
            self.jam_mode = False

        return self.regular_mode, self.jam_mode

    def determine_mode_flexible_merge_point(self, ls_wsB_asc, ls_wsA_asc, ls_r_leader_up):
        '''
        for flexible-merging-point scenario
            Determine whether the merging controller should use regular mode or jam mode.
        Jam mode is activated by two conditions:
        1. Severe local density: density >= rho_jam
        2. Dense and slow traffic: density >= rho_warning and average speed <= v_jam
        A hysteresis logic is used for recovery to avoid frequent mode switching.
        params:
            ls_wsB: weaving section veh (from mainline)
            ls_wsA: weaving section veh (from ramp)
            ls_r_leader_up: list of rav leader (head) before merging
        :return:
        '''
        rho_jam = 90 # 90 veh/km.lane
        rho_warning = 80 # 70 veh/km.lane
        v_jam = 5.0 # speed threshold (m/s)
        check_length = 100 # the last 100m on mainline
        jam_threshold = rho_jam * check_length / 1000  # → 9 vehicles
        warning_threshold = rho_warning * check_length / 1000 # 7 vehicles

        # Count vehicles number in the first 100 m on wsA and wsB (check section)
        ls_wsB_check_veh = [
            vid for vid in ls_wsB_asc
            if self.data_recorder.dic_pos[vid] <= check_length
        ] # from mainlane

        ls_wsA_check_veh = [
            vid for vid in ls_wsA_asc
            if self.data_recorder.dic_pos[vid] <= check_length
        ] # from ramp

        speeds_wsB_check = [self.data_recorder.dic_speed[vid] 
                      for vid in ls_wsB_check_veh if vid in self.data_recorder.dic_speed]
        if speeds_wsB_check:
            avg_speed_wsB_check = sum(speeds_wsB_check) / len(speeds_wsB_check)
        else:
            avg_speed_wsB_check = float("inf")

        # Calculate average speed in the ramp detection area
        speeds_wsA_check = [self.data_recorder.dic_speed[vid] 
                      for vid in ls_wsA_check_veh if vid in self.data_recorder.dic_speed]

        if speeds_wsA_check:
            avg_speed_wsA_check = sum(speeds_wsA_check) / len(speeds_wsA_check)
        else:
            avg_speed_wsA_check = float("inf")

        # Condition 1: severe local density
        high_density = (
                len(ls_wsB_check_veh) >= jam_threshold
                or len(ls_wsA_check_veh) >= jam_threshold
        )

        # Condition 2: warning-level density with low speed
        dense_and_slow = (
                 len(ls_wsB_check_veh) >= warning_threshold
                 and avg_speed_wsB_check <= v_jam
         ) or (
                 len(ls_wsA_check_veh) >= warning_threshold
                 and avg_speed_wsA_check <= v_jam
         )

        if self.regular_mode and (high_density or dense_and_slow):
        # if self.regular_mode and high_density:
            # on jam condition
            self.regular_mode = False
            self.jam_mode = True

        ls_veh_c1_0_0 = self.traci.lane.getLastStepVehicleIDs(':c1_0_0')
        num_leader_c1_0_0 = sum(1 for vid in ls_veh_c1_0_0 if 'ravh' in vid) # junction between ramp_proper and wsA
        num_leader_wsA = sum(1 for vid in ls_wsA_asc if 'ravh' in vid)
        num_leader_wsB = sum(1 for vid in ls_wsB_asc if 'ravh' in vid)
        num_leader_ramp_proper = len(ls_r_leader_up)
        num_leader_ramp_ws = num_leader_c1_0_0 + num_leader_wsA + num_leader_wsB + num_leader_ramp_proper

        recover_condition = (
                num_leader_ramp_ws < 1
                and len(ls_wsB_check_veh) < warning_threshold
                and len(ls_wsA_check_veh) < warning_threshold
                and avg_speed_wsB_check > v_jam
                and avg_speed_wsA_check > v_jam
        )

        if self.jam_mode and recover_condition: # new condtion: len(ls_veh_f) < max_jam_vnum
            self.regular_mode = True
            self.jam_mode = False

        return self.regular_mode, self.jam_mode


    def determine_mode_low_sensor_reliance(self, ls_r_leader_wsA_asc,
                                           ls_wsB_av_asc):
        '''
        Determine regular/jam mode with low-sensor inputs.

        Jam mode is entered when both conditions become true within the
        configured trigger window:
        1. Any AV on ws_1 drops below v_jam.
        2. The farthest downstream ramp leader on ws_0 reaches the back half
           of ws_0.

        Jam mode is recovered when the ramp queue has cleared and AV speeds
        on ws_0 and ws_1 are both above v_jam.

        Parameters
        ----------
        ls_m_leader_up_asc: list of leader on inflow_highway_0
        ls_r_leader_wsA_asc: list of ramp leader on ws_0 (or ms)
        ls_wsB_av_asc: list of av on ws_1 (or ms)

        Returns
        -------

        '''
        v_jam = 5.0

        ws1_slow = any(
            self.data_recorder.dic_speed.get(vid, float("inf")) < v_jam
            for vid in ls_wsB_av_asc
        )

        # Ramp leader closest to the merge point.
        # ls_r_leader_wsA_asc is sorted by position, so the first element is the farthest downstream.
        r_leader_past_half_acc = False
        if ls_r_leader_wsA_asc:
            farthest_r_leader = ls_r_leader_wsA_asc[0]
            farthest_pos = self.data_recorder.dic_pos.get(farthest_r_leader)
            if farthest_pos is not None:
                r_leader_past_half_acc = farthest_pos >= self.length_ws0 / 2

        now = self.traci.simulation.getTime()
        if ws1_slow:
            self.ws1_slow_ts = now
        if r_leader_past_half_acc:
            self.r_leader_past_half_acc_ts = now

        ws1_slow_recent = (
            self.ws1_slow_ts is not None
            and now - self.ws1_slow_ts <= self.mode_trigger_window
        )
        r_leader_past_half_recent = (
            self.r_leader_past_half_acc_ts is not None
            and now - self.r_leader_past_half_acc_ts <= self.mode_trigger_window
        )

        ls_veh_c1_0_0 = self.traci.lane.getLastStepVehicleIDs(':c1_0_0')
        ls_r_leader_up = self.data_recorder.dic_vid_groups['ls_r_leader_up']
        num_leader_ramp_proper = len(ls_r_leader_up)
        num_leader_c1_0_0 = sum(1 for vid in ls_veh_c1_0_0 if 'ravh' in vid)
        num_leader_wsA = len(ls_r_leader_wsA_asc)
        num_leader_wsB = sum(1 for vid in ls_wsB_av_asc if 'ravh' in vid)
        num_leader_ramp_ws = num_leader_c1_0_0 + num_leader_wsA + num_leader_wsB + num_leader_ramp_proper

        ws0_av_speeds = [
            self.data_recorder.dic_speed[vid]
            for vid in ls_r_leader_wsA_asc
        ]
        ws1_av_speeds = [
            self.data_recorder.dic_speed[vid]
            for vid in ls_wsB_av_asc
            if vid in self.data_recorder.dic_speed
        ]

        if self.regular_mode and ws1_slow_recent and r_leader_past_half_recent:
            self.regular_mode = False
            self.jam_mode = True

        ls_ihA_av_asc = self.data_recorder.dic_vid_groups['ls_ihA_av_asc']
        mainline_clear = (
                len(ls_ihA_av_asc) == 0
                and not any(vid.startswith("m") for vid in ls_wsB_av_asc)
        )

        normal_recover = (
                num_leader_ramp_ws < 1
                and all(speed > v_jam for speed in ws0_av_speeds)
                and all(speed > v_jam for speed in ws1_av_speeds)
        )

        recover_condition = normal_recover or mainline_clear

        if self.jam_mode and recover_condition:
            self.regular_mode = True
            self.jam_mode = False

        return self.regular_mode, self.jam_mode
