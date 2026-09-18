# state_builder.py

import numpy as np

class StateBuilder:
    """
    State vector constructor for RL-based AV lane change decision.
    Inputs: candidate AV ID and target platoon info.
    Output: normalized state vector (8-dimensional).
    """
    def __init__(self, traci, data_recorder, max_gap=100.0,
                 max_lane_pos=1200.0):
        self.traci = traci
        self.data_recorder = data_recorder

        self.max_speed = data_recorder.max_speed
        self.length_pfz = data_recorder.length_pfz
        self.max_platoon_size = 12  # max_platoon_size

        self.max_gap = max_gap
        self.max_lane_pos = max_lane_pos # take note this number


    def build_state_se(self, av_id: str, leader_id, tail_id,
                       p_info: list) -> np.ndarray:
        """
        split_insert (splitting expert)

        Build the state vector for a given candidate AV.
        1. side-av remainig dis to MCZ
        2. v_side-av
        3. speed diff between leader and side-av
        4. pos diff between leader and side-av
        5. entry time diff between leader and side-av
        6. entry time diff between side-av and platoon-tail

        Parameters:
        - av_id: candidate av (side-av)
        - pMember: members list of oversized platoon
        - p_info: list = [head_pos, tail_pos, avg_speed, size]

        Returns:
        - np.ndarray: normalized state vector with 10 elements
        """
        try:
            # side-AV features
            v_av = self.data_recorder.get_vid_states(av_id)['v']
            pos_av = self.data_recorder.get_vid_states(av_id)['pos'] # dis to the start of lane
            entry_ts_av = int(''.join(filter(str.isdigit, av_id))) / 10 # get entry time from av id
            dis_to_mcz_av = self.length_pfz - pos_av

            # leader features
            v_leader = self.data_recorder.get_vid_states(leader_id)['v']
            pos_leader = self.data_recorder.get_vid_states(leader_id)['pos']
            dis_to_mcz_leader = self.length_pfz - pos_leader
            entry_ts_leader = int(''.join(filter(str.isdigit, leader_id))) / 10

            # tail features
            entry_ts_tail = int(''.join(filter(str.isdigit, tail_id))) / 10

            # Relative speed and position
            delta_v = v_av - v_leader
            delta_dis = dis_to_mcz_av - dis_to_mcz_leader

            # Relative entry time
            delta_ts_av_leader = entry_ts_av - entry_ts_leader
            delta_ts_av_tail = entry_ts_av - entry_ts_tail

            # min travel time and platoon size
            t_pfz = self.length_pfz/self.max_speed
            n = p_info[3]

            # Construct normalised state vector
            state = np.array([
                dis_to_mcz_av / self.length_pfz,
                v_av / self.max_speed,
                delta_v / self.max_speed,
                delta_dis / self.length_pfz,
                delta_ts_av_leader / t_pfz,
                delta_ts_av_tail / t_pfz,
                n / self.max_platoon_size,
            ], dtype=np.float32)

        except Exception as e:
            print(f"[StateBuilder-SE] Failed to extract state for {av_id}: {e}")
            return None

        return state


    def build_state_ce(self, cand_leader: str, target_sparse_platoon,
                       dic_platoon_member) -> np.ndarray:
        """
        Build the seven-feature state vector for the collecting expert (CE).
        1. side-AV remaining distance to MCZ
        2. side-AV instantaneous speed
        3. speed difference between side-AV and target platoon leader
        4. entry-time difference between side-AV and last connected-state vehicle
        5. entry-time difference between side-AV and first free-state vehicle
        6. entry-time difference between side-AV and last free-state vehicle
        7. number of free-state followers in the target platoon


        target_sparse_platoon = {leader_id: [first_free_follower_id, ...]}

        The free-state segment starts at target_sparse_platoon[leader] and
        ends at the target platoon tail. The vehicle immediately before the
        first free-state vehicle is the last connected-state vehicle.
        """
        try:
            leader = next(iter(target_sparse_platoon.keys()))
            first_free_follower = next(iter(target_sparse_platoon.values()))
            platoon_members = dic_platoon_member[leader]
            idx_first_free = platoon_members.index(first_free_follower)
            if idx_first_free == 0:
                raise ValueError(
                    f"First free-state vehicle {first_free_follower} has no preceding vehicle"
                )

            last_constrained_follower = platoon_members[idx_first_free - 1]
            last_free_follower = platoon_members[-1]
            n_free = len(platoon_members) - idx_first_free

            candidate_state = self.data_recorder.get_vid_states(cand_leader)
            leader_state = self.data_recorder.get_vid_states(leader)
            pos_candidate = candidate_state['pos']
            v_candidate = candidate_state['v']
            v_leader = leader_state['v']

            d_candidate_to_mcz = self.length_pfz - pos_candidate
            delta_v_lead = v_candidate - v_leader

            def emerge_time(vehicle_id):
                return int(''.join(filter(str.isdigit, vehicle_id))) / 10

            t_candidate = emerge_time(cand_leader)
            delta_t_pre = t_candidate - emerge_time(last_constrained_follower)
            delta_t_first = t_candidate - emerge_time(first_free_follower)
            delta_t_last = t_candidate - emerge_time(last_free_follower)
            t_pfz = self.length_pfz / self.max_speed

            state = np.array([
                d_candidate_to_mcz / self.length_pfz,
                v_candidate / self.max_speed,
                delta_v_lead / self.max_speed,
                delta_t_pre / t_pfz,
                delta_t_first / t_pfz,
                delta_t_last / t_pfz,
                n_free / self.max_platoon_size,
            ], dtype=np.float32)

        except Exception as e:
            print(f"[StateBuilder-CE] Failed to extract state for {cand_leader}: {e}")
            return None

        return state

    
    def _get_insertion_gap(self, ls_pMember, ls_upA_asc, av_pos):
        '''
        Compute the front and rear gap for a candidate AV insertion position,
        based on platoon members and downstream vehicles on the same lane.
        :param ls_pMember: ['mav635', 'mhv666', 'mhv710', 'mhv735', 'mhv760']
        :param ls_upA: ['mhv960', 'mhv1069', 'mhv1092']
        :return:
            front_gap (float): Distance to the nearest vehicle in front of AV.
            rear_gap (float): Distance to the nearest vehicle behind AV.
        '''
        ls_front_gap = []
        ls_rear_gap = []

        leader_id = ls_pMember[0]

        # Get all vehicles from leader backward (i.e., all vehicles behind the leader in lane order)
        idx_leader = ls_upA_asc.index(leader_id)
        ls_id = ls_upA_asc[idx_leader:]
        for id in ls_id:
            # try:
            #     pos = self.traci.vehicle.getLanePosition(id)
            # except self.traci.TraCIException:
            #     continue
            # pos = self.data_recorder.dic_pos.get(id)
            pos = self.data_recorder.get_vid_states(id)['pos']
            if pos is None:
                continue
            gap = pos - av_pos
            if gap > 0:
                ls_front_gap.append(gap)
            else:  # gap <= 0
                ls_rear_gap.append(gap)
        # Nearest front gap = smallest positive gap
        if not ls_front_gap:
            pass
        front_gap = abs(min(ls_front_gap))
        # Nearest rear gap = smallest (least negative) → largest value
        rear_gap = abs(max(ls_rear_gap)) if ls_rear_gap else av_pos
        return front_gap, rear_gap