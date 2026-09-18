# platoon_oversized_handler.py
# SC1: Handle oversized platoons - find and provide info for RL splitting

class PlatoonOversizedHandler:
    def __init__(self, traci, data_recorder, platoon_basic):
        self.traci = traci
        self.data_recorder = data_recorder
        self.p_basic = platoon_basic

    def find_oversizedP_nearbyAV(self, ls_ihB_av_asc, dic_platoon_size, dic_platoon_members):
        '''
        Identifies oversized platoons and finds nearby side-lane AVs of oversized platoon
        :param
                ls_ihB_av (descending, new av => old av)
                dic_platoon_size: {leader_AV : size, ...}
                dic_platoon_members: {leader_AV : [leader, follower1, follower2, ...], ...}

        :return: dic_oversized_platoon_states => {leader_AV : [head_pos,
                                                               tail_pos,
                                                               avg_speed,
                                                               size]}
                 dic_leader_candidates => {oversized_platoon leader_AV: [outer_candidates_av1, candidates_av2]}
                 all lane_B AV that behind target leader
        '''
        dic_oversized_platoon_states = {}  # oversized platoon
        for leader_id, size in dic_platoon_size.items():
            if size > self.p_basic.max_team_size:
                ls_members = self.p_basic.dic_platoon_members[leader_id]
                tail_id = ls_members[-1]
                head_pos = self.data_recorder.get_vid_states(leader_id)['pos']
                tail_pos = self.data_recorder.get_vid_states(tail_id)['pos']
                avg_speed = sum(self.data_recorder.get_vid_states(vid)['v'] for vid in ls_members) / len(ls_members)
                platoon_states = [head_pos, tail_pos, avg_speed, size]
                dic_oversized_platoon_states[leader_id] = platoon_states
        dic_nonOversized = self._find_non_oversizedP(dic_platoon_members,
                                                    dic_oversized_platoon_states)
        # nearby lane AV list
        dic_leader_candidates = {}
        if not dic_oversized_platoon_states:
            return dic_oversized_platoon_states, dic_leader_candidates, dic_nonOversized
        leaders = list(dic_platoon_members)
        for leader_id, platoon_states in dic_oversized_platoon_states.items():
            if leader_id == 'm_av1202':
                pass
            pos_leader = platoon_states[0]
            leader_idx = leaders.index(leader_id)
            next_leader_id = leaders[leader_idx + 1] if leader_idx + 1 < len(leaders) else None
            next_leader_pos = self.data_recorder.dic_pos.get(next_leader_id) if next_leader_id else None
            for index, outer_lane_av in enumerate(ls_ihB_av_asc):
                pos_outer_av = self.data_recorder.get_vid_states(outer_lane_av)['pos']
                if pos_outer_av < pos_leader:
                    av_candidates = ls_ihB_av_asc[index:]
                    if next_leader_pos is not None:
                        av_candidates = [av for av in av_candidates
                                         if next_leader_pos <= self.data_recorder.dic_pos.get(av, -1) <= pos_leader]
                    dic_leader_candidates[leader_id] = av_candidates # dic_target_leader_av_candidates
                    break
        return dic_oversized_platoon_states, dic_leader_candidates, dic_nonOversized

    def _find_non_oversizedP(self, dic_platoon_members, dic_oversized_platoon_states):
        '''
        get current non oversized platoon
        :param dic_platoon_members: all current platoon
               dic_oversized_platoon_states: all current oversized platoon
        :return: dic_nonOversized = {leader: [leader, follower1, follower2, ...], ...}
        '''
        # oversized_leader = list(dic_oversized_platoon_states.keys())
        # all_leader = list(dic_platoon_members.keys())
        # non_oversized_leader = list(set(all_leader) - set(oversized_leader))
        oversized_leader = set(dic_oversized_platoon_states)
        all_leader = list(dic_platoon_members)
        non_oversized_leader = [
            leader
            for leader in all_leader
            if leader not in oversized_leader
        ]
        dic_nonOversized = {k: dic_platoon_members[k] for k in non_oversized_leader}
        return dic_nonOversized
