# platoon_basic.py
# Core platoon management operations: tagging, size tracking, speed control, recording

import os
import re
import joblib
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


class PlatoonBasic:
    def __init__(self, traci, data_recorder, pass_recorder, max_team_size=11):
        self.traci = traci
        self.data_recorder = data_recorder
        self.pass_recorder = pass_recorder
        # Load Random Forest model for follower state prediction
        # fs_model_name = 'follower_state_prediction_model_251121_ndarray.pkl'
        fs_model_name = 'follower_state_prediction_model_260715_ndarray.pkl'
        # fs_model_name = 'follower_state_prediction_model_260829_ndarray_final.pkl'
        self.fs_model = joblib.load(
            os.path.join(project_root, 'rf_models', fs_model_name))

        self.dic_pass_time = self.pass_recorder.dic_pass_time
        self.max_speed = self.data_recorder.max_speed # 27.78 m/s => 100km/h
        self.speed_level3 = 25 # 
        self.speed_level2 = 22.22  # 19.44 m/s (70km/h);
        self.speed_level1 = 16.67  # 16.67 m/s (60km/h)
        self.max_team_size = max_team_size

        self.dec_av = []
        self.dic_tags = {}
        self.recover_speed_map = {}
        self.set_speed_ok = set()  # av_id that speed restore back to max (27.78 m/s)
        self.set_speed_level3 = set()
        self.dic_platoon_size = {}  # all leaderAV and its platoon size
        self.dic_platoon_members = {}  # all leaderAV and its members

        self.dic_AVroleChange = {}  # dic_AVroleChange = {AV_id: type, ...} record AV changed its role
        self.set_leader_fol_states_checked = set()
        self.set_leader_fol_states_checked_sensor = set() # sensor measurement version
        self.ls_leader_AV = []
        self.ls_follower_AV = []
        self.ls_ihA_lastStep = []  # ls_upA (upstream AV) last Step

        self.dic_follower_state = {}  # record the predicted following state (algo application); free_mode/following_mode
        self.dic_follower_state_sensor = {} # record following state based on sensor measurement (for evaluation use)

        self.dic_final_platoon_info = {}  # m_dpt_type
        self.dic_id_features = {}  # {id:[f1, f2,..., ], ...}
        self.dic_id_preState = {}  # Predicted state of each follower
        # Track last leader for each follower to detect leader changes
        self.dic_fol_last_leader = {}
        self.dic_leader_free_triggered = {}

    def get_platoon_size3(self, ls_ihA_asc, ls_leader):
        '''
        get the platoon size/platoon members for each LEADER currently on the road
        record each platoon members
        :param ls_ihA: ['mhv3171', 'mav3287', 'mhv3305',  ...] ascending, vehicle list on inflow_highway outer
        :param ls_leader: all leaderAV CURRENTLY on upstream_0, ['mav2744', 'mav3038'], ascending order
        :return: dic_id_size = {'mav158': 10, 'mav44': 4, 'mav661': 2},
                        current ls_leader and its corresponding size
                 dic_platoon_members = {
                                        'mav19': ['mav19', 'mhv46', 'mhv64', 'mhv74', 'mhv86'],
                                        "veh_2": ["veh_2", "veh_4"]
                                        }, current ls_leader and its corresponding members
                 self.dic_platoon_members (all history platoon and platoon members)
        '''
        dic_leaderAV_index = {}  # leader_id and its index in the traffic; {'mav2744': 1, 'mav3038': 8}
        dic_platoon_size = {}
        dic_platoon_members = {}

        for leader in ls_leader:
            idx_ih = ls_ihA_asc.index(leader)  # idx on inflow_highway_0 (outer)
            dic_leaderAV_index[leader] = idx_ih

        # get leader's corresponding size
        leaders = list(dic_leaderAV_index.keys())
        # index: one position; indices: multiple positions
        indices = list(dic_leaderAV_index.values())
        for i in range(len(indices)):
            start = indices[i]
            end = indices[i + 1] if i < len(indices) - 1 else len(ls_ihA_asc)
            platoon = ls_ihA_asc[start:end]
            dic_platoon_size[leaders[i]] = len(platoon)
            dic_platoon_members[leaders[i]] = platoon
            self.dic_platoon_size[leaders[i]] = len(platoon)
            self.dic_platoon_members[leaders[i]] = platoon
        # record leaderAV on mainline
        self.data_recorder.ls_m_leader_his_asc = list(self.dic_platoon_size.keys())
        return self.dic_platoon_size, dic_platoon_size, dic_platoon_members

    def tag_vehicles13(self, ls_ihA_asc, max_team_size=11, enable_lhr=True):
        '''
        260301 crucial update, only keep dic_tags for vehicles still on current road network

        label each vehicle: 0 => follower_HV, 1 => leader_AV, 2 => follower_AV;
        also for split_promote (promote a follower_AV to leader_AV to split oversized platoon)

        :param ls_ihA: vehicle list ordered from newest to oldest (descending)
               ls_ihA_asc: oldest to newest
        :param max_team_size: maximum allowed platoon size
        :return: updated dic_tags,
                 ls_leader_AV, ascending order
                 ls_follower_AV
        '''
        self.max_team_size = max_team_size

        if self.ls_ihA_lastStep != ls_ihA_asc:
            old_dic_tags = self.dic_tags.copy()
            # === get new_dic, make sure the order is correct ===
            # find the first id in ls_ihA_asc that already exists in dic_tags
            anchor_id = next((id for id in ls_ihA_asc if id in self.dic_tags), None)
            # Keep only the part of dic_tags before the anchor_id (maintain order)
            new_dic = {}
            for k in self.dic_tags:
                if k == anchor_id:
                    break
                new_dic[k] = self.dic_tags[k]
            # Extend new_dic with all IDs from ls_ihA_asc, assigning temporary value -1
            # This overwrites existing keys and inserts new ones in the given order
            for id in ls_ihA_asc:
                new_dic[id] = -1
            self.dic_tags = new_dic

            # === tag every id in ls_ihA_asc ===
            current_leader = None
            current_team_size = 0

            for i, id in enumerate(ls_ihA_asc):
                if id == 'm_av1489':
                    pass
                if 'av' in id:
                    # Reconstruct leader and team size based on previous tagging
                    ls_keys = list(self.dic_tags.keys())
                    this_index = ls_keys.index(id)
                    n = 0
                    # range(start, stop, step); step = -1 (count backward)
                    for idx in range(this_index - 1, -1, -1):
                        n += 1
                        tagged_id = ls_keys[idx]
                        if self.dic_tags.get(tagged_id) == 1:
                            current_leader = tagged_id
                            current_team_size = n + 1
                            break

                    # === Proactive split: check if too many HVs follow this AV ===
                    too_many_hv_behind = False
                    if (current_leader is not None) and (current_team_size <= max_team_size):
                        hv_count = 0
                        remaining_slots = max_team_size - current_team_size
                        for j in range(i + 1, len(ls_ihA_asc)):
                            id_next = ls_ihA_asc[j]
                            if 'hv' in id_next:
                                hv_count += 1
                            else:
                                break
                        if hv_count > remaining_slots:
                            too_many_hv_behind = True

                    # === Assign AV as leader if needed ===
                    if ((current_leader is None) or (current_team_size > max_team_size)
                            or (id in self.dic_AVroleChange)):
                        self.dic_tags[id] = 1  # Mark as leader
                        current_leader = id
                        current_team_size = 1
                    elif enable_lhr and too_many_hv_behind:
                        if 'b' in id:
                            self.dic_AVroleChange[id] = 'split_insert'
                        else:
                            self.dic_AVroleChange[id] = 'split_promote'
                        self.dic_tags[id] = 1  # Mark as leader
                        current_leader = id
                        current_team_size = 1
                    else:
                        # new added
                        if old_dic_tags.get(id) == 1 and current_leader in self.dic_AVroleChange:
                            self.dic_tags[id] = 1
                            current_leader = id
                            current_team_size = 1
                        else: # Assign AV as follower
                            self.dic_tags[id] = 2
                            if id == 'm_av1489':
                                pass
                            removed1 = self.dic_platoon_members.pop(id, None)
                            removed2 = self.dic_platoon_size.pop(id, None)
                            current_team_size += 1
                else:
                    # HV vehicle
                    self.dic_tags[id] = 0
                    if current_leader is not None:
                        current_team_size += 1

            self.ls_ihA_lastStep = ls_ihA_asc

            # filter out vehicles that have left the road network
            ls_vehid = self.data_recorder.dic_vid_groups['ls_vehid'] # all vehicle in this step
            self.dic_tags = {k: v for k, v in self.dic_tags.items() if k in ls_vehid}

            # Update leader and follower lists for current control section
            dic_leader_AV = {k: v for k, v in self.dic_tags.items() if v == 1}
            dic_follower_AV = {k: v for k, v in self.dic_tags.items() if v == 2}

            dic_leader_AV_c = {k: v for k, v in dic_leader_AV.items() if k in ls_ihA_asc}
            dic_follower_AV_c = {k: v for k, v in dic_follower_AV.items() if k in ls_ihA_asc}

            self.ls_leader_AV = list(dic_leader_AV_c.keys())
            self.ls_follower_AV = list(dic_follower_AV_c.keys())

        return self.dic_tags, self.ls_leader_AV, self.ls_follower_AV, self.dic_AVroleChange

    def form_platoon3(self, ls_vehid, ls_leader_av, ls_follower_av):
        '''
        based on desire_v to determining leader_AV, and leader_AV decrease to create gaps between platoons
        :param ls_leader_av: the list of leader_AV (asc; small=>large)
        :param ls_follower_av: the list of follower_AV
        :return:
            110 km/h = 30.06 m/s
            100 km/h = 27.78 m/s
            90 km/h = 25.00 m/s
            80 = 22.22
            70 = 19.44
            60 = 16.67
        '''
        min_dis = 150 # 100
        speed_level2 = self.speed_level2  # 22.22m/s => 80km/h; 19.44m/s => 70km/h
        speed_level1 = self.speed_level1  # 16.67m/s => 60km/h

        # make sure all veh in ls_leader_av and ls_follower_av in ls_vehid
        ls_leader_av = [id for id in ls_leader_av if id in ls_vehid]
        ls_follower_av = [id for id in ls_follower_av if id in ls_vehid]
        ls_second_decAV = []
        for id in ls_follower_av:
            self.traci.vehicle.setColor(id, (255, 0, 0, 255))  # red
            # current_max_ori = self.traci.vehicle.getMaxSpeed(id)
            current_max = self.data_recorder.get_vid_states(id)['max_speed']
            if current_max < self.data_recorder.max_speed:
                self.traci.vehicle.setMaxSpeed(id, self.data_recorder.max_speed)
            if id in self.dec_av: self.dec_av.remove(id)

        for leader in ls_leader_av:
            self.traci.vehicle.setColor(leader, (255, 255, 0, 255))  # yellow
            # platoon space control
            if leader not in self.dec_av: # av in max speed_level2 (22.22)
                if leader == 'm_av2847':
                    pass
                self.traci.vehicle.setMaxSpeed(leader, speed_level2)
                self.dec_av.append(leader)

        ls_ihA_av_mcz_asc = self.data_recorder.dic_vid_groups['ls_ihA_av_mcz_asc']
        ls_leader_av = [leader for leader in ls_leader_av if leader not in ls_ihA_av_mcz_asc]
        # check if any leader av is very close to it's preceding vehicle
        for index, leader in enumerate(ls_leader_av):  # ascending order
            preceding_veh_info = self.traci.vehicle.getLeader(leader)
            if preceding_veh_info is not None:
                dis_to_pv = preceding_veh_info[1]
                speed_leader = self.data_recorder.get_vid_states(leader)['v']
                # current_max_speed_ori = self.traci.vehicle.getMaxSpeed(leader)
                current_max_speed = self.data_recorder.get_vid_states(leader)['max_speed']
                if dis_to_pv < min_dis and current_max_speed != speed_level1 and speed_leader <= speed_level2:
                    '''
                    if find LEADER very close to its preceding veh, 
                    this LEADER (and other LEADER after this LEADER) need to 
                    take a second-level dec action (to level_1 speed)
                    '''
                    ls_second_decAV = ls_leader_av[index:] # check order
                    break
        self._set_hold_speed2(ls_second_decAV, speed_level1, min_dis)

    def restore_speed_limit2(self, ls_av):
        '''
        restor av max_speed to 27.78 m/s
        max_speed = 27.78
        :param ls_av: list of AV IDs on weaving section B
        :return:
        '''

        for vid in ls_av:
            if vid == 'm_av2847':
                pass
            if vid in self.set_speed_ok:
                continue
            current_max = self.traci.vehicle.getMaxSpeed(vid)
            if current_max >= self.max_speed:
                self.set_speed_ok.add(vid)
                continue
            self.traci.vehicle.setMaxSpeed(vid, self.max_speed)  # restore to 27.78 m/s

    def restore_speed_limit3(self, step, ls_leader, ls_m_leader_up_asc, dic_id_preState):
        """
        Restore to level3 speed for platoon leaders based on their spatial location.

        Control Strategies:
        1. Merging Control Zone: Unconditional speed restoration.
        2. Platoon Formation Zone: Conditional speed restoration.
           - Condition A: All own followers are in 'following_mode'.
           - Condition B: ALL preceding leaders (downstream) have successfully restored speed.
                          (If a front leader waits, all upstream/subsequence leaders must also wait).

        Parameters
        ----------
        ls_leader : List of all leaders on inflow_highway_0. ascending order
        ls_m_leader_up_asc : List of leaders that have already entered the merging control section.
        self.dic_platoon_members = {'mav19': ['mav19', 'mhv46', 'mhv64'],
                                    'veh_2': ['veh_2', 'veh_4']}
        """
        speed_level3 = self.speed_level3  # 25 m/s => 90km/h
        leaders_on_merging_control = set(ls_m_leader_up_asc)

        # CHAIN REACTION FLAG:
        # If any downstream leader fails to accelerate (waiting for followers),
        # this becomes True and blocks ALL subsequent leaders behind it.
        front_blocked = False

        # NOTE: Ensure ls_leader is ordered from FRONT to BACK (Downstream to Upstream).
        for i, leader in enumerate(ls_leader):
            # If already at level3 speed, it doesn't block anyone behind it. Skip.
            if leader in self.set_speed_level3:
                continue

            current_max = self.traci.vehicle.getMaxSpeed(leader)
            # if current_max >= level3_speed and leader in self.dec_av:
            if current_max >= speed_level3 and leader in self.dec_av:
                self.set_speed_level3.add(leader)
                continue

            # STRATEGY A: Merging Zone
            if leader in leaders_on_merging_control:
                if leader == 'mb_av5939':
                    pass
                # Unconditional acceleration to level3 speed for merging zone
                self.traci.vehicle.setMaxSpeed(leader, speed_level3)

            # STRATEGY B: Formation Zone
            else: # Check if blocked by a leader ahead
                if front_blocked:
                    # A leader ahead is waiting for its followers.
                    # This leader MUST wait too, regardless of its own platoon state.
                    continue

                # Check if this is the FURTHEST UPSTREAM leader (the newest one emerged)
                is_newest_leader = (i == len(ls_leader) - 1)
                if is_newest_leader:
                    # it MUST wait because more followers might emerge behind it.
                    front_blocked = True  # Block state
                    continue

                ls_followers = self.dic_platoon_members.get(leader, [])[1:]

                # Single vehicle (no followers): accelerates immediately
                if not ls_followers:
                    if leader == 'mb_av5939':
                        pass
                    self.traci.vehicle.setMaxSpeed(leader, speed_level3)
                    continue

                # Platoon integrity check
                all_following = True
                for fol in ls_followers:
                    if fol in ['m_hv_cons5973', 'm_hv_agg6067']:
                        pass
                    state = dic_id_preState.get(fol)

                    # features = self._get_RFfeatures2(fol)
                    # features[0][0] = speed_level3 # use level3 speed as input for prediction; as once leader accelerates, followers state may change
                    # state = self.fs_model.predict(features)

                    if state in ('free_mode', 0):
                        all_following = False
                        break

                if all_following and leader not in self.recover_speed_map.keys():
                    if leader == 'mb_av422':
                        pass
                    # Platoon is intact, leader accelerates
                    # self.traci.vehicle.setMaxSpeed(leader, speed_level3)
                    for fol in ls_followers: # record followers's state as '1' (following mode)
                        self.dic_follower_state[fol] = ['following_mode', leader]
                    self.set_leader_fol_states_checked.add(leader) # record this leader then no need to check its followers' state
                    # record final platoon information and pass to merging controller for later use
                    self._get_final_platoon_info(step, self.dic_follower_state)
                else:
                    # PLATOON BROKEN! This leader cannot accelerate.
                    # TRIGGER CHAIN REACTION: Block all subsequent leaders behind this one!
                    front_blocked = True

    def get_cor_leader(self, follower_id):
        '''
        according to dic and follower_id, find its leader_id,
        and the follower's position index in that leader's follower list.
        :param self.dic_platoon_members = {leader: [leader, follower1, follower2,...],...}}
        :param follower_id:
        :return:
        '''
        for leader, followers in self.dic_platoon_members.items():
            if follower_id in followers:
                index = followers.index(follower_id)
                return leader, index
        return None, None

    def update_member_to_leader(self, dic_platoon_members):
        """Build a reverse mapping from follower ID to its platoon leader.
        dic_member_to_leader = {follower_id: leader_id,...}
        """
        dic_member_to_leader = {}
        for leader_id, members in dic_platoon_members.items():
            for veh_id in members:
                dic_member_to_leader[veh_id] = leader_id
        return dic_member_to_leader

    def record_follower_state2(self, step, dic_id_preState):
        """
        original name "record_follower_state2"
        When an AV leader enters the merging control section for the first time,
        record the current states (free_mode / following_mode) of all its platoon followers.

        :param
        :return: self.dic_follower_state: {follower_id: [state, leader_id]}
                 self.dic_final_platoon_info: {66: 'AHHHHHHH', 138: 'AHHHHHH', 174: 'AH', 177: 'AHH'}
        """
        # 1. Leaders that have reached the merging control section (> length_pf)
        ls_mc_leaders = self.data_recorder.dic_vid_groups['ls_m_leader_up_asc']
        # No leader in the merging control section
        if not ls_mc_leaders:
            return self.dic_follower_state, self.dic_final_platoon_info
        # Take the most recently arrived leader
        leader_mc_newest = ls_mc_leaders[-1]
        # Ensure this leader is recorded only once
        if leader_mc_newest in self.set_leader_fol_states_checked:
            return self.dic_follower_state, self.dic_final_platoon_info
        self.set_leader_fol_states_checked.add(leader_mc_newest)
        # 2. Retrieve all followers belonging to this leader's platoon
        platoon_followers = self.dic_platoon_members.get(leader_mc_newest, [])[1:]
        # 3. Record the state of each follower at the moment the leader enters 800m
        free_mode_detected = False # detect any free_mode fol, then all fol (same leader) behind it are in free_mode
        for fol in platoon_followers:
            state_pre = dic_id_preState.get(fol)
            state_pre = 1 # ingore following state as in MCZ leader will continue to collect followers
            state = state_pre
            if free_mode_detected or state in ('free_mode', 0):
                state = 'free_mode'
                free_mode_detected = True
            self.dic_follower_state[fol] = [state, leader_mc_newest]
        self.data_recorder.dic_follower_state = self.dic_follower_state
        # record final platoon information
        self._get_final_platoon_info(step, self.dic_follower_state)
        return self.dic_follower_state # change to leftmost lane and keep until the end of the road

    def record_follower_state_by_sensor(self):
        """
        Call traci to get precise follower states to evaluate the following states as platoon formation
        performance indicator;

        When an AV leader enters the merging control section for the first time,
        record the current states (free_mode / following_mode) of all its platoon followers.

        :return: self.dic_follower_state_sensor: {follower_id: [state, leader_id]}
        """
        # 1. Leaders that have reached the merging control section (> length_pf)
        ls_mc_leaders = self.data_recorder.dic_vid_groups['ls_m_leader_up_asc']
        # No leader in the merging control section
        if not ls_mc_leaders:
            return self.dic_follower_state_sensor
        # Take the most recently arrived leader
        leader_mc_newest = ls_mc_leaders[-1]
        # Ensure this leader is recorded only once
        if leader_mc_newest in self.set_leader_fol_states_checked_sensor:
            return self.dic_follower_state_sensor
        self.set_leader_fol_states_checked_sensor.add(leader_mc_newest)
        # 2. Retrieve all followers belonging to this leader's platoon
        platoon_followers = self.dic_platoon_members.get(leader_mc_newest, [])[1:]
        # 3. Record the state of each follower at the moment the leader enters 800m
        free_mode_detected = False # detect any free_mode fol, then all fol (same leader) behind it are in free_mode
        ls_decision_info = []
        for fol in platoon_followers:
            decision_info = self._check_state(fol)  # Determine free_mode or following_mode
            state = decision_info[-1] if decision_info is not None else None
            if free_mode_detected or state in ('free_mode', 0):
                state = 'free_mode'
                free_mode_detected = True # chain reaction trigger: all followers behind this one are in free_mode
            # fol_id, v_ego, actual_gap, s_headway_thresh, following_state_receding, following_state_leader
            decision_info.append(state)
            ls_decision_info.append(decision_info)
            self.dic_follower_state_sensor[fol] = [state, leader_mc_newest]
        return self.dic_follower_state_sensor

    def set_follower_color(self):
        '''
        set followers color as light green

        dic_follower_state = {follower_id: [state, leader_id]}
        state = free_mode/following_mode

        av_leader: yellow, av_follower: red
        action_av: flashing between yellow and red
        hv_follower: green, hv_free: white
        '''
        light_green = (144, 238, 144) # follower's color
        dic_follower_state = self.dic_follower_state
        try:
            ls_ihAB_hv = self.data_recorder.dic_vid_groups['ls_ihAB_hv']
            for vid in ls_ihAB_hv:
                if self.traci.vehicle.getColor(vid) == light_green:
                    continue
                item = dic_follower_state.get(vid)
                following_state = item[0] if item else None
                if following_state == 'free_mode':
                    self.traci.vehicle.setColor(vid, light_green)  # light green; green (0, 255, 0)
        except:
            pass

    # TODO: Optimise prediction efficiency—consider batch processing or caching leader lookups to reduce computational burden when processing many HV followers
    def predict_flw_state(self, dic_id_type, ls_vehid, model=False):
        """
        Updated platoon-wise prediction:
        - Add per-follower last leader mapping in `self.dic_fol_last_leader`.
        - Add per-leader free cascade flag in `self.dic_leader_free_triggered`.
        - Scan vehicles newest -> oldest and (re)predict any follower that is new or whose leader changed.

        Params:
        - dic_id_type/dic_tags
        - ls_vehid: tuple, all vehicle in this step
        - self.dic_id_features: {id: [f1, f2, ...,], ...} # update, add fol's leader
        - dic_fol_last_leader: {follower: last_leader, ...}
        - self.dic_id_preState: {1: following_mode, 0: free_mode}
        """
        if not dic_id_type:
            return self.dic_id_preState, self.dic_id_features
        # Clean / reset when a vehicle becomes leader
        for vid, tag in dic_id_type.items():
            if tag == 1:
                self.dic_fol_last_leader.pop(vid, None)
                self.dic_id_features.pop(vid, None)  # removes id from the dic; returns None if id doesn't exist.
                self.dic_id_preState.pop(vid, None)  # removes the prediction state for id.
                self.dic_leader_free_triggered[vid] = False

        # Process newest -> oldest
        items = list(dic_id_type.items())
        for vid, tag in items:
            if vid == 'm_hv_mean3672':
                pass
            # only handle followers (both AV and HV)
            if tag == 1:
                continue

            # if left network, remove records so can re-evaluate later
            if vid not in ls_vehid:
                self.dic_fol_last_leader.pop(vid, None)
                # self.dic_id_features.pop(vid, None)
                self.dic_id_preState.pop(vid, None)
                continue

            # find current leader for this follower
            leader_id, _ = self.get_cor_leader(vid)
            if leader_id is None:
                # no leader mapping now; clear last leader so it will be retried later
                self.dic_fol_last_leader.pop(vid, None)
                continue

            # skip if already predicted for this leader
            if vid in self.dic_fol_last_leader and self.dic_fol_last_leader[vid] == leader_id:
                continue

            # extract features (this also stores features in self.dic_id_features)
            # v_leader, dis_leader_to_MCZ, n_veh_between, time_headway_to_leader
            arr_select_features = self._get_RFfeatures2(vid)
            if arr_select_features is None:
                # missing data now; clear last-leader to try again later
                self.dic_fol_last_leader.pop(vid, None)
                continue
            # record mapping that this follower was evaluated for this leader
            self.dic_fol_last_leader[vid] = leader_id

            # perform prediction if requested
            if model:
                # respect per-leader free cascade
                if self.dic_leader_free_triggered.get(leader_id, False):
                    pre_state = 0
                else:
                    pre_state = int(self.fs_model.predict(arr_select_features)[0])
                self.dic_id_preState[vid] = pre_state
                # if this follower is free, subsequent followers in same platoon are free
                if pre_state == 0:
                    self.dic_leader_free_triggered[leader_id] = True

        return self.dic_id_preState, self.dic_id_features

    def _get_RFfeatures2(self, new_follower_id):
        '''
        current selected features: v_leader, dis_leader_to_MCZ, n_veh_between, time_headway_to_leader
        previous selected features: dis_to_pv, v_pv, dis_to_leaderAV, veh_num
        get Random Forest features of new_follower_id

        :return: df_select_features: v_leader, dis_leader_to_MCZ, n_veh_between, time_headway_to_leader
        '''
        # get its leader_AV id and index of this veh (COResponding)
        if new_follower_id == 'm_hv_agg6067':
            pass
        leader_id, index_this = self.get_cor_leader(new_follower_id)
        if leader_id is None:
            return None  # No leader found, skip
        # num_veh_between_ego_and_leaderAV; how many veh between this veh and its leader_AV
        n_veh_between = index_this + 1
        # get pos of leader_id
        if leader_id in self.traci.vehicle.getIDList():
            pos_leader = self.data_recorder.get_vid_states(leader_id)['pos']
        else:
            pos_leader = self.data_recorder.length_pfz
        # leader speed
        v_leader = self.data_recorder.get_vid_states(leader_id)['v']
        # leader dis to MCZ
        dis_leader_to_MCZ = self.data_recorder.length_pfz - pos_leader
        # pass time difference
        this_pass_time = self.dic_pass_time['pfz_entry'].get(new_follower_id)
        if this_pass_time is None:
            return None
        leader_pass_time = self.dic_pass_time['pfz_entry'].get(leader_id)

        if leader_pass_time is None:
            fol_id, _ = self.traci.vehicle.getFollower(leader_id, 1e6)
            fol_pass_time = self.dic_pass_time['pfz_entry'].get(fol_id)
            if fol_pass_time is None: # based on leader's preceding vehicle to estimate leader's pass time
                prev_id, gap_leader_prev = self.traci.vehicle.getLeader(leader_id, 1e6)
                th_leader_prev = (gap_leader_prev+5+2.5)/ v_leader # time headway between leader and its preceding vehicle
                prev_pass_time = self.dic_pass_time['pfz_entry'].get(prev_id)
                if prev_pass_time is None:
                    prev_pass_time = int(''.join(filter(str.isdigit, prev_id))) / 10
                    pass
                leader_pass_time = prev_pass_time + th_leader_prev
            else: # based on leader's following vehicle to estimate leader's pass time
                v_fol = self.data_recorder.get_vid_states(fol_id)['v']
                _, gap_fol_leader = self.traci.vehicle.getLeader(fol_id, 1e6)
                th_fol_leader = (gap_fol_leader+5+2.5) / v_fol # time headway between fol and leader
                leader_pass_time = fol_pass_time - th_fol_leader
            # Store virtual pass time for inserted leader
            self.dic_pass_time['pfz_entry'][leader_id] = leader_pass_time

        time_headway_to_leader = this_pass_time - leader_pass_time

        features = [v_leader, dis_leader_to_MCZ, n_veh_between, time_headway_to_leader]
        if new_follower_id == 'mhv66':
            pass
        self.dic_id_features[new_follower_id] = features
        arr_select_features = np.array(features, dtype=float).reshape(1, -1)
        return arr_select_features

    def _set_hold_speed2(self, ls_second_decAV, set_v, gap_threshold):
        """
        Apply a temporary speed limit and recover it based on gap conditions,
        with a leader-first recovery constraint.

        Description:
            Leads are temporarily assigned a reduced maximum speed //set_v (speed_level1): 16.67.
            A leader can recover its original speed only when:
                1. The gap to its preceding vehicle exceeds a predefined threshold.
                2. Its leader has already recovered (or is not under control). ??

            This ensures a front-to-back recovery order and avoids gap shrinkage
            caused by rear vehicles accelerating earlier than their leaders.

        Parameters:
            ls_second_decAV (list): list of leaders need to down to speed_level1 (ordered from front to back)
            set_v (float): temporary speed limit (speed_level1)
            gap_threshold (float): minimum gap required for speed recovery (meters)

        Internal state:
            self.recover_speed_map = {vid: original_speed, ...}
        """

        # Step 1: Collect all currently controlled vehicles
        controlled_ids = list(self.recover_speed_map.keys())
        # fileter out vehicles that left
        ls_ihA_av_asc = self.data_recorder.dic_vid_groups['ls_ihA_av_asc'] # all av in inflow_highway_0
        controlled_ids = [vid for vid in controlled_ids if vid in ls_ihA_av_asc]

        # Step 2: Sort vehicles from front to back based on lane position
        sorted_vids = sorted(
            controlled_ids,
            key=lambda vid: self.data_recorder.get_vid_states(vid)['pos'],
            reverse=True
        )
        recovered_set = set()  # vehicles recovered in this step

        # Step 3: Check recovery condition with leader-first constraint
        for i, vid in enumerate(sorted_vids):
            ori_v = self.recover_speed_map[vid]

            # Find the nearest controlled AV leader ahead.
            front_controlled_leader = sorted_vids[i - 1] if i > 0 else None
            # Enforce AV-leader-first recovery:
            # if the front controlled AV leader has not recovered, this leader can't recover.
            if (front_controlled_leader is not None
                and front_controlled_leader not in recovered_set):
                continue

            # Retrieve preceding vehicle information (vehicle_id, gap)
            prev_veh_info = self.traci.vehicle.getLeader(vid, 99999)
            if prev_veh_info is not None:
                prev_veh, gap = prev_veh_info
                if gap < gap_threshold:
                    continue
            if vid == 'm_av2847':
                pass
            self.traci.vehicle.setMaxSpeed(vid, ori_v)
            recovered_set.add(vid)

        # Step 4: Remove recovered vehicles from tracking map
        for vid in recovered_set:
            if vid in self.recover_speed_map:
                if vid == 'm_av2847':
                    pass
                del self.recover_speed_map[vid]

        # Step 5: Apply speed_level1 to new leaders
        for vid in ls_second_decAV:
            if vid not in self.recover_speed_map:
                if vid == 'm_av2847':
                    pass
                # Store original max speed
                ori_v = self.traci.vehicle.getMaxSpeed(vid)
                # Apply temporary speed constraint
                self.traci.vehicle.setMaxSpeed(vid, set_v)
                # Save for future recovery
                self.recover_speed_map[vid] = ori_v


    def _check_state(self, veh_id):
        """
        Check whether a vehicle is in free-flow mode or in a coupled following mode.

        This following state check is for model training and indicator obtaining, therefore,
        Traci has been used to gain all related information, including hv driving style
        """
        # gap tolerance factor # mhv1174
        tolerance_factor = 2 # 2.0

        # obtain id category
        veh_type = re.sub(r".*_([A-Za-z]+)[0-9]+$", r"\1", veh_id)
        jam_dis, t_headway, accel_ego, _   = self._extract_vehicle_params(veh_type)

        # get id speed
        p_veh_info = self.traci.vehicle.getLeader(veh_id)
        if p_veh_info is None:
            return "free_mode"
        pre_vid, gap = p_veh_info
        _, _, _, decel_pre = self._extract_vehicle_params(pre_vid)
        actual_gap = jam_dis + gap
        v_ego = self.data_recorder.get_vid_states(veh_id)['v']

        # get space headway threshold
        s_headway_thresh = jam_dis + tolerance_factor*t_headway*v_ego
        decision_info = [veh_id, v_ego, actual_gap, s_headway_thresh]

        if actual_gap <= s_headway_thresh:
            follow_state = 'following_mode'
        else:
            follow_state = 'free_mode'
        decision_info.append(follow_state)
        return decision_info

    def _extract_vehicle_params(self, veh_type: str) -> tuple[float, float]:
        if veh_type == 'av':
            jam_dis, t_headway, acc, dec = 1.5, 0.5, 2.6, 4.5 # original
            # jam_dis, t_headway, acc, dec = 2.0, 1.0, 2.6, 4.5 # test
        elif veh_type == 'agg':
            jam_dis, t_headway, acc, dec = 2.0, 0.6, 3.2, 5.0
        elif veh_type == 'mean':
            jam_dis, t_headway, acc, dec = 2.5, 1.0, 2.6, 4.5
        elif veh_type == 'cons':
            jam_dis, t_headway, acc, dec = 3.0, 2.0, 2.2, 4.0
        else:  # HV, no driving style consideration
            jam_dis, t_headway, acc, dec = 2.5, 1.0, 2.6, 4.5
        return jam_dis, t_headway, acc, dec

    def _get_final_platoon_info(self, step, dic_follower_state):
        """
        Count ONLY the followers belonging to the newest AV leader in dic_follower_state.
        Ignore all previous platoons from earlier leaders.
        Return:
                self.dic_final_platoon_info = {66: 'AHHHHHHHHHH', 90: 'AHHHHH', 138: 'AHHHHHHHHH', 174: 'AH', 177: 'AHH'}
        """

        if not dic_follower_state:
            return
        # 1. Find the newest leader (the last leader_id in the dict order)
        # Since dict preserves insertion order in Python 3.7+
        last_item = next(reversed(dic_follower_state.items()))
        newest_leader = last_item[1][1]  # info[1] = leader_id
        # 2. Count followers that belong to ONLY this newest leader
        #    and are not free_mode
        count = 0
        for vid, (state, leader_id) in dic_follower_state.items():
            if leader_id == 'mav40':
                pass
            if leader_id == newest_leader and state != "free_mode":
                count += 1
        # 3. If no valid followers, do not record
        if count == 0:
            return
        # 4. Construct platoon string: "A" + "H" * count
        platoon_string = "A" + "H" * count
        if platoon_string == 'A':
            pass
        if len(platoon_string) > self.max_team_size:
            pass
        # 5. Save result
        self.dic_final_platoon_info[step//10] = platoon_string
        # update to dic_avhid_ptype; this is truly pass to merging controller
        self.data_recorder.get_avhid_ptype(m_dpt_type={newest_leader: platoon_string})
