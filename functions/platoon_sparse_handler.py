# platoon_sparse_handler.py
# SC2: Handle sparse platoons - RF prediction, find sparse, promote
# SC3: Collect free followers - find nearby AVs, execute collection (NEW)

import os
import joblib
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

class PlatoonSparseHandler:
    def __init__(self, traci, data_recorder, platoon_basic):
        self.traci = traci
        self.data_recorder = data_recorder
        self.p_basic = platoon_basic

        self.free_triggered = False  # for record_predict3
        # Load Random Forest model for follower state prediction
        self.fs_model = joblib.load(
            os.path.join(project_root, 'rf_models', 'follower_state_prediction_model_260829_ndarray_final.pkl'))

    # TODO: Optimise prediction efficiency—consider batch processing or caching leader lookups to reduce computational burden when processing many HV followers
    def predict_flw_state_discard(self, dic_id_type, ls_vehid, model=False):
        """
        Updated platoon-wise prediction:
        - Add per-follower last leader mapping in `self.dic_fol_last_leader`.
        - Add per-leader free cascade flag in `self.dic_leader_free_triggered`.
        - Scan vehicles newest -> oldest and (re)predict any follower that is new or whose leader changed.

        Params:
        - dic_id_type/dic_tags
        - ls_vehid: tuple, all vehicle in this step
        - dic_fol_last_leader: {follower : last_leader, ...}
        """
        if not dic_id_type:
            return self.dic_id_preState, self.dic_id_features
        # Clean / reset when a vehicle becomes leader
        for vid, tag in dic_id_type.items():
            if tag == 1:
                self.dic_fol_last_leader.pop(vid, None)
                self.dic_id_features.pop(vid, None)  # removes id from the dictionary; returns None if id doesn't exist.
                self.dic_id_preState.pop(vid, None)  # removes the prediction state for id.
                self.dic_leader_free_triggered[vid] = False

        # Process newest -> oldest
        # items = list(dic_id_type.items())[::-1] # seems has a problem in this sequence!
        items = list(dic_id_type.items())
        for vid, tag in items:
            # only handle followers (both AV and HV)
            if tag == 1:
                continue

            # if left network, remove records so we can re-evaluate later
            if vid not in ls_vehid:
                self.dic_fol_last_leader.pop(vid, None)
                self.dic_id_features.pop(vid, None)
                self.dic_id_preState.pop(vid, None)
                continue

            # find current leader for this follower
            leader_id, _ = self.p_basic.get_cor_leader(vid)
            if leader_id is None:
                # no leader mapping now; clear last leader so it will be retried later
                self.dic_fol_last_leader.pop(vid, None)
                continue

            # skip if already predicted for this leader
            if vid in self.dic_fol_last_leader and self.dic_fol_last_leader[vid] == leader_id:
                continue

            # extract features (this also stores features in self.dic_id_features)
            arr_select_features = self.get_RFfeatures(vid)
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

    def get_RFfeatures_discard(self, new_follower_id):
        '''
        get Random Forest features of new_follower_id
        :return: df_select_features
        '''
        minGap = 4.5
        veh_length = 5
        # == get features ==
        preceding_info = self.traci.vehicle.getLeader(new_follower_id)
        if preceding_info is None:
            return None  # No leading vehicle, skip
        pv_id, pv_dis = preceding_info
        # real dis to the preceding veh; FEATURE 1
        dis_to_pv = pv_dis + minGap + veh_length

        # velocity of preceding veh; FEATURE 2
        # v_pv = self.traci.vehicle.getSpeed(pv_id)
        v_pv = self.data_recorder.get_vid_states(pv_id)['v']
        # get pos of this veh; FEATURE 4
        # pos_this = self.traci.vehicle.getLanePosition(new_follower_id)
        pos_this = self.data_recorder.get_vid_states(new_follower_id)['pos']

        # get its leader_AV id and index of this veh (COResponding)
        leader_id, index_this = self.p_basic.get_cor_leader(new_follower_id)
        if leader_id is None:
            return None  # No leader found, skip
        # size (veh_num); FEATURE 5
        veh_num = index_this + 1  # size, how many veh between this veh and its leader_AV, start from 0
        # get pos of leader_id
        if leader_id in self.traci.vehicle.getIDList():
            # pos_leader = self.traci.vehicle.getLanePosition(leader_id)
            pos_leader = self.data_recorder.get_vid_states(leader_id)['pos']
        else:
            pos_leader = 2000
        # dis to the leader_AV; FEATURE 3
        dis_to_leaderAV = pos_leader - pos_this
        leader_id_start = leader_id
        features = [new_follower_id, dis_to_pv, v_pv, dis_to_leaderAV, pos_this, veh_num, leader_id_start]
        self.dic_id_features[new_follower_id] = features

        # filtered features
        select_features = [dis_to_pv, v_pv, dis_to_leaderAV, veh_num]  # 1, 2, 3, 5
        arr_select_features = np.array(select_features, dtype=float).reshape(1, -1)
        return arr_select_features

    def find_sparse_platoon(self, dic_nonOversized, dic_id_preState):
        '''
        identify sparse platoon
        :param dic_id_preState:
        :return: dic_sparse_platoon = {leader_id : first_free_follower, ...}
                 dic_standard_platoon = {leader_id : [leader, follower1, follower2, ...], ...}
                 // those platoons are non-oversized and without free followers
        '''
        dic_sparse_platoon = {}
        dic_standard_platoon = {}
        for leader, ls_followers in dic_nonOversized.items():
            if leader == 'm_av1389':
                pass
            if leader in ['m_av39', 'mb_av412', 'm_av580', 'm_av1389', 'm_av1659',
                          'm_av1874', 'm_av2119', 'm_av2437', 'm_av3218', 'm_av5277',
                          'm_av5842', 'm_av6141', 'm_av6373', 'm_av6652', 'm_av7058',
                          'm_av7253', 'm_av7727', 'm_av8010', 'm_av8602', 'm_av9062',
                          'm_av9350', 'm_av9786', 'm_av11149', 'm_av11327', 'm_av11606',
                          'm_av12019', 'm_av13729', 'm_av13892', 'm_av14487']:
                pass
            first_free_follower = None
            for follower in ls_followers:
                if (follower in dic_id_preState and dic_id_preState[follower] == 0):
                        # and leader not in self.dic_AVroleChange):  # temporary measure, future multiple step prediction
                    # tag as sparse platoon
                    first_free_follower = follower
                    dic_sparse_platoon[leader] = first_free_follower
                    break
            if first_free_follower is None:
                dic_standard_platoon[leader] = ls_followers
        return dic_sparse_platoon, dic_standard_platoon

    def free_promote(self, dic_sparse_platoon, dic_platoon_members):
        '''
        promote AV_follower (from 2 to 1) between leader_AV and first_free_follower if possible
        :param dic_sparse_platoon: {sparse_leader : first_free_follower}
        :param dic_platoon_members: all platoon leader and its followers
        :return:
        '''
        addressed_sparse_leaders = set()
        for sparse_leader, first_free_follower in dic_sparse_platoon.items():
            if sparse_leader == 'm_av3413':
                pass
            platoon_members = dic_platoon_members[sparse_leader]
            idx_first_free = platoon_members.index(first_free_follower)
            ls_following_fol = platoon_members[1:idx_first_free]
            ls_free_fol = platoon_members[idx_first_free:]
            # check if there are any av_follower before first_free_follower
            ls_av_following_fol = [vid for vid in ls_following_fol if 'av' in vid]
            ls_av_free_fol = [vid for vid in ls_free_fol if 'av' in vid]
            promote_av = None
            if 'av' in first_free_follower: # s1: first_free_follower is an AV
                promote_av = first_free_follower
            elif ls_av_following_fol: # s2: there are AV followers before first_free_follower
                promote_av = ls_av_following_fol[-1]
            elif ls_av_free_fol: # s3: there are AV followers after first_free_follower
                promote_av = ls_av_free_fol[0]
            # if no AV is available to promote, skip
            if promote_av is None:
                continue
            addressed_sparse_leaders.add(sparse_leader)
            # free_promote
            self.p_basic.dic_tags[promote_av] = 1
            self.p_basic.dic_AVroleChange[promote_av] = 'free_promote'
            self.free_triggered = True
            # Remove prediction state since it's now a leader
            self.p_basic.dic_id_preState.pop(promote_av, None)
            # self.dic_id_preState.pop(promote_av, None)
        return addressed_sparse_leaders

    def filter_out_AV_followers(self, dic_sparse_platoon, dic_platoon_members):
        '''
        filter out AV followers, as the role of AV_follower could change
        Params:
            - dic_sparse_platoon: {sparse_leader : first_free_follower}
            - dic_platoon_members: {leader: [leader, follower1, follower2, ...], ...}
        Return:
            - dic_sparse_platoon_filtered: {sparse_leader: first_free_hv_follower}
        '''
        dic_sparse_platoon_filtered = {}
        for sparse_leader, first_free_fol in dic_sparse_platoon.items(): # leader == 'm_av1285'
            if sparse_leader == 'm_av1202':
                pass
            platoon_members = dic_platoon_members[sparse_leader]
            idx_first_free = platoon_members.index(first_free_fol)
            ls_free_fol = platoon_members[idx_first_free:]
            # Filter to keep only HV followers
            ls_hv_free_fol = [fol for fol in ls_free_fol if 'hv' in fol]
            if len(ls_hv_free_fol) >= 1: # >=
                dic_sparse_platoon_filtered[sparse_leader] = ls_hv_free_fol[0]
        return dic_sparse_platoon_filtered

    def find_sparseP_nearbyAV(self, ls_ihB_av_asc, dic_sparse_platoon):
        """
        Find nearby side-lane AVs for sparse platoons with free followers

        :param ls_ihB_av_asc: side-lane AVs (ascending order)
        :param dic_sparse_platoon: {sparse_leader: first_free_follower}
        :param dic_platoon_members: platoon membership info
        :return: dic_sparse_candidates = {sparse_leader: [candidate_av1, candidate_av2, ...]}
        """
        dic_sparse_candidates = {}
        leaders = list(self.p_basic.dic_platoon_members)

        for sparse_leader, first_free_fol in dic_sparse_platoon.items():
            if sparse_leader == 'm_av1389':
                pass
            # Get position of first free follower
            try:
                pos_sparse_leader = self.data_recorder.get_vid_states(sparse_leader)['pos']
            except:
                continue

            # Find side-lane AVs that are behind this free follower
            for index, side_av in enumerate(ls_ihB_av_asc):
                try:
                    pos_side_av = self.data_recorder.get_vid_states(side_av)['pos']
                except:
                    continue

                if pos_side_av <= pos_sparse_leader:
                    # All AVs from this point onward are candidates
                    av_candidates = ls_ihB_av_asc[index:]
                    leader_idx = leaders.index(sparse_leader)
                    next_leader_id = leaders[leader_idx + 1] if leader_idx + 1 < len(leaders) else None
                    next_leader_pos = self.data_recorder.dic_pos.get(next_leader_id) if next_leader_id else None
                    if next_leader_pos is not None:
                        av_candidates = [av for av in av_candidates
                                         if next_leader_pos <= self.data_recorder.dic_pos.get(av, -1) <= pos_sparse_leader]
                    dic_sparse_candidates[sparse_leader] = av_candidates
                    break

        return dic_sparse_candidates
