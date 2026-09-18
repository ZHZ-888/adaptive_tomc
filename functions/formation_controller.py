from functions import platoon_basic as pbasic
from functions import platoon_oversized_handler as poversized
from functions import platoon_sparse_handler as psparse
from functions import platoon_lane_manager as plane
from functions import v2x_disturbance as v2x
from functions import detector_pass_recorder as detector

from rl_model import split_agent_handler as split_agent
from rl_model import collect_agent_handler as collect_agent
from rl_model import tsg_manager


FC_MODES = {
    'dla_only': {'tsc': False, 'lhr': False, 'ce': False, 'se': False},
    'dla_tsc': {'tsc': True, 'lhr': False, 'ce': False, 'se': False},
    'dla_tsc_lhr': {'tsc': True, 'lhr': True, 'ce': False, 'se': False},
    'dla_tsc_lhr_ce': {'tsc': True, 'lhr': True, 'ce': True, 'se': False},
    'full': {'tsc': True, 'lhr': True, 'ce': True, 'se': True},
}


class FormationController:
    def __init__(self, data_recorder, traci, sa_mode='predict', ca_mode='predict',
                 tsg_mode='off', exp_name='default_run', loss_rate=0, learning_rate=5e-4,
                 train_interval=32, expert_hidden_dims=(64, 64),
                 gate_hidden_dims=(16, 16), max_team_size=12, fc_mode='full',
                 comm_rng=None, se_model_path=None, ce_model_path=None, tsg_model_path=None):
        '''
        Train split_agent, "sa_mode='train', ca_mode='off'"
        Train collect_agent, "sa_mode='off', ca_mode='train'

        Evaluate split_agent, "sa_mode='predict', ca_mode='off'"
        Evaluate collect_agent, "sa_mode='off', ca_mode='predict'
        '''

        self.data_recorder = data_recorder
        self.max_team_size = max_team_size
        self.fc_mode = fc_mode
        self.modules = FC_MODES[fc_mode]

        # Initialise platoon modules
        self.pass_recorder = detector.DetectorPassRecorder(
            traci, data_recorder) # use sumo instantInductionLoop detector to count platoon members
        self.p_basic = pbasic.PlatoonBasic(traci, data_recorder, self.pass_recorder, max_team_size=max_team_size)
        self.p_oversized = poversized.PlatoonOversizedHandler(traci, data_recorder, self.p_basic)
        self.p_sparse = psparse.PlatoonSparseHandler(traci, data_recorder, self.p_basic)
        self.p_lane = plane.PlatoonLaneManager(traci, data_recorder)

        self.loss_rate = loss_rate
        if self.loss_rate == 0:
            self.sa_buffer = None
            self.ca_buffer = None
            self.lane_buffer = None
        else:
            self.sa_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate, rng=comm_rng)  # buffer for splitting agent
            self.ca_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate, rng=comm_rng)  # buffer for collecting agent
            self.lane_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate, rng=comm_rng)

        self.train_interval = train_interval # rl training update interval
        self.update_interval = 10 # test

        if tsg_mode == 'fix': # with gating threshold
            self.sa_gating = 0.2 if sa_mode == 'predict' else 0
            self.ca_gating = 0.2 if ca_mode == 'predict' else 0
        else:
            # no gating or use tsg_mode
            self.sa_gating = 0 if sa_mode == 'predict' else 0
            self.ca_gating = 0 if ca_mode == 'predict' else 0

        # RL Agents
        self.sa_mode = sa_mode
        self.ca_mode = ca_mode
        self.tsg_mode = tsg_mode

        self.tsg_manager = tsg_manager.TSGManager(
            tsg_mode=tsg_mode,
            exp_name=exp_name,
            lr=learning_rate,
            train_interval=train_interval,
            hidden_dims=gate_hidden_dims,
            predict_model_path=tsg_model_path,
        )

        self.split_agent = split_agent.SplitAgentHandler(
            traci, data_recorder, mode=sa_mode, tsg_mode=tsg_mode,
            exp_name=exp_name, lr=learning_rate,
            gate_agent=self.tsg_manager.gate_agent,
            tsg_manager=self.tsg_manager,
            model_path=se_model_path,
            hidden_dims=expert_hidden_dims) if sa_mode !='off' else None

        self.collect_agent = collect_agent.CollectAgentHandler(
            traci, data_recorder, self.p_basic, mode=ca_mode, tsg_mode=tsg_mode,
            exp_name=exp_name, lr=learning_rate,
            gate_agent=self.tsg_manager.gate_agent,
            tsg_manager=self.tsg_manager,
            model_path=ce_model_path,
            hidden_dims=expert_hidden_dims) if ca_mode != 'off' else None

    def platoon_initialise(self, ls_ihA_asc, ls_vehid, rf_model):
        # ******** PLATOON INITIALISATION ********
        dic_tags, ls_leader_AV, ls_follower_AV, dic_AVroleChange \
            = self.p_basic.tag_vehicles13(
                ls_ihA_asc, max_team_size=self.max_team_size, enable_lhr=self.modules['lhr'])
        his_dic_platoon_size, dic_platoon_size, dic_platoon_members \
            = self.p_basic.get_platoon_size3(ls_ihA_asc, ls_leader_AV)
        dic_id_preState, dic_id_features \
            = self.p_basic.predict_flw_state(dic_tags, ls_vehid, model=rf_model)
        return (dic_tags, ls_leader_AV, ls_follower_AV, dic_platoon_size, dic_platoon_members,
                his_dic_platoon_size, dic_id_preState, dic_id_features)

    def splitting(self, st, step, ls_ihA_asc, ls_ihB_av_asc,
                  dic_platoon_size, dic_platoon_members, selected_vid):
        # ******** HANDLE OVERSIZED PLATOONS ********
        if self.modules['se'] and self.split_agent:
            self.split_agent.release_insertion(step, self.sa_buffer)
        if step % self.update_interval != 0:
            return {}
        dic_oversized_platoon_states, dic_split_candidates, dic_nonOversizedP \
            = self.p_oversized.find_oversizedP_nearbyAV(ls_ihB_av_asc, dic_platoon_size, dic_platoon_members)
        # ** SPLIT_INSERT ** agent
        if self.modules['se'] and self.split_agent:
            self.split_agent.run_agent_decision(
                step, dic_platoon_members,
                dic_oversized_platoon_states,
                dic_split_candidates,
                ls_ihA_asc, gating_value=self.sa_gating)
            selected_vid.update(self.split_agent.selected_avs_this_step)
            dic_score_reward = self.split_agent.update_reward(step, st, dic_platoon_members,
                                                              train_interval=self.train_interval)
            self.split_agent.record_scores(step, st)
            self.split_agent.record_loss(step, st)
        return dic_nonOversizedP

    def collecting(self, st, step, ls_ihB_av_asc, dic_nonOversizedP,
                   dic_platoon_members, dic_id_preState, selected_vid):
        '''
        ******** HANDLE SPARSE PLATOONS ********
        Parameters
        ----------
        dic_collect_candidates = {sparse_leader: [candidate_av1, candidate_av2, ...]}

        Returns
        -------

        '''
        # ** free_promote **
        if self.modules['ce'] and self.collect_agent:
            self.collect_agent.release_insertion(step, self.ca_buffer)
        if step % self.update_interval != 0:
            return {}
        dic_sparseP, dic_standard_platoon = self.p_sparse.find_sparse_platoon(dic_nonOversizedP, dic_id_preState)
        if self.modules['lhr']:
            addressed_leaders = self.p_sparse.free_promote(dic_sparseP, dic_platoon_members)
        else:
            addressed_leaders = set()
        # filter out AV followers from sparse platoons
        dic_sparseP_filered_temp = self.p_sparse.filter_out_AV_followers(dic_sparseP, dic_platoon_members)
        dic_sparseP_filered = {
            leader: first_free
            for leader, first_free in dic_sparseP_filered_temp.items()
            if leader not in addressed_leaders
        } # if leader has been free-promoted, do not consider it for free-insertion
        # Find nearby side-lane AVs
        dic_collect_candidates = self.p_sparse.find_sparseP_nearbyAV(ls_ihB_av_asc, dic_sparseP_filered)
        dic_collect_candidates = {
            leader: [av_id for av_id in candidates if av_id not in selected_vid]
            for leader, candidates in dic_collect_candidates.items()
        }
        # Remove target platoons that have no candidates left.
        dic_collect_candidates = {
            leader: candidates
            for leader, candidates in dic_collect_candidates.items()
            if candidates
        }
        # ** FREE_INSERT ** agent
        if self.modules['ce'] and self.collect_agent:
            dic_free_insertedAV = self.collect_agent.run_free_insert_decision(step,
                                                                              dic_platoon_members,
                                                                              dic_sparseP_filered,
                                                                              dic_collect_candidates,
                                                                              gating_value=self.ca_gating)  # 0.4 for predict
            dic_free_score_reward = self.collect_agent.update_reward(step, st, dic_platoon_members,
                                                                     train_interval=self.train_interval)
            self.collect_agent.record_scores(step, st)
            self.collect_agent.record_loss(step, st)
        return dic_standard_platoon

    def control_platoon_gap(self, step, ls_vehid, ls_leader_AV, ls_follower_AV,
                            ls_m_leader_up_asc, ls_wsB_av_asc, dic_id_preState):
        '''
        leader speed control
        control gaps between platoons; do not rely on V2X, it is onboard decision
        '''
        if step % self.update_interval != 0:
            return
        self.p_basic.form_platoon3(ls_vehid, ls_leader_AV, ls_follower_AV)
        self.p_basic.restore_speed_limit3(step, ls_leader_AV, ls_m_leader_up_asc, dic_id_preState)  # improve to 25 m/s if platoon formed
        self.p_basic.restore_speed_limit2(ls_wsB_av_asc)  # restore to 27.78 m/s on wsB
        # set follower color as light green
        self.p_basic.set_follower_color()

    def step(self, st, step, lc, rf_model=True):
        '''

        :param st:
        :param step:
        :param lc: whether allow lane change for HV / av_followers; default True
        :return:
            dic_socre_reward: {'mbav1533': [0.6405384540557861]}
            dic_follower_state: {'mhv48': ['following_mode', 'mav38'], 'mhv65': ['following_mode', 'mav38']}
            his_dic_platoon_size: {'mav38': 11, 'mav278': 11, 'mav754': 11}; history of dic_platoon_size
            dic_id_features: {'mhv48': ['mhv48', 81.32679695334296, 16.66558558968912, 79.32679695334296, 5.1, 2, 'mav38'],
                                'mhv65': ['mhv65', 50.64862995222547, 24.270287929523167, 110.99295710039699, 5.1, 3, 'mav38']}
            dic_final_platoon_info: {66: 'AHHHHHHHHHH', 90: 'AHHHHH', 138: 'AHHHHHHHHH', 174: 'AH', 177: 'AHH'}
        '''
        selected_vid = set()
        # === Unpack veh info ===
        # print(f"*******step: {step}*********")
        dic_vid_groups = self.data_recorder.record_multi_lane_info()
        ls_vehid = dic_vid_groups['ls_vehid']  # tuple, all vehicle in this step
        ls_ihA_asc = dic_vid_groups['ls_ihA_asc']  # all veh on inflow_highway, ascending order
        ls_ihAB_av_asc = dic_vid_groups['ls_ihAB_av_asc']
        ls_ihB_av_asc = dic_vid_groups['ls_ihB_av_asc']
        ls_wsB_av_asc = dic_vid_groups['ls_wsB_av_asc']
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']

        # ******** PLATOON INITIALISATION (DLA, dynamic leader assignment) ********
        # ** split_promote **
        (dic_tags, ls_leader_AV, ls_follower_AV, dic_platoon_size,
         dic_platoon_members, his_dic_platoon_size, dic_id_preState, dic_id_features) \
            = self.platoon_initialise(ls_ihA_asc, ls_vehid, rf_model=rf_model)

        # ******** HANDLE OVERSIZED PLATOONS (SE) ********
        dic_nonOversizedP = self.splitting(st, step, ls_ihA_asc, ls_ihB_av_asc, dic_platoon_size,
                                           dic_platoon_members, selected_vid)
        # ******** HANDLE SPARSE PLATOONS (CE) ********
        # ** free_promote **
        dic_standard_platoon = self.collecting(st, step, ls_ihB_av_asc, dic_nonOversizedP,
                                               dic_platoon_members, dic_id_preState, selected_vid)
        # ******** SELF-GATING TRAINING ********
        self.tsg_manager.train_if_needed(step, st)
        # Flashing lane change side AVs
        # self.merge_regular.flashing_lane_changing(step, dic_insertedAV, ls_ihB)

        # ******** LANE CHANGE AND SPEED CONTROL ********
        self.p_lane.manage_hv_lc_behaviour(lc, dic_tags) # mange hv_followers lane change behavior
        self.p_lane.restrict_av_lc(ls_ihAB_av_asc) # only restrict AV strategic lane change
        # encourage AV_leader without follower move to inner lane (from lane0 to lane1)
        lane_commands = self.p_lane.move_leader_no_fol_to_inner(
            ls_leader_AV, dic_platoon_members)
        # encourage AV followers move to inner lane (from lane0 to lane1)
        lane_commands += self.p_lane.move_av_fol_to_inner(
            ls_leader_AV, dic_standard_platoon, lc)
        if self.lane_buffer:
            self.lane_buffer.push(step, lane_commands)
            lane_commands = self.lane_buffer.maybe_release(step) or []
        self.p_lane.execute_av_commands(lane_commands)

        # Control gaps between platoons
        if self.modules['tsc']:
            self.control_platoon_gap(step, ls_vehid, ls_leader_AV, ls_follower_AV,
                                     ls_m_leader_up_asc, ls_wsB_av_asc, dic_id_preState)

        # ******** UPDATE PLATOON MEMBERS BY LOOP DETECTOR RECORD INFO ********
        _ = self.pass_recorder.update(step)
        _ = self.pass_recorder.count_platoon_size(ls_leader_AV, 'mcz_entry')
        # update the last passed vehicles' speed at the exit of Merging Section
        self.data_recorder.ms_exit_speed = self.pass_recorder.ms_exit_speed

        # ******** RECORD INFO ********
        dic_member_to_leader = self.p_basic.update_member_to_leader(dic_platoon_members)
        self.data_recorder.dic_member_to_leader = dic_member_to_leader
        # record target value
        _ = self.p_basic.record_follower_state2(step, dic_id_preState) # for algorithm
        # use detector to recognise follower state; for performance evaluation only
        dic_follower_state = self.p_basic.record_follower_state_by_sensor()
        return (dic_follower_state, his_dic_platoon_size, dic_id_features)
