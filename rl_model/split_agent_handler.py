# split_agent_handler.py

import os
import numpy as np
from datetime import datetime

from rl_model.rl_module import RLScoringAgent

current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the absolute path of the current script's directory
project_root = os.path.dirname(current_dir) # Get the parent directory as the project root

# model_name = 'split_score_model_251124_1900.pt'
# path_pt = os.path.join(project_root, 'rl_model', 'saved_models', model_name)

class SplitAgentHandler:
    def __init__(self, traci, data_recorder, scoring_interval=10, mode='train',
                 tsg_mode='off', exp_name='default_run', lr=5e-4,
                 hidden_dims=(64, 64), gate_agent=None, tsg_manager=None,
                 model_path=None):
        self.traci = traci
        self.data_recorder = data_recorder
        self.mode = mode
        self.tsg_mode = tsg_mode

        active_exp_name = exp_name if mode == 'train' else f"EVAL_{exp_name}"

        # default_model = 'split_score_model_251124_1900.pt'
        # default_model = 'RL_Training_8847168/SA_LR0.0001_I16_HA64HB64_S21_20260908_1118__task9/models/final_SA.pt'
        default_model = 'se_260909_0724.pt'
        default_path = os.path.join(
            project_root, 'rl_model', 'saved_models', default_model)
        score_model_path = (model_path or default_path) if mode == "predict" else None
        self.agent = RLScoringAgent(
            traci,
            data_recorder,
            exp_name=active_exp_name,
            model_path=score_model_path,
            lr=lr,
            hidden_dims=hidden_dims)

        self.gate_agent = gate_agent
        self.tsg_manager = tsg_manager
        if self.tsg_mode in ("train", "predict", "audit") and self.gate_agent is None:
            raise ValueError("[SA-Gate] tsg_mode requires a shared gate_agent")
        self.task_name = 'splitting'
        self.gate_collected = 0

        self.scoring_interval = scoring_interval  # Minimum interval (in steps) between scoring attempts
        self.training_warmup_steps = 1800 # 180 s
        self.next_save_step = 10000
        self.collected = 0  # Track how many transitions have been collected
        self.target_lane = 0

        self.ls_splited_platoon = [] # splited platoon leader
        self.insert_buffer = []
        self.last_score_step = {} # Records last scoring step for each leader_id (cooldown control)
        self.ls_score = []
        self.dic_split_insertedAV = {} # plan taking action but may still in process
        self.dic_score_reward = {} # record lc_av and [score, reward], dic = {lc_av:[score, reward]}
        self.dic_tsg_meta = {}  # track deploy-time metadata for logging

        self.payloads = []
        self.selected_avs_this_step = set()

    def run_agent_decision(self, step, dic_platoon_members,
                           dic_oversized_platoon_states,
                           dic_leader_candidates, ls_upA_asc,
                           gating_value=None):
        """
        dic_oversized_platoon_states => {leader_AV : [head_pos,
                                                    tail_pos,
                                                    avg_speed,
                                                    size]}
        Main function to handle AV insertion decisions.
        """
        self.payloads = []
        self.selected_avs_this_step = set()

        if self.mode == 'train' and step < self.training_warmup_steps:
            return {}
        if not dic_oversized_platoon_states:
            return {}
        # Target platoons inherit downstream-to-upstream order (closest to MCZ first).
        for oversize_leader in dic_oversized_platoon_states:
            if oversize_leader in ['mb_av9304', 'm_av11610']:
                pass
            if oversize_leader in self.ls_splited_platoon or oversize_leader not in dic_leader_candidates:
                continue

            available_candidates = [
                av_id
                for av_id in dic_leader_candidates[oversize_leader]
                if av_id not in self.selected_avs_this_step
            ]

            if not available_candidates:
                continue

            current_candidates = dict(dic_leader_candidates)
            current_candidates[oversize_leader] = available_candidates

            selected_av, selected_state, score, gate_input = self._evaluate_candidates(
                step, oversize_leader, dic_platoon_members,
                current_candidates, dic_oversized_platoon_states,
                ls_upA_asc, gating_value)

            if selected_av:
                platoon_snapshot = list(dic_platoon_members[oversize_leader])
                self.payloads.append((oversize_leader, selected_av,
                    selected_state, platoon_snapshot,
                    score, gate_input))
                self.selected_avs_this_step.add(selected_av)

        return self.dic_split_insertedAV

    def release_insertion(self, step, laneChange_buffer):
        if laneChange_buffer:
            for payload in self.payloads:
                laneChange_buffer.push(step, payload)

            self.payloads.clear()

            # Execute every command whose communication delay has expired.
            while True:
                delayed_payload = laneChange_buffer.maybe_release(step)
                if delayed_payload is None:
                    break
                self._execute_insertion(step, *delayed_payload)

        else:
            for payload in self.payloads:
                self._execute_insertion(step, *payload)

            self.payloads.clear()

    def update_reward(self, current_step, st, dic_platoon_members, train_interval):
        """
        Check insert_buffer and issue rewards if leader has exited control zone.
        """
        updated = False

        # iterate over a shallow copy to safely remove items during loop
        for record in self.insert_buffer[:]:
            leader_id = record['leader_id']

            try:
                lane_id = self.traci.vehicle.getLaneID(leader_id)
            except self.traci.TraCIException:
                lane_id = None

            if lane_id == 'ws_1':  # inflow_highway_0
                lc_av = record['av_id']
                platoon_snapshot = record["platoon_snapshot"]

                reward = self._evaluate_insertion_reward(
                    lc_av,
                    platoon_snapshot,
                    dic_platoon_members
                )

                self.dic_score_reward[lc_av].append(reward)
                self.agent.log_score_reward(
                    lc_av, self.dic_score_reward[lc_av][0], reward
                )
                meta = self.dic_tsg_meta.pop(lc_av, None)
                if self.tsg_mode in ("predict", "audit") and self.tsg_manager and meta is not None:
                    self.tsg_manager.log_tsg_reward(
                        decision_step=meta["step"],
                        reward_step=current_step,
                        category="se",
                        cand_av_id=lc_av,
                        target_platoon_leader_id=meta["target_platoon_leader_id"],
                        final_reward=reward,
                        tsg_execute=meta.get("tsg_execute"),
                        real_execute=True
                    )

                # === Different training targets for different modes ===
                if self.mode == 'train':
                    # Train the original utility scoring model
                    self.agent.record_transition(
                        record['state'],
                        reward
                    )
                    self.collected += 1

                elif self.tsg_mode == 'train':
                    # Train the self-gating model using delayed outcome label
                    self.gate_agent.record_transition(
                        gate_input=record["gate_input"],
                        reward=reward,
                        task_name=self.task_name
                    )

                print(
                    f"[SplitInsert] {lc_av} reward: "
                    f"{'+' if reward > 0 else ''}{reward:.3f}"
                )

                self.insert_buffer.remove(record)
                updated = True

        # === Train original scorer ===
        if updated and self.mode == 'train':
            if self.collected >= train_interval:
                self.agent.log_training_metrics(current_step)
                self.agent.train_on_recorded(
                    current_step,
                    epochs=5,
                    batch_size=int(train_interval / 2)
                )
                self.collected = 0

        # Train any completed transitions left below train_interval at shutdown.
        if (self.mode == 'train' and current_step == st * 10 - 10
                and self.agent.memory):
            self.agent.log_training_metrics(current_step)
            self.agent.train_on_recorded(
                current_step,
                epochs=5,
                batch_size=max(1, int(train_interval / 2))
            )
            self.collected = 0

        if self.mode == 'train':
            self._save_model_if_needed(current_step, st)

        return self.dic_score_reward


    def _evaluate_insertion_reward(self, lc_av, platoon_snapshot,
                                   dic_platoon_members):
        """
        Evaluate insertion success and return a reward score for split_insert scenario.

        REWARD SETTINGS SUMMARY (Split Agent):
        ========================================

        1. FAILURE PENALTIES (return -0.1):
           - Wrong lane: AV not on 'inflow_highway_0' (inner lane)
           - Missed insertion: AV position < platoon tail position (didn't cut in)
           - Invalid split: AV inserted at head/tail only (num_front < 1 or num_rear < 1)
           - TraCI exception: Vehicle not found or simulation error

        2. SUCCESS REWARDS (return value in (0, 1]):
           Formula: reward = 1.0 - imbalance_ratio
           where: imbalance_ratio = |num_front - num_rear| / (total_vehicles - 1)

           Reward Scale Examples:
           - Perfect balance (5 front, 5 rear): reward = 1.0
           - Moderate imbalance (6 front, 4 rear): reward ≈ 0.8
           - High imbalance (10 front, 2 rear): reward ≈ 0.27

           Goal: Encourage splits near the middle of oversized platoons to create
                 two balanced sub-platoons for optimal throughput.

        3. EVALUATION CASES:
           Case 1: AV becomes new leader with followers (dic_platoon_members[lc_av] exists)
                   - Count rear: all followers in new platoon (len(new_platoon) - 1)
                   - Count front: position of split_anchor_id in original platoon snapshot

           Case 2: AV not yet recognized as leader (transitional state)
                   - Count by position: compare lanePosition of all platoon members
                   - num_front starts at 1 (original leader), num_rear starts at 0

        Args:
            lc_av: ID of the inserted autonomous vehicle.
            platoon_snapshot: oversized platoon members at the moment of insertion decision
            dic_platoon_members: dict mapping leader ID to list of current platoon member IDs.

        Returns:
            float: Reward in [-0.1, 1.0] range
                  -0.1 = failure (wrong lane, missed insertion, invalid split)
                  (0, 1.0] = success (quality based on split balance)
        """
        try:
            if lc_av == 'mb_av948':
                pass
            leader_av = platoon_snapshot[0]
            tail_id = platoon_snapshot[-1]

            current_laneID = self.traci.vehicle.getLaneID(lc_av)
            this_pos = self.traci.vehicle.getLanePosition(lc_av)
            tail_pos = self.traci.vehicle.getLanePosition(tail_id)
            if current_laneID != 'inflow_highway_0': # upstream_0
                # AV must be on inflow_highway_0 (upstream_0)'
                return -0.1
            if this_pos < tail_pos:
                # lc_AV didn't cut into target oversized platoon
                return -0.1

            if lc_av in dic_platoon_members and len(dic_platoon_members[lc_av]) > 1:
                # Case1: when lc_av becomes a new leader or has followers (a valid split)
                new_platoon = dic_platoon_members[lc_av]
                num_rear = len(new_platoon)-1 # all followers
                # Try to find the first AV in new_platoon that also appears in platoon_snapshot
                split_anchor_id = None
                for vid in new_platoon[1:]:
                    if vid in platoon_snapshot:
                        split_anchor_id = vid
                        break
                # Use its position in platoon_snapshot as the split index
                if split_anchor_id:
                    num_front = platoon_snapshot.index(split_anchor_id)
                else:
                    num_front = 0  # fallback: no match found
            else:
                # Case2: lc_av currently not a leader, or its new platoon has no followers
                num_front = 1 # add av_leader
                num_rear = 0
                for id in platoon_snapshot[1:]: # exclude original leader
                    if id == lc_av:
                        continue # skip self
                    pos = self.traci.vehicle.getLanePosition(id)
                    if pos > this_pos:
                        num_front += 1
                    else: # pos <= current_pos
                        num_rear += 1

            if num_front >= 1 and num_rear >= 1:
                imbalance = abs(num_front - num_rear)
                total = num_front + num_rear + 1
                imbalance_ratio = imbalance / (total - 1)
                reward = 1.0 - imbalance_ratio
                return reward
            else:
                return -0.1
        except self.traci.TraCIException:
            return -0.1

    def record_loss(self, current_step, st):
        if current_step != st*10 - 10 or self.mode != 'train':
            return
        self.agent.record_plot_loss()  # plot loss curve

    def record_scores(self, current_step, st):
        if current_step != st*10 - 10 or self.mode != 'train':
            return
        self.agent.record_plot_scores(self.ls_score)
        self.agent.record_plot_score_reward(self.dic_score_reward)

    def _save_model_if_needed(self, current_step, st):
        """
        Periodically save the trained model.
        """
        save_interval = 30000  # every 10k steps
        if current_step > self.next_save_step or current_step == st*10 - 10:
            # os.makedirs("saved_models", exist_ok=True)
            timestamp = datetime.now().strftime("%y%m%d_%H%M")  # e.g. 250517_1915
            filename = f"sa_{timestamp}.pt"
            self.agent.save_model(filename)
            if current_step == st * 10 - 10:
                self.agent.save_model("final_SA.pt")
            print(f"[Model-SA] Auto-saved at step {current_step}")
            self.next_save_step += save_interval  # set next checkpoint

    def _evaluate_candidates(self, step, leader_id,
                             dic_platoon_members, dic_leader_candidates,
                             dic_oversized_platoon_states, ls_upA_asc, gating_value):
        """
        Evaluate candidate AVs for a given leader.
        Parameters:
            dic_oversized_platoon_states => {leader_AV : [head_pos,
                                                            tail_pos,
                                                            avg_speed,
                                                            size]}
            platoon_states => [head_pos, tail_pos, avg_speed, size]

        Procedure:
            1. Score all candidate AVs using the pretrained scoring model.
            2. Select the top-ranked candidate.
            3. Build gate_input at the decision moment.
            4. Decide whether to execute:
                - gate_train: always execute top candidate to collect delayed labels
                - selfgate_predict: use self-gate
                - train/predict: keep original fixed gating
        """

        last_step = self.last_score_step.get(leader_id, -999)
        if step - last_step < self.scoring_interval:
            return None, None, None, None
        self.last_score_step[leader_id] = step

        ls_candidateAV = dic_leader_candidates[leader_id]
        if not ls_candidateAV:
            return None, None, None, None

        pMember = dic_platoon_members[leader_id]
        tail_id = pMember[-1]
        platoon_states = dic_oversized_platoon_states[leader_id]

        candidate_states = []
        scores = []
        valid_candidates = []

        # === Score all candidate AVs ===
        for av_id in ls_candidateAV:
            try:
                state = self.agent.state_builder.build_state_se(
                    av_id, leader_id, tail_id, platoon_states)
                if state is None:
                    continue
                score = self.agent.predict_score(state)
                valid_candidates.append(av_id)
                candidate_states.append(state)
                scores.append(score)
            except Exception as e:
                print(f"[SplitInsert] {av_id} failed to score: {e}")
                continue

        if not scores:
            return None, None, None, None

        # === Select top-ranked AV ===
        top_idx = int(np.argmax(scores))
        selected_av = valid_candidates[top_idx]
        selected_state = candidate_states[top_idx]
        best_score = float(scores[top_idx])

        # === Build gate input at decision time ===
        gate_input = None

        if self.tsg_mode in ("train", "predict", "audit"):

            leader_pos = float(self.data_recorder.get_vid_states(leader_id)['pos'])
            d_target_to_MCZ = self.data_recorder.length_pfz - leader_pos

            gate_input = self.gate_agent.build_gate_input(
                scores=scores,
                top_idx=top_idx,
                task_name=self.task_name,
                d_target_to_MCZ=d_target_to_MCZ,
                target_platoon_size=float(platoon_states[3]),
                max_platoon_size=self.data_recorder.max_platoon_size,
                length_pfz=self.data_recorder.length_pfz,
            )

        # === Decide whether to execute ===
        if self.tsg_mode == "train":
            # Do not reject during gate training.
            # Always execute top-ranked AV to collect delayed outcome labels.
            execute_decision = True

        elif self.tsg_mode in ("predict", "audit"):
            execute_decision, gate_logits, gate_probs = self.gate_agent.predict_execute(gate_input)
            gate_execute = bool(execute_decision)
            if self.tsg_mode == "audit":
                execute_decision = True

            print(
                f"[SA-Gate] leader={leader_id}, av={selected_av}, "
                f"score={best_score:.3f}, "
                f"reject_prob={gate_probs[0]:.3f}, "
                f"execute_prob={gate_probs[1]:.3f}, "
                f"tsg_execute={gate_execute}, real_execute={execute_decision}"
            )

            if self.tsg_manager is not None:
                self.tsg_manager.log_tsg_decision(
                    decision_step=step,
                    category="se",
                    cand_av_id=selected_av,
                    target_platoon_leader_id=leader_id,
                    score=float(best_score),
                    reject_prob=float(gate_probs[0]),
                    execute_prob=float(gate_probs[1]),
                    tsg_execute=gate_execute,
                    real_execute=bool(execute_decision),
                )

        else:
            # Original fixed-gating logic for train / predict mode.
            gating_value = 0 if len(pMember) > 20 else gating_value

            if gating_value is not None and best_score < gating_value:
                execute_decision = False
            else:
                execute_decision = True

        # === If rejected, return no selected AV ===
        if not execute_decision:
            self.ls_score.append(best_score)
            self.agent.log_score(best_score)
            print(f"[SA-Gate] Reject candidate: {selected_av}, score={best_score:.3f}")
            return None, None, best_score, gate_input

        # === If executed, record score for delayed reward update ===
        self.dic_score_reward[selected_av] = [best_score]
        if self.tsg_mode in ("predict", "audit"):
            self.dic_tsg_meta[selected_av] = {
                "target_platoon_leader_id": leader_id,
                "step": step,
                "tsg_execute": gate_execute if 'gate_execute' in locals() else None,
            }
        self.ls_score.append(best_score)
        self.agent.log_score(best_score)

        return selected_av, selected_state, best_score, gate_input


    def _execute_insertion(self, step, leader_id, selected_av, selected_state,
                           platoon_snapshot, score, gate_input):
        """
        Execute the lane change for the selected AV and update tracking structures.
        """
        try:
            self.traci.vehicle.changeLane(selected_av, self.target_lane, duration=100)
            print(f"[SplitInsert] {selected_av} selected with score {score:.3f}")
            self.ls_splited_platoon.append(leader_id)

            self.insert_buffer.append({
                "leader_id": leader_id,
                "platoon_snapshot": platoon_snapshot,
                "av_id": selected_av,
                "state": selected_state,
                "gate_input": gate_input,
                "score": score,
                "step": step
            }) # this data is for training

            self.dic_split_insertedAV[selected_av] = 'split_insert'
            print(f'[SplitInsert] split_inserted_AV: {self.dic_split_insertedAV}')
        except self.traci.TraCIException:
            print(f"[SplitInsert] {selected_av} failed to insert due to TraCI exception")
