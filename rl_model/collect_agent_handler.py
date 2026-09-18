# collect_agent_handler.py

import os
import numpy as np
from datetime import datetime
from rl_model.rl_module import RLScoringAgent

# Model path for free-insert agent
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

class CollectAgentHandler:
    """RL agent for free-insert scenario: Score side-lane AVs for inserting
    ahead of free followers in sparse platoons.

    Goal: Maximize collection of free followers into new dense platoons.
    """

    def __init__(self, traci, data_recorder, p_basic, scoring_interval=10,
                 mode='train', tsg_mode='off', exp_name='default_run',
                 lr=5e-4, hidden_dims=(64, 64), gate_agent=None, tsg_manager=None,
                 model_path=None):
        """
        Initialise the free-insert agent.

        Args:
            traci: SUMO traci connection
            data_recorder: DataRecording instance
            merge_regular: MergeRegular instance (for lane change execution)
            scoring_interval: Minimum steps between scoring decisions
            mode: 'train' or 'predict'
        """
        self.traci = traci
        self.data_recorder = data_recorder
        self.p_basic = p_basic
        self.mode = mode
        self.tsg_mode = tsg_mode

        active_exp_name = exp_name if mode == 'train' else f"EVAL_{exp_name}"

        # default_model = 'free_insert_score_model_260303_2333_second_version.pt'
        # default_model = 'ce_test_260907_0018.pt'
        # default_model = 'RL_Training_8847168/CA_LR0.0001_I16_HA64HB64_S21_20260908_1118__task0/models/final_CA.pt'
        default_model = 'ce_260909_0724.pt'
        default_path = os.path.join(
            project_root, 'rl_model', 'saved_models', default_model)
        score_model_path = (model_path or default_path) if mode == "predict" else None
        # Initialize RL scoring agent
        self.agent = RLScoringAgent(traci, data_recorder,
                                    exp_name=active_exp_name,
                                    model_path=score_model_path,
                                    lr=lr, hidden_dims=hidden_dims) # 0.0005

        self.gate_agent = gate_agent
        self.tsg_manager = tsg_manager

        if self.tsg_mode in ("train", "predict", "audit") and self.gate_agent is None:
            raise ValueError("[CA-Gate] tsg_mode requires a shared gate_agent")

        self.task_name = 'collecting'
        self.gate_collected = 0

        # Configuration
        self.scoring_interval = scoring_interval
        self.training_warmup_steps = 1800
        self.next_save_step = 10000
        self.collected = 0  # Transition count
        self.target_lane = 0  # Inner lane

        # Tracking structures
        self.ls_free_inserted = []  # Successfully inserted AVs
        self.insert_buffer = []  # Pending reward evaluation
        self.last_score_step = {}  # Cooldown control per sparse platoon
        self.ls_score = []  # Score history for plotting
        self.dic_collect_insertedAV = {}  # {av_id: 'free_insert'}
        self.dic_score_reward = {}  # {av_id: [score, reward]}
        self.dic_tsg_meta = {}  # track deploy-time metadata for logging

        self.payloads = []
        self.selected_avs_this_step = set()

    def run_free_insert_decision(self, step, dic_platoon_members,
                                 dic_sparse_platoons,
                                 dic_sparse_candidates, gating_value=0):
        """Score and reserve one distinct AV for each sparse platoon."""
        self.payloads = []
        self.selected_avs_this_step = set()

        if self.mode == 'train' and step < self.training_warmup_steps:
            return {}
        if not dic_sparse_candidates:
            return {}

        # Target platoons inherit downstream-to-upstream order (closest to MCZ first).
        for sparse_leader, ls_candidates in dic_sparse_candidates.items():
            if sparse_leader not in dic_sparse_platoons:
                continue

            available_candidates = [
                av_id for av_id in ls_candidates
                if av_id not in self.selected_avs_this_step
            ]
            if not available_candidates:
                continue

            selected_av, selected_state, best_score, gate_input = self._evaluate_candidates(
                step, sparse_leader, dic_platoon_members, dic_sparse_platoons,
                available_candidates, gating_value)

            if selected_av:
                first_free_follower = dic_sparse_platoons[sparse_leader]
                sparse_members_snapshot = list(
                    dic_platoon_members.get(sparse_leader, [])
                )
                self.payloads.append((
                    sparse_leader, selected_av, selected_state,
                    first_free_follower, sparse_members_snapshot,
                    best_score, gate_input
                ))
                self.selected_avs_this_step.add(selected_av)

        return self.dic_collect_insertedAV

    def release_insertion(self, step, laneChange_buffer):
        if laneChange_buffer:
            for payload in self.payloads:
                laneChange_buffer.push(step, payload)
            self.payloads.clear()

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
        Check insert_buffer for completed insertions and calculate rewards.
        """
        updated = False

        # use shallow copy because entries may be removed
        for entry in self.insert_buffer[:]:
            sparse_leader = entry['sparse_leader']

            try:
                lane_id = self.traci.vehicle.getLaneID(sparse_leader)
            except self.traci.TraCIException:
                self.insert_buffer.remove(entry)
                continue

            if lane_id != 'ws_1':
                continue

            # --- Leader exited → evaluate reward ---
            lc_av = entry['lc_av']

            reward = self.evaluate_free_insert_reward(
                lc_av,
                entry['first_free_follower'],
                entry['sparse_snapshot'],
                dic_platoon_members
            )

            if lc_av in self.dic_score_reward:
                self.dic_score_reward[lc_av].append(reward)
                self.agent.log_score_reward(
                    lc_av, self.dic_score_reward[lc_av][0], reward
                )
            meta = self.dic_tsg_meta.pop(lc_av, None)
            if self.tsg_mode in ("predict", "audit") and self.tsg_manager and meta is not None:
                self.tsg_manager.log_tsg_reward(
                    decision_step=meta["step"],
                    reward_step=current_step,
                    category="ce",
                    cand_av_id=lc_av,
                    target_platoon_leader_id=meta["target_platoon_leader_id"],
                    final_reward=reward,
                    tsg_execute=meta.get("tsg_execute"),
                    real_execute=True
                )

            print(f"[FreeInsert] {lc_av} reward: {reward:+.3f}")

            # === Train CA scorer ===
            if self.mode == 'train':
                self.agent.record_transition(
                    entry['state'],
                    reward
                )
                self.collected += 1

            # === Train TSG gate ===
            elif self.tsg_mode == 'train':
                self.gate_agent.record_transition(
                    gate_input=entry['gate_input'],
                    reward=reward,
                    task_name=self.task_name
                )
                self.gate_collected += 1

            self.insert_buffer.remove(entry)
            updated = True

        # === Update CA scorer ===
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

    def evaluate_free_insert_reward(self, lc_av, first_free_follower,
                                    sparse_snapshot, dic_platoon_members):
        """
        Calculate reward for free-insert action.

        REWARD LOGIC:
        - Goal: Maximize number of free followers converted to following mode
        - Success: captured_free_fol / total_free - captured_norm_fol * 0.01 (penalty for capturing non-free followers)
        - Failure: -0.1 (wrong lane, no followers, exception)

        Args:
            lc_av: Inserted AV ID
            sparse_snapshot: {sparse_leader: orignal followers} at decision time
            first_free_follower
            dic_platoon_members: Current platoon structure, only record platoon leaders on inflow_highway

        Returns:
            float: Reward value in [-0.1, 1.0]
        """
        penalty = -0.1

        try:
            # Check if AV is on correct lane
            lane_id = self.traci.vehicle.getLaneID(lc_av)
            if 'inflow_highway_0' not in lane_id:
                print(f"[FreeInsert] {lc_av} not on inner lane: {lane_id}")
                return penalty
            # Check if AV became a leader with followers
                # no follower => penalty
            if lc_av not in dic_platoon_members:
                print(f"[FreeInsert] {lc_av} not a platoon leader")
                return penalty
                # followers are not original free followers => penalty
            current_members = dic_platoon_members[lc_av]
            if len(current_members) <= 1:  # Only leader, no followers
                print(f"[FreeInsert] {lc_av} has no followers")
                return penalty

            # Count captured free followers
            original_leader = next(iter(sparse_snapshot.keys()))
            original_members = sparse_snapshot[original_leader] # leader + followers
            first_free_idx = original_members.index(first_free_follower)
            original_free_list = original_members[first_free_idx:]
            captured_free = [fid for fid in current_members[1:] if fid in original_free_list]
            total_free = len(original_free_list)
            # Count captured norm followers
            original_norm_list = original_members[1:first_free_idx]
            captured_norm = [fid for fid in current_members[1:] if fid in original_norm_list]

            # capture 0 = penalty
            if len(captured_free) == 0:
                print(f"[FreeInsert] {lc_av} captured no free followers")
                return penalty
            # capture > 0 => reward proportional to capture rate
            captured_free_rate = len(captured_free) / max(total_free, 1)
            reward = captured_free_rate - len(captured_norm) * 0.01
            print(f"[FreeInsert] {lc_av} captured {len(captured_free)}/{total_free} free followers: {reward:.3f}"
                  f"; captured norm followers: {len(captured_norm)}")
            return reward

        except Exception as e:
            print(f"[FreeInsert] {lc_av} exception: {e}")
            return penalty

    def _evaluate_candidates(self, step, sparse_leader, dic_platoon_members,
                             dic_sparse_platoons,
                             ls_candidates, gating_value):
        """
        Score all candidate AVs and select the best one.
        Also build gate_input for TSG.

        dic_sparse_platoons: {sparse_leader: first_free_follower_id}
        """
        # Cooldown check
        last_step = self.last_score_step.get(sparse_leader, -999)
        if step - last_step < self.scoring_interval:
            return None, None, None, None

        self.last_score_step[sparse_leader] = step

        if not ls_candidates:
            return None, None, None, None

        candidate_states = []
        scores = []
        valid_candidates = []

        # === Score all candidate AVs ===
        for av_id in ls_candidates:
            try:
                state = self.agent.state_builder.build_state_ce(
                    cand_leader=av_id,
                    target_sparse_platoon={sparse_leader: dic_sparse_platoons[sparse_leader]},
                    dic_platoon_member=dic_platoon_members,
                )

                if state is None:
                    continue

                score = self.agent.predict_score(state)
                valid_candidates.append(av_id)
                candidate_states.append(state)
                scores.append(score)

            except Exception as e:
                print(f"[FreeInsert] {av_id} failed to score: {e}")
                continue

        if not scores:
            return None, None, None, None

        # === Select top-ranked AV ===
        top_idx = int(np.argmax(scores))
        selected_av = valid_candidates[top_idx]
        selected_state = candidate_states[top_idx]
        best_score = float(scores[top_idx])

        # === Build gate input ===
        gate_input = None

        if self.tsg_mode in ("train", "predict", "audit"):
            leader_pos = float(self.data_recorder.get_vid_states(sparse_leader)['pos'])
            d_target_to_MCZ = self.data_recorder.length_pfz - leader_pos

            platoon_members = dic_platoon_members[sparse_leader]
            first_free_follower = dic_sparse_platoons[sparse_leader]
            first_free_idx = platoon_members.index(first_free_follower)

            n_free = len(platoon_members) - first_free_idx
            n_follower = len(platoon_members) - 1

            gate_input = self.gate_agent.build_gate_input(
                scores=scores,
                top_idx=top_idx,
                task_name=self.task_name,
                d_target_to_MCZ=d_target_to_MCZ,
                n_free=n_free,
                n_follower=n_follower,
                max_platoon_size=self.data_recorder.max_platoon_size,
                length_pfz=self.data_recorder.length_pfz,
            )

        # === Decide whether to execute ===
        if self.tsg_mode == "train":
            # During TSG training, always execute top candidate to collect labels
            execute_decision = True

        elif self.tsg_mode in ("predict", "audit"):
            execute_decision, gate_logits, gate_probs = self.gate_agent.predict_execute(gate_input)
            gate_execute = bool(execute_decision)
            if self.tsg_mode == "audit":
                execute_decision = True

            print(
                f"[CA-Gate] sparse_leader={sparse_leader}, top_av={selected_av}, "
                f"score={best_score:.3f}, "
                f"reject_prob={gate_probs[0]:.3f}, "
                f"execute_prob={gate_probs[1]:.3f}, "
                f"tsg_execute={gate_execute}, real_execute={execute_decision}"
            )

            if self.tsg_manager is not None:
                self.tsg_manager.log_tsg_decision(
                    decision_step=step,
                    category="ce",
                    cand_av_id=selected_av,
                    target_platoon_leader_id=sparse_leader,
                    score=float(best_score),
                    reject_prob=float(gate_probs[0]),
                    execute_prob=float(gate_probs[1]),
                    tsg_execute=gate_execute,
                    real_execute=bool(execute_decision),
                )

        else: # self.tsg_mode = None; off
            # Original fixed-gating logic
            if gating_value is not None and best_score < gating_value:
                execute_decision = False
            else:
                execute_decision = True

        # === If rejected ===
        if not execute_decision:
            self.ls_score.append(best_score)
            self.agent.log_score(best_score)
            print(
                f"[CA-Gate] Reject candidate: {selected_av}, "
                f"score={best_score:.3f}"
            )
            return None, None, best_score, gate_input

        # === If executed ===
        self.dic_score_reward[selected_av] = [best_score]
        if self.tsg_mode in ("predict", "audit"):
            self.dic_tsg_meta[selected_av] = {
                "target_platoon_leader_id": sparse_leader,
                "step": step,
                "tsg_execute": gate_execute if 'gate_execute' in locals() else None,
            }
        self.ls_score.append(best_score)
        self.agent.log_score(best_score)

        return selected_av, selected_state, best_score, gate_input

    def _execute_insertion(self, step, sparse_leader, selected_av,
                           selected_state, first_free_follower,
                           sparse_members_snapshot, score, gate_input):
        """
        Execute lane change and update tracking structures.

        selected_state:
        sparse_snapshot: {sparse_leader: ori_followers} at decision time
        """
        try:
            # Check if this sparse_leader is already being tracked
            if any(entry['sparse_leader'] == sparse_leader for entry in self.insert_buffer):
                # print(f"[FreeInsert] {sparse_leader} already being tracked, skipping insertion")
                return
            self.traci.vehicle.changeLane(selected_av, self.target_lane, duration=100)
            print(f"[FreeInsert] {selected_av} selected with score {score:.3f}")

            sparse_snapshot = {sparse_leader: sparse_members_snapshot}
            # Record insertion for delayed reward
            self.insert_buffer.append({
                'sparse_leader': sparse_leader,
                'step': step,
                'first_free_follower': first_free_follower,
                'lc_av': selected_av,
                'state': selected_state,
                'sparse_snapshot': sparse_snapshot,
                'tsg_category': 'ce',
                'gate_input': gate_input
            })

            self.ls_free_inserted.append(sparse_leader)
            self.dic_collect_insertedAV[selected_av] = 'free_insert'
            # record free insert AV and tag it
            self.p_basic.dic_tags[selected_av] = 1 # tag as leader
            self.p_basic.dic_AVroleChange[selected_av] = 'free_insert'
            print(f'[FreeInsert] free_inserted_AV: {self.dic_collect_insertedAV}')

        except self.traci.TraCIException:
            print(f"[FreeInsert] {selected_av} insert failed as TraCI exception")

    def _save_model_if_needed(self, current_step, st, model_type='sa'):
        """
        Periodically save the trained model.
        """
        save_interval = 30000
        if current_step > self.next_save_step or current_step == st * 10 - 10:
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            filename = f'free_insert_score_model_{timestamp}.pt'
            self.agent.save_model(filename)
            if current_step == st * 10 - 10:
                self.agent.save_model("final_CA.pt")
            print(f"[Model] Auto-saved at step {current_step}")
            self.next_save_step += save_interval

    def record_loss(self, current_step, st):
        if current_step != st * 10 - 10 or self.mode != 'train':
            return
        self.agent.record_plot_loss()  # plot loss curve

    def record_scores(self, current_step, st):
        """Plot distribution of predicted scores."""
        if current_step != st * 10 - 10 or self.mode != 'train':
            return
        self.agent.record_plot_scores(self.ls_score)
        self.agent.record_plot_score_reward(self.dic_score_reward)
