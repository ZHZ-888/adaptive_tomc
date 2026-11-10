"""Generate and execute synthetic mainline yielding actions for RF data collection."""


class RFActionController:
    def __init__(self, data_recorder, merge_regular, rng,
                 yield_ratio_ranges=None):
        """
        Args:
            data_recorder: DataRecording instance containing current vehicle states.
            merge_regular: MergingControlRegular instance used to build and apply actions.
            rng: random.Random instance used for reproducible sampling.
            yield_ratio_ranges: Continuous sampling strata, e.g. ((0.1, 0.3), (0.3, 0.5)).
        """
        if yield_ratio_ranges is None:
            yield_ratio_ranges = (
                (0.0, 0.0),
                (0.10, 0.30),
                (0.30, 0.50),
                (0.50, 0.70),
                (0.70, 0.90))
            # yield_ratio_ranges = [(0.0, 0.0)]

        if rng is None:
            raise ValueError("An action-data random generator is required.")

        self.data_recorder = data_recorder
        self.merge_regular = merge_regular
        self.traj_optimiser = merge_regular.traj_optimiser
        self.rng = rng
        self.yield_ratio_ranges = yield_ratio_ranges
        self.pending_yield_ratio_ranges = []

        self.action_leader = None
        self.action_end_ts = None
        self.sampled_leaders = set()
        self.dic_action_params = {}

    def prepare_action(self, step, dic_platoon_info,
                       dic_mplatoon_et, dic_vid_groups,
                       update_interval):
        """
        Generate and store a new feasible action without executing it.

        Args:
            step: Current simulation step, e.g. 1200 for 120.1 s.
            dic_platoon_info: {leader: [[platoon_type, tail_id], length_history]}.
            dic_mplatoon_et: {leader: [platoon_type, head_ts, tail_ts, update_ts]}.
            dic_vid_groups: Current vehicle-group dictionary from DataRecording.
            update_interval: Action-generation interval in simulation steps, e.g. 60.
        Returns:
            new_action_demand: Newly generated action-demand dictionary, or None.
        """
        c_ts = round(step / 10 + 0.1, 1)
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']
        new_action_demand = None

        action_due = (
            self.action_leader is None
            and step % update_interval == 0)

        if action_due:
            action_demand = self._generate_action_demand(
                c_ts, dic_platoon_info, dic_mplatoon_et,
                ls_m_leader_up_asc)

            if action_demand is not None:
                action_params = self._build_action_params(action_demand)
                if action_params is not None:
                    self._start_action(action_demand, action_params)
                    new_action_demand = action_demand

        return new_action_demand

    def execute_action(self, step, dic_vid_groups):
        """
        Execute the active action and clear it after completion.

        Args:
            step: Current simulation step.
            dic_vid_groups: Current vehicle-group dictionary from DataRecording.
        Returns:
            action_m_leader: Leader controlled in the current step, or None.
        """
        c_ts = round(step / 10 + 0.1, 1)
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']

        action_m_leader = self._apply_action(
            step, ls_m_leader_up_asc)
        self._finish_action(c_ts, ls_m_leader_up_asc)

        return action_m_leader

    def _sample_yield_ratio(self):
        """
        Sample one continuous yield ratio from each stratum per cycle.

        Returns:
            yield_ratio: Continuous action-strength ratio between 0 and 1.
        """
        if not self.pending_yield_ratio_ranges:
            self.pending_yield_ratio_ranges = list(
                self.yield_ratio_ranges)
            self.rng.shuffle(self.pending_yield_ratio_ranges)

        lower, upper = self.pending_yield_ratio_ranges.pop()
        yield_ratio = self.rng.uniform(lower, upper)

        return yield_ratio

    def _generate_action_demand(
            self, c_ts, dic_platoon_info,
            dic_mplatoon_et, ls_m_leader_up_asc):
        """
        Select an eligible leader and generate a feasible yielding demand.

        Args:
            c_ts: Current simulation time in seconds.
            dic_platoon_info: {leader: [[platoon_type, tail_id], length_history]}.
            dic_mplatoon_et: {leader: [platoon_type, head_ts, tail_ts, update_ts]}.
            ls_m_leader_up_asc: Mainline leader IDs before the merging point.
        Returns:
            action_demand: {
                'leader': str,
                'action_start_ts': float,
                'est_head_dur': float,
                'max_delta_t': float,
                'yield_ratio': float,
                'yield_dur': float,
                'desired_duration': float,
                'target_head_ts': float,
            }, or None when no feasible leader exists.
        """
        action_candidates = []

        for leader in ls_m_leader_up_asc:
            if leader in self.sampled_leaders:
                continue

            platoon_type = self.data_recorder.dic_leader_ptype.get(leader)
            platoon_info = dic_platoon_info.get(leader)
            tail_id = platoon_info[0][1] if platoon_info is not None else None

            if platoon_type is None or len(platoon_type) <= 1:
                continue
            if tail_id is None or leader not in dic_mplatoon_et:
                continue

            states = self.data_recorder.get_vid_states(leader)
            dis = states['dis']
            v0 = states['v']

            if dis is None or v0 is None or dis <= 0:
                continue

            est_head_dur = (
                self.merge_regular.estimate_travel_time(v0, dis))
            max_delta_t = self.traj_optimiser.get_max_delta_t(
                v0, dis, est_head_dur)

            if max_delta_t <= 0:
                continue

            action_candidates.append({
                'leader': leader,
                'action_start_ts': c_ts,
                'est_head_dur': est_head_dur,
                'max_delta_t': max_delta_t})

        if not action_candidates:
            return None

        action_demand = self.rng.choice(action_candidates)
        yield_ratio = self._sample_yield_ratio()
        yield_dur = yield_ratio * action_demand['max_delta_t']
        desired_duration = (
            action_demand['est_head_dur'] + yield_dur)

        action_demand['yield_ratio'] = yield_ratio
        action_demand['yield_dur'] = yield_dur
        action_demand['desired_duration'] = desired_duration
        action_demand['target_head_ts'] = (
            c_ts + desired_duration)

        return action_demand

    def _build_action_params(self, action_demand):
        """
        Convert a yielding demand into executable trajectory parameters.

        Args:
            action_demand: Dictionary returned by _generate_action_demand().
        Returns:
            action_params: [t1, a1, t3, a3, v_reach, action_start_ts], or None.
        """
        leader = action_demand['leader']
        states = self.data_recorder.get_vid_states(leader)
        dis = states['dis']
        v0 = states['v']
        desired_duration = action_demand['desired_duration']

        action_params = list(
            self.merge_regular.get_action_params(
                desired_duration, dis, v0))
        action_params.append(action_demand['action_start_ts'])

        action_is_feasible = (
            len(action_params) >= 6
            and action_params[0] is not None)

        if not action_is_feasible:
            return None

        return action_params

    def _start_action(self, action_demand, action_params):
        """
        Store a newly generated action as the active action.

        Args:
            action_demand: Dictionary returned by _generate_action_demand().
            action_params: [t1, a1, t3, a3, v_reach, action_start_ts].
        Returns:
            None.
        """
        leader = action_demand['leader']
        action_start_ts = action_demand['action_start_ts']

        self.action_leader = leader
        self.action_end_ts = (
            action_start_ts + action_params[0] + action_params[2])
        self.sampled_leaders.add(leader)
        self.dic_action_params = {leader: action_params}

    def _apply_action(self, step, ls_m_leader_up_asc):
        """
        Apply the active action at the current simulation step.

        Args:
            step: Current simulation step.
            ls_m_leader_up_asc: Mainline leader IDs before the merging point.
        Returns:
            action_m_leader: Leader controlled in the current step, or None.
        """
        valid_actions = {
            leader: action_params
            for leader, action_params in self.dic_action_params.items()
            if leader in ls_m_leader_up_asc}

        if valid_actions:
            self.merge_regular.apply_leader_action(step, valid_actions)
            self.merge_regular.flashing_merging(
                step, list(valid_actions))

        action_m_leader = next(iter(valid_actions), None)
        return action_m_leader

    def _finish_action(self, c_ts, ls_m_leader_up_asc):
        """
        Clear an action after it finishes or leaves the control area.

        Args:
            c_ts: Current simulation time in seconds.
            ls_m_leader_up_asc: Mainline leader IDs before the merging point.
        Returns:
            None.
        """
        if self.action_leader is None:
            return

        action_finished = (
            self.action_leader not in ls_m_leader_up_asc
            or c_ts >= self.action_end_ts)

        if not action_finished:
            return

        self.dic_action_params = {}
        self.action_leader = None
        self.action_end_ts = None
