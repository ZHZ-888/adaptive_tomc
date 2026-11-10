# dp_core.py


def _required_time(batch):
    """Calculate the required time for a batch of vehicles to merge.

    Parameters:
        batch: Vehicle IDs in release order, e.g. ['A1', 'H1', 'H2'].
    Returns:
        Required merge time in seconds, using the temporary toy formulation.
    """
    return 1.753 * len(batch) + 0.989

def _get_batch(l, nl, queue,
               av_positions):
    """Vehicles from the current leader up to (excluding) the next leader
    form the current release batch.

    Parameters:
        l: Current leader index in the AV list (not the queue index).
        nl: Next leader index in the AV list. nl == L releases all remaining
            vehicles, where L is the total number of leaders.
        queue: Vehicle IDs in ramp release order.
        av_positions: AV indices in the queue, e.g. [0, 3, 5, 8].
    Returns:
        List of vehicle IDs in the current release batch.
    """
    end = av_positions[nl] if nl < len(av_positions) else len(queue)
    return queue[av_positions[l]:end]

def _remaining_vehicles(l, queue, av_positions):
    """Count waiting vehicles starting from the current leader, inclusive.

    Parameters:
        l: Current leader index in the AV list; l == L means none remain.
        queue: Vehicle IDs in ramp release order.
        av_positions: AV indices in the queue; L = len(av_positions).
    Returns:
        Number of vehicles still waiting.
    """
    return 0 if l == len(av_positions) else len(queue) - av_positions[l]

def _propagate_yield(g, delta, gaps, absorption_gaps,
                     release_times, gap_ids, ls_max_delta_t):
    """Propagate the effect of a local mainline yielding action downstream.

    Parameters:
        g: Current gap index.
        delta: Required yielding time for the current gap.
        gaps: Current usable gap durations.
        release_times: the release time for each gap
        ls_max_delta_t: Maximum allowable yielding duration for each gap
        gap_ids: IDs of the gaps; ['mav1', 'mav2', ...]
    Returns:
        Updated gap list after downstream propagation.
    """
    new_gaps = list(gaps)
    new_absorption_gaps = list(absorption_gaps)
    new_release_times = list(release_times)
    new_ls_max_delta_t = list(ls_max_delta_t)
    mainline_actions = {}
    mainline_delays = {}

    if delta <= 0:
        return (new_gaps, new_absorption_gaps, new_release_times,
                new_ls_max_delta_t, mainline_actions, mainline_delays)

    # current leader actively yields
    leader = gap_ids[g]
    if leader is None:
        return None
    if delta > new_ls_max_delta_t[g]:
        return None

    mainline_actions[leader] = delta
    mainline_delays[leader] = delta
    new_ls_max_delta_t[g] -= delta # occupied delta, then minus delta
    new_gaps[g] += delta # Current gap is enlarged by delta
    new_absorption_gaps[g] += delta
    # This leader becomes the front platoon of the next gap,
    # so the next release opportunity is delayed by the same amount
    if g + 1 < len(new_release_times):
        new_release_times[g + 1] += delta
    remaining_delta = delta
    # Propagate the compression to downstream gaps
    for k in range(g + 1, len(new_gaps)):
        if remaining_delta <= 0:
            break

        absorbed_delta = min(remaining_delta, new_absorption_gaps[k])
        new_gaps[k] -= absorbed_delta
        new_absorption_gaps[k] -= absorbed_delta
        remaining_delta -= absorbed_delta

        if remaining_delta <= 0:
            break
        leader = gap_ids[k]
        if leader is None:
            continue
        required_delta = remaining_delta
        available_delta = max(0.0, new_ls_max_delta_t[k])
        # active_delta is the delay actively applied to the downstream leader.
        active_delta = min(required_delta, available_delta)
        if active_delta > 0:
            mainline_actions[leader] = active_delta
            new_ls_max_delta_t[k] -= active_delta
        # Record the total delay caused by both active and passive responses.
        mainline_delays[leader] = required_delta
        # The total delay, including the passive response, propagates downstream.
        if k + 1 < len(new_release_times):
            new_release_times[k + 1] += required_delta

    return (new_gaps, new_absorption_gaps, new_release_times,
            new_ls_max_delta_t, mainline_actions, mainline_delays)

def _get_mainline_cost(mainline_delays, dic_leader_size):
    """Calculate the cumulative mainline delay cost."""
    cost = 0
    for leader, delta in mainline_delays.items():
        n = dic_leader_size[leader]
        cost += n * delta
    return cost

def _get_ramp_cost(g, l, queue,
                   av_positions, release_times, current_time):
    """Calculate ramp-side waiting cost at the current DP stage."""
    n = _remaining_vehicles(l, queue, av_positions)
    if g == 0:
        time_interval = max(0.0, release_times[g] - current_time)
    else:
        time_interval = max(0.0, release_times[g] - release_times[g-1])
    return n * time_interval

def _dp(g, l, queue,
        gaps, absorption_gaps, release_times,
        ls_max_delta_t, gap_ids, dic_leader_size,
        av_positions, cache,
        current_time, allow_mainline_yield=True):
    """Solve DP state (g, l) using the supplied per-call cache.

    Parameters:
        g: Current mainline gap index; G = len(gaps).
        l: Current ramp leader index in the AV list; L = len(av_positions).
        queue: Vehicle IDs in ramp release order.
        gaps: Mainline gap durations in seconds, in scheduling order.
        release_times:
        ls_max_delta_t: max yielding duration available for each mainline gap
        av_positions: AV indices in the queue.
        cache: Dictionary {(g, l): (best_cost, best_plan)} for this solve only.
        cache: (g, l, future_gaps, future release times) -> (2, 1, (12.0, 6.5, 5.0), (5.0, 12.0, 20.0)))
    Returns:
        best_cost: veh·s
        best_plan: {
            "batch": batch,                    # Ramp vehicles released in this batch
            "required_time": t_req,            # Time required for the batch to merge
            "delta": delta,                    # Additional time required for the current gap
            "mainline_actions": actions,       # Required yielding delays for mainline leaders
            "mainline_cost": mainline_cost,    # Delay cost imposed on the mainline
            }

        The cost accumulates remaining-vehicle waiting time in veh·s.
        penalty of 20 per vehicle left after the final gap.
    """
    L = len(av_positions) # number of AVs
    G = len(gaps) # number of gaps

    # 1. Cached state
    previous_release_time = (
        None if g == 0
        else round(release_times[g - 1], 3))

    state_key = (
        g, l, previous_release_time, tuple(round(x, 3) for x in gaps[g:]),
        tuple(round(x, 3) for x in absorption_gaps[g:]),
        tuple(round(x, 3) for x in release_times[g:]),
        tuple(round(x, 3) for x in ls_max_delta_t[g:]))

    if state_key in cache:
        return cache[state_key]

    # 2. Terminal states
    if l == L: # this leader is the last one, all vehicles are released
        return 0, []
    if g == G: # this gap is the last gap
        remaining = _remaining_vehicles(l, queue, av_positions)
        terminal_wait = 900.0
        return remaining * terminal_wait, [] # terminal_wait is the penalty for unreleased vehicles

    ramp_cost = _get_ramp_cost(
        g, l, queue, av_positions, release_times, current_time)

    # 3. Option 1: HOLD
    future_cost, future_plan = _dp(
        g + 1, l, queue, gaps, absorption_gaps, release_times,
        ls_max_delta_t, gap_ids, dic_leader_size, av_positions,
        cache, current_time, allow_mainline_yield)
    best_cost = ramp_cost + future_cost
    best_plan = [("hold", None)] + future_plan

    # 4. Option 2: RELEASE
    # the last observed AV leads an open platoon and must reamin as the stopping boundary
    for nl in range(l + 1, L):
        batch = _get_batch(l, nl, queue, av_positions)
        t_req = _required_time(batch)

        # Reject batches that exceed the maximum feasible gap
        gap = gaps[g]
        max_delta_t = ls_max_delta_t[g]
        max_feasible_gap = gap + max_delta_t
        if t_req > max_feasible_gap:
            continue

        delta = max(0.0, t_req - gaps[g])
        # DISABLED mainline yielding
        if delta > 0 and not allow_mainline_yield:
            continue
        # mainline leader yielding time for this batch
        # no mainline leader is available to yield
        if delta > 0 and gap_ids[g] is None:
            continue
        if delta > 0:
            propagation_result = _propagate_yield(
                g, delta, gaps, absorption_gaps,
                release_times, gap_ids, ls_max_delta_t)
            if propagation_result is None:
                continue
            (new_gaps, new_absorption_gaps, new_release_times,
             new_ls_max_delta_t, mainline_actions, mainline_delays) \
                = propagation_result
            mainline_cost = _get_mainline_cost(
                mainline_delays,
                dic_leader_size)
        else:
            new_gaps = gaps
            new_absorption_gaps = absorption_gaps
            new_release_times = release_times
            new_ls_max_delta_t = ls_max_delta_t
            mainline_actions = {}
            mainline_delays = {}
            mainline_cost = 0

        # Future optimal solution after releasing this batch
        future_cost, future_plan = _dp(
            g + 1, nl, queue, new_gaps, new_absorption_gaps, new_release_times,
            new_ls_max_delta_t, gap_ids, dic_leader_size, av_positions, cache,
            current_time, allow_mainline_yield)
        total_cost = ramp_cost + mainline_cost + future_cost
        # Keep the better decision
        if (total_cost < best_cost or
                (total_cost == best_cost and best_plan[0][0] == "hold")):
            best_cost = total_cost
            best_plan = ([(nl, {
                "batch": batch,
                "required_time": t_req,
                "delta": delta,
                "mainline_actions": mainline_actions,
                "mainline_delays": mainline_delays,
                "mainline_cost": mainline_cost
            })] + future_plan)

    # 5. Save and return
    result = (best_cost, best_plan)
    cache[state_key] = result

    return result

def _solve_full_plan(queue_input, gaps_input, dic_leader_size,
                     current_time, allow_mainline_yield=True):
    """Schedule {leader_id: [gap_seconds, leader_release_time]} in insertion order.

    Nonpositive gaps result in HOLD. Each call owns its data and DP cache.

    Parameters:
        queue_input: Vehicle IDs in release order, beginning with an AV leader,
            e.g. ['rav20', 'rhv46', 'rav99'].
        gaps_input: {m_leader: [gap_seconds, leader_release_time, max_delta_t, absorption_gap]},
                    e.g. {'mav50': [6.5, 12.3, 3.0]}.
        dic_leader_size: {leader: platoon_size, ...}
    Returns:
        (best_cost, best_plan), using the cost and action format in _dp().
        best_cost: veh·s
        best_plan:
            each decision is either ("hold", None)
                                 or (nl, info)
                nl: Index of the next unreleased ramp AV in av_ids.
                info:
                {
                "batch": batch,                    # Ramp vehicles released in this batch
                "required_time": t_req,            # Time required for the batch to merge
                "delta": delta,                    # Additional time required for the current gap
                "mainline_actions": actions,       # Required yielding delays for mainline leaders
                "mainline_cost": mainline_cost,    # Delay cost imposed on the mainline
                }
    """
    queue = list(queue_input)
    av_positions = [i for i, veh in enumerate(queue) if veh.startswith("rav")]

    gap_info = list(gaps_input.values())
    gap_ids = list(gaps_input.keys())

    gaps = [item[0] for item in gap_info]
    release_times = [item[1] for item in gap_info]
    ls_max_delta_t = [item[2] for item in gap_info]
    absorption_gaps = [item[3] for item in gap_info]

    best_cost, best_plan = _dp(0, 0, queue, gaps, absorption_gaps, release_times,
                               ls_max_delta_t, gap_ids, dic_leader_size, av_positions,
                               {}, current_time, allow_mainline_yield)

    return best_cost, best_plan

def solve_ramp_schedule(current_time, queue_input, gaps_input,
                        dic_leader_size, stop_state,
                        allow_mainline_yield=True):
    """Return the first scheduled release using the current queue and gaps.

    Parameters:
        queue_input: Ramp vehicle IDs in release order, starting with a rav leader.
        gaps_input: {m_leader: [gap_seconds, leader_release_time, max_delta_t]}, in insertion order.
            A None key identifies a rear window without a mainline leader.
        dic_leader_size: {leader: platoon_size, ...}
    Returns:
        released_leader:
        next_leader:
        chosen_m_leader:
        mainline_actions: {'mav1': 4.248, 'mav2': 1.248}
    """
    if not stop_state:
        return None, None, None, {}
    queue = list(queue_input)
    # Keep both local variables available for inspection in the debugger.
    best_cost, best_plan = _solve_full_plan(queue, gaps_input, dic_leader_size,
                                            current_time, allow_mainline_yield)
    print(f"Best plan cost: {best_cost:.3f} veh·s")
    print_best_plan(best_plan, queue, gaps_input)
    av_ids = [veh for veh in queue if veh.startswith("rav")]
    gap_ids = list(gaps_input)
    for g, (decision, info) in enumerate(best_plan):
        if decision != "hold":
            next_leader = av_ids[decision] if decision < len(av_ids) else None
            released_leader = info['batch'][0]
            chosen_m_leader = gap_ids[g]

            if released_leader in ['rav423']:
                pass

            return released_leader, next_leader, chosen_m_leader, info['mainline_actions']
    return None, None, None, {}

def print_best_plan(best_plan, queue_input, gaps_input):
    """Print using the same inputs that produced the plan.

    Parameters:
        best_plan: Action list returned by _solve_full_plan().
        queue_input: Original queue used to solve this plan.
        gaps_input: Original gap dictionary, with the same insertion order.
    Returns:
        None. Prints gap IDs, durations and release times, HOLD or release batches, required
        merge times, and the next leader (END when all vehicles are released).
    """
    gap_items = list(gaps_input.items())
    av_ids = [veh for veh in queue_input if veh.startswith("rav")]
    for g, (decision, info) in enumerate(best_plan):
        gap_id, gap_info = gap_items[g]
        gap_seconds = gap_info[0]
        leader_release_time = gap_info[1]
        max_delta_t = gap_info[2]

        print(f"\nGap G{g + 1} ({gap_id})")
        print(f"  Original gap: {gap_seconds:.1f} s")
        print(f"  Original release time: {leader_release_time:.1f} s")
        print(f"  Maximum yielding delay: {max_delta_t:.1f} s")

        if decision == "hold":
            print("  Action: HOLD")
        else:
            next_leader = av_ids[decision] if decision < len(av_ids) else "END"

            formatted_actions = {
                leader: round(delta, 3)
                for leader, delta in info["mainline_actions"].items()}

            formatted_delays = {
                leader: round(delta, 3)
                for leader, delta in info["mainline_delays"].items()}

            print(f"  Release batch: {info['batch']}")
            print(f"  Required time: {info['required_time']:.3f} s")
            print(f"  Required yielding: {info['delta']:.3f} s")
            print(f"  Mainline actions: {formatted_actions}")
            print(f"  Mainline delays: {formatted_delays}")
            print(f"  Mainline cost: {info['mainline_cost']:.3f}")
            print(f"  Next ramp leader: {next_leader}")


class DPScheduler:
    """Store and periodically update the current DP scheduling decision."""

    def __init__(self):
        self.reset()

    def update(self, step, current_time, update_interval, queue_input,
               gaps_input, dic_leader_size,
               stop_state, allow_mainline_yield=True):
        """
        Update the stored DP decision at the configured interval.
        Params:
            leader_sizes: {leader_id, platoon_size}
        Returns:
            dp_updated: True if the DP decision was updated; otherwise False.
            mainline_actions: Newly generated mainline actions, or None if not updated.
        """
        if not stop_state:
            return False, None

        # Keep the current plan until its release is completed.
        if self.released_leader is not None:
            return False, None

        if (self.last_update_step is not None
                and step - self.last_update_step < update_interval):
            return False, None

        (self.released_leader, self.next_leader,
         self.chosen_m_leader, self.mainline_actions) = solve_ramp_schedule(
            current_time, queue_input, gaps_input,
            dic_leader_size, stop_state, allow_mainline_yield)

        if self.released_leader == 'rav1844':
            pass

        gap_info = (gaps_input.get(self.chosen_m_leader)
                    if self.released_leader is not None else None)
        self.release_ts = gap_info[1] if gap_info is not None else None
        self.last_update_step = step

        return True, self.mainline_actions

    def is_release_due(self, current_time):
        if self.released_leader is None:
            return False
        if self.release_ts is None:
            return False
        if current_time < self.release_ts:
            return False
        return True

    def reset(self):
        """Clear the stored DP scheduling decision."""
        self.released_leader = None
        self.next_leader = None
        self.chosen_m_leader = None
        self.mainline_actions = {}
        self.release_ts = None
        self.last_update_step = None


if __name__ == "__main__":
    c_ts = 1.0

    queue_test = [
        'rav1', 'rhv1',
        'rav2', 'rhv2']

    gaps_test = {
        'mav1': [3.0, 0.0, 2.0],
        'mav2': [5.0, 5.0, 2.0],
        'mav3': [5.0, 10.0, 2.0]}

    dic_leader_size = {
        'mav1': 2,
        'mav2': 4,
        'mav3': 3}

    released_leader, next_leader, chosen_m_leader, mainline_actions = solve_ramp_schedule(
        c_ts, queue_test, gaps_test, dic_leader_size,
        stop_state=True, allow_mainline_yield=True)
    print("First scheduled release:", released_leader, next_leader, chosen_m_leader)
    formatted_actions = {
        leader: round(delta, 3)
        for leader, delta in mainline_actions.items()}
    print("Mainline actions:", formatted_actions)
