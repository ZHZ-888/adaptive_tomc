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

def _dp(g, l, queue,
        gaps, av_positions, cache):
    """Solve DP state (g, l) using the supplied per-call cache.

    Parameters:
        g: Current mainline gap index; G = len(gaps).
        l: Current ramp leader index in the AV list; L = len(av_positions).
        queue: Vehicle IDs in ramp release order.
        gaps: Mainline gap durations in seconds, in scheduling order.
        av_positions: AV indices in the queue.
        cache: Dictionary {(g, l): (best_cost, best_plan)} for this solve only.
    Returns:
        (best_cost, best_plan) for the remaining gaps and vehicles.
        Each plan entry is ('hold', None) or
        (nl, {'batch': vehicle_ids, 'required_time': seconds}), where nl is
        the next leader index and nl == L means all vehicles are released.
        The cost adds waiting-vehicle counts after each gap and a terminal
        penalty of 20 per vehicle left after the final gap.

        leader index counts from 0
    """
    L = len(av_positions) # number of AVs
    G = len(gaps) # number of gaps

    # 1. Cached state
    if (g, l) in cache:
        return cache[g, l]

    # 2. Terminal states
    if l == L: # this leader is the last one, all vehicles are released
        return 0, []
    if g == G: # this gap is the last gap
        stage_cost = _remaining_vehicles(l, queue, av_positions)
        return 20 * stage_cost, [] # 20 is the penalty for unreleased vehicles

    # 3. Option 1: HOLD
    future_cost, future_plan = _dp(
        g + 1, l, queue, gaps, av_positions, cache)
    stage_cost = _remaining_vehicles(
        l, queue, av_positions)
    best_cost = stage_cost + future_cost
    best_plan = [("hold", None)] + future_plan

    # 4. Option 2: RELEASE
    for nl in range(l + 1, L + 1):
        batch = _get_batch(l, nl, queue, av_positions)
        t_req = _required_time(batch)
        # Batch cannot use this gap
        if t_req > gaps[g]:
            continue
        # Future optimal solution after releasing this batch
        future_cost, future_plan = _dp(
            g + 1, nl,
            queue, gaps, av_positions, cache)
        stage_cost = _remaining_vehicles(
            nl, queue, av_positions)
        total_cost = stage_cost + future_cost
        # Keep the better decision
        if total_cost < best_cost:
            best_cost = total_cost
            best_plan = ([(nl,
                          {"batch": batch, "required_time": t_req})]
                         + future_plan)

    # 5. Save and return
    result = (best_cost, best_plan)
    cache[g, l] = result

    return result

def _solve_full_plan(queue_input, gaps_input):
    """Schedule {leader_id: gap_seconds} in insertion order.

    Nonpositive gaps result in HOLD. Each call owns its data and DP cache.

    Parameters:
        queue_input: Vehicle IDs in release order, beginning with an AV leader,
            e.g. ['rav20', 'rhv46', 'rav99']. IDs starting with 'rav' are leaders.
        gaps_input: {m_leader: gap_seconds}, e.g. {'mav50': 6.5}.
            Dictionary insertion order defines the scheduling order.
    Returns:
        (best_cost, best_plan), using the cost and action format in _dp().
        best_cost: number
        best_plan: [('hold', None),
                    (1, {'batch': ['rav20', 'rhv46', 'rhv67', 'rhv82'], 'required_time': 8.001}),
                    (2, {'batch': ['rav99', 'rhv128', 'rhv141', ...], 'required_time': 9.754}),
                    (6, {'batch': ['rav251', 'rav275', 'rav297', ...], 'required_time': 11.507})]

                    (nl, {'batch': vehicle_ids, 'required_time': seconds})
    """
    queue = list(queue_input)
    gaps = list(gaps_input.values())
    av_positions = [i for i, veh in enumerate(queue) if veh.startswith("rav")]
    return _dp(0, 0, queue, gaps, av_positions, {})

def solve_ramp_schedule(queue_input, gaps_input, stop_state):
    """Return the first scheduled release using the current queue and gaps.

    Parameters:
        queue_input: Ramp vehicle IDs in release order, starting with a rav leader.
        gaps_input: {m_leader: gap_seconds}, in chronological insertion order.
            A None key identifies a rear window without a mainline leader.
    Returns:
        (released_leader, next_leader, chosen_m_leader): the first release's
        ramp leader, the first leader left waiting, and the chosen gap key.
        next_leader is None when the batch releases all remaining vehicles.
        Returns (None, None, None) if no release is planned.
        A future chosen gap is a scheduled release, not an immediate command.
    """
    if not stop_state:
        return None, None, None
    queue = list(queue_input)
    # Keep both local variables available for inspection in the debugger.
    best_cost, best_plan = _solve_full_plan(queue, gaps_input)
    print_best_plan(best_plan, queue, gaps_input)
    av_ids = [veh for veh in queue if veh.startswith("rav")]
    gap_ids = list(gaps_input)
    for g, (decision, info) in enumerate(best_plan):
        if decision != "hold":
            next_leader = av_ids[decision] if decision < len(av_ids) else None
            released_leader = info['batch'][0]
            choosen_m_leader = gap_ids[g]
            return released_leader, next_leader, choosen_m_leader
    return None, None, None

def print_best_plan(best_plan, queue_input, gaps_input):
    """Print using the same inputs that produced the plan.

    Parameters:
        best_plan: Action list returned by _solve_full_plan().
        queue_input: Original queue used to solve this plan.
        gaps_input: Original gap dictionary, with the same insertion order.
    Returns:
        None. Prints gap IDs and durations, HOLD or release batches, required
        merge times, and the next leader (END when all vehicles are released).
    """
    gap_items = list(gaps_input.items())
    av_ids = [veh for veh in queue_input if veh.startswith("rav")]
    for g, (decision, info) in enumerate(best_plan):
        gap_id, gap_seconds = gap_items[g]
        print(f"Gap G{g + 1} ({gap_id}) = {gap_seconds:.1f} s")
        if decision == "hold":
            print("  Action: HOLD")
        else:
            next_leader = av_ids[decision] if decision < len(av_ids) else "END"
            print(f"  Release batch: {info['batch']}")
            print(f"  Required time: {info['required_time']:.1f} s")
            print(f"  Next leader: {next_leader}")

if __name__ == "__main__":
    queue_test2 = ['rav20', 'rhv46', 'rhv67', 'rhv82',
                   'rav99', 'rhv128', 'rhv141', 'rhv155', 'rhv204',
                   'rav251',
                   'rav275',
                   'rav297', 'rhv309',
                   'rav324', 'rhv339']

    gaps_test2 = {'mav17': 7.0,
                  'mav50': 8.5,
                  'mav243': 10.0,
                  'mbav367': 12.0}

    released_leader, next_leader, chosen_m_leader = solve_ramp_schedule(
        queue_test2, gaps_test2, stop_state=True)
    print("First scheduled release:", released_leader, next_leader, chosen_m_leader)
