from functools import lru_cache


def get_gap_length(g):
    start, end = gaps[g]
    return end - start

def get_batch(l, nl):
    """
    Vehicles from current leader up to (but excluding)
    next leader form the current release batch.

    L: total number of leaders
    l: Current leader index in AV list
    nl: Next leader index in  AV list
    av_positions: [0, 3, 5, 8], avs index in the queue

    nl == L means release all remaining vehicles.
    """
    start = av_positions[l] # l: current leader index
    if nl < L: # next leader index
        end = av_positions[nl]
    else:
        end = len(queue)
    batch = queue[start:end]
    return batch

def required_time(batch):
    '''
    Caculate the required time for a batch of vehicles to merge.
    batch: ['A1', 'H1', 'H2']
    '''
    n = len(batch)
    # Temporary toy formulation
    return 1.753 * n + 0.989

def remaining_vehicles(l):
    '''
    Count the number of vehicles still waiting in the queue after the current leader l.
    L: total number of leaders
    l: current leader index in AV list
    '''
    if l == L:
        return 0

    return len(queue) - av_positions[l]

@lru_cache(None) # Cache the solution of each DP state (g, l) to improve efficiency
def dp(g, l):
    """
    State (g, l):
    g: current mainline gap
    l: current (ramp) leader index in AV list
    nl: Next leader index in  AV list

    L: total number of leaders
    G: total number of gaps
    """
    # All ramp vehicles released
    if l == L:
        return 0, []

    # No more mainline gaps
    if g == G:
        # Penalise vehicles still waiting
        terminal_cost = 20 * remaining_vehicles(l)
        return terminal_cost, []

    # Option 1: HOLD
    # Skip current gap
    future_cost, future_plan = dp(g + 1, l)
    cost_hold = (remaining_vehicles(l) + future_cost)
    best_cost = cost_hold
    best_plan = [("hold", None)] + future_plan

    # Option 2: Select a next leader
    for nl in range(l + 1, L + 1):
        batch = get_batch(l, nl)
        t_req = required_time(batch)
        # Check whether batch fits current gap
        if t_req <= get_gap_length(g):
            future_cost, future_plan = dp(g + 1, nl)
            stage_cost = remaining_vehicles(nl)
            total_cost = (stage_cost + future_cost)
            if total_cost < best_cost:
                best_cost = total_cost
                best_plan = ([(nl,{"batch": batch, "required_time": t_req})]
                             + future_plan)

    return best_cost, best_plan

def print_best_plan(best_plan):
    for g, action in enumerate(best_plan):
        decision, info = action
        print(f"Gap G{g + 1} = {get_gap_length(g):.1f} s")
        if decision == "hold":
            print("  Action: HOLD")
        else:
            if decision < L:
                next_leader = av_ids[decision]
            else:
                next_leader = "END"
            print(f"  Release batch: {info['batch']}")
            print(f"  Required time: {info['required_time']:.1f} s")
            print(f"  Next leader: {next_leader}")

def solve_ramp_schedule(queue_input, gaps_input):
    global queue, gaps, av_positions, av_ids, G, L

    queue = queue_input
    gaps = gaps_input
    av_positions = [i for i, veh in enumerate(queue)
                    if veh.startswith("A")]

    av_ids = [queue[i] for i in av_positions]
    L = len(av_ids)
    G = len(gaps)
    dp.cache_clear()
    return dp(0, 0)


if __name__ == "__main__":
    '''
    stop_times = {'r_av_id': ts, ...} / av_based_rm.py
    '''
    queue_test = ["A1", "H1", "H2",
                  "A2", "H3",
                  "A3", "H4", "H5",
                  "A4"]

    gaps_test = [(10.0, 16.5), # 6.5 s
                 (20.0, 30.0), # 10.0 s
                 (34.0, 37.0)] # 3.0 s

    t_stop = {"A1": 1.0}

    av_positions = [g for g, veh in enumerate(queue_test)
                    if veh.startswith("A")] # AV positions in the queue
    av_ids = [queue_test[g] for g in av_positions]

    L = len(av_ids) # Number of AVs (candidate AVs)
    G = len(gaps_test) # number of gaps

    # 7. Solve from initial state
    best_cost, best_plan = solve_ramp_schedule(queue_test, gaps_test)
    print("Best cost:", best_cost)
    print("Best plan:")
    print_best_plan(best_plan)






