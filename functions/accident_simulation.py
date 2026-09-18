accident_state = False

def sudden_accident(traci, step,
                    data_recorder, stop_time=400,
                    stop_position=100):
    '''
    simulation of a sudden vehicle stop incident
    :param stop_time:
    :param stop_position: on the merge lane (center_0), suggest stop_pos = 50
    :return:
    '''
    global accident_state
    c_ts = round(step/10 + 0.1, 1)

    if c_ts < stop_time:
        return

    if accident_state:
        return

    ls_centerA_asc = data_recorder.record_multi_lane_info()['ls_centerA_asc']
    stop_duration = 60 # second

    for id in ls_centerA_asc:
        pos = data_recorder.dic_pos[id]
        if pos < stop_position:
            try:
                traci.vehicle.setStop(id, 'center', stop_position, laneIndex=0, duration=stop_duration)
                accident_state = True
                print(f'stop_id:{id}')
                break
            except Exception as e:
                print(f"setStop failed for {id}: {e}")
                continue