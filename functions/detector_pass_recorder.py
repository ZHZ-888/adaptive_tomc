class DetectorPassRecorder:
    """
    Detector-based vehicle passing recorder.

    This class records vehicle IDs and their passing times at the MCZ entry
    detector. The insertion order of dic_pass_time represents the passing order.
    """

    def __init__(self, traci, data_recorder):
        self.traci = traci
        self.data_recorder = data_recorder
        self.detector_ids = ['pfz_entry', 'mcz_entry']

        # Vehicle passing time at the detector.
        # Format: {veh_id: pass_time}
        # The insertion order represents the passing order.
        self.dic_pass_time = {'pfz_entry': {}, 'mcz_entry': {}}
        self.ms_exit_speed = None

        # Newly detected vehicles in the current simulation step.
        self.new_pass_vid = {
            "pfz_entry": None,
            "mcz_entry": None
        }
        # {leader_id: platoon_size}
        self.dic_leader_platoon_size = {}

    def update(self, step):
        """
        Update detector passing records.

        This function should be called at every simulation step.
        If a new vehicle passes the detector, record its passing time and store its ID.

        detector_id: 'mcz_entry'; 'pfz_entry'
        """

        c_ts = round(step / 10 + 0.1, 1)

        ms_exit_speed = self.traci.inductionloop.getLastStepMeanSpeed('ms_exit')
        if ms_exit_speed >= 0:
            self.ms_exit_speed = ms_exit_speed

        # Reset newly passed vehicle at each step.
        self.new_pass_vid = {
            "pfz_entry": None,
            "mcz_entry": None
        }
        for detector_id in self.detector_ids:
            try:
                passed_vehicle_ids = (
                    self.traci.inductionloop.getLastStepVehicleIDs(detector_id)
                )
            except self.traci.TraCIException:
                continue

            for veh_id in passed_vehicle_ids:
                # Record the first detector passing time only.
                if veh_id not in self.dic_pass_time[detector_id]:
                    self.dic_pass_time[detector_id][veh_id] = c_ts
                    self.new_pass_vid[detector_id] = veh_id
                    break

        return self.new_pass_vid


    def count_platoon_size(self, ls_leader, detector_id):
        """
        Call: self.new_pass_vid
              self.dic_pass_time

        Count the follower number of the previous leader when a new leader passes
        the detector.
        Return:
            - dic_leader_follower_count: {leader_id: follower_count}
        """

        new_pass_vid = self.new_pass_vid[detector_id]
        dic_pass_time = self.dic_pass_time[detector_id]

        # No new vehicle passes the detector at this step.
        if new_pass_vid is None:
            return self.dic_leader_platoon_size

        # Only trigger counting when the newly passed vehicle is a leader.
        if new_pass_vid not in ls_leader:
            return self.dic_leader_platoon_size

        pass_order = list(dic_pass_time.keys())

        index_this_leader = ls_leader.index(new_pass_vid)

        # The first leader has no previous leader to finalise.
        if index_this_leader == 0:
            return self.dic_leader_platoon_size

        last_leader = ls_leader[index_this_leader - 1]

        # The previous leader must have passed the detector.
        if last_leader not in dic_pass_time:
            return self.dic_leader_platoon_size

        index_last_pass = pass_order.index(last_leader)
        index_this_pass = pass_order.index(new_pass_vid)

        # If the detector passing order is abnormal, skip this case.
        if index_this_pass <= index_last_pass:
            return self.dic_leader_platoon_size

        # Vehicles between the last leader and the current leader are followers
        # of the last leader.
        follower_ids = pass_order[index_last_pass + 1:index_this_pass]
        follower_count = len(follower_ids)

        platoon_size = follower_count + 1
        self.dic_leader_platoon_size[last_leader] = platoon_size

        # update dic_leader_ptype
        try:
            if len(self.data_recorder.dic_leader_ptype[last_leader]) != platoon_size:
                self.data_recorder.dic_leader_ptype[last_leader] = 'A' + 'F' * (platoon_size-1)
        except:
            pass

        return self.dic_leader_platoon_size