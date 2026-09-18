# platoon_lane_manager.py
# Lane change behavior management near weaving section

import random


class PlatoonLaneManager:
    def __init__(self, traci, data_recorder):
        self.traci = traci
        self.data_recorder = data_recorder
        self.max_speed = self.data_recorder.max_speed


        self.encourage_change_mark = set()  # record id that has been order to change to inner lane
        self.lcKeepRight_disabled = set()
        self.pending_changes = set()
        self.lc_restricted_av = set() # no lc hv
        self.leader_move_commands_sent = set()
        self.no_lc_hv = set()  # Lane change control
        self.std_leaders_done = set() # record standard platoon leaders processed for av_fol jump innner lane

    def encourage_inner_lane_change(
            self,
            ls_ihA_hv: list,
            length_ih,
            p_to_inner: float = 0.8,
            weaving_influence_range: float = 200.0,
    ):
        """
        Let HVs on the outer lane (lane 0) near the ramp
        have a random tendency to move to the inner lane (lane 1).

        Parameters
        ----------
        ls_ihA_hv : list
            Vehicle IDs of HVs currently on outer lane (lane 0).
        p_to_inner : float
            Probability (0–1) that a HV attempts to change to the inner lane.
        weaving_influence_range : float
            Distance range (m) on inflow_highway influenced by the weaving which
            the inner-lane changing.
        """
        if not ls_ihA_hv:
            return
        for veh_id in ls_ihA_hv:
            if veh_id in self.encourage_change_mark:
                continue
            try:
                lane_id = self.traci.vehicle.getLaneIndex(veh_id)
                if lane_id != 0:
                    continue
                lane_pos = self.traci.vehicle.getLanePosition(veh_id)  # encourage only vehicles on lane 0
                dis_to_weaving = length_ih - lane_pos  # distance to the weaving section
            except Exception:
                continue  # vehicle might have left the network
            # only apply within upstream distance (e.g. 200 m before ramp)
            if dis_to_weaving > weaving_influence_range:
                continue
            self.encourage_change_mark.add(veh_id)
            # random tendency: some HVs will change, some not
            if random.random() > p_to_inner:
                continue
            try:
                # lane 1 = inner lane (away from ramp)
                self.traci.vehicle.changeLane(veh_id, 1, 3)
            except Exception:
                continue  # skip if unsafe or invalid

    def restrict_av_lc(self, ls_id):
        '''
        forbid auto lane_change

        0: Disable all lane changes (complete override)
        256 (0b100000000): Disable strategic lane changes only//own route needs
        512 (0b1000000000): Disable cooperative lane changes//help others
        1621 (default): Full autonomous mode with all lane change reasons enabled

        :param ls_id: list of veh id
        :return:
        '''
        for vid in ls_id:
            if vid not in self.lc_restricted_av:
                self.traci.vehicle.setLaneChangeMode(vid, 256) # 256
                self.lc_restricted_av.add(vid)

    def manage_hv_lc_behaviour(self, lc, dic_tags):
        '''
        manage hv lane changing behavior
        '''
        ls_HV_followers = [v for v, t in dic_tags.items() if t == 0]

        if not lc: # lc == False
            # Only disable lane changing for followers that haven't been processed yet
            to_disable = set(ls_HV_followers) - self.no_lc_hv
            for vid in to_disable:
                try:
                    self.traci.vehicle.setLaneChangeMode(vid, 0) # disables all automatic lane changes
                except Exception:
                    pass
                self.no_lc_hv.add(vid)
        else: # lc == True
            pass

    def manage_lc_behavior_near_ws(self, lc, ls_ihAB_hv, ls_wsBC_hv, length_ih,
                                   p_to_inner=0.8, weaving_influence_range=200.0):
        """
        Adaptive lane-changing control near ramp.
        """
        if not lc:  # if lc (lane_changing) is False

            return
        # if lc is True
        self._disable_keepRight_in_weaving(ls_ihAB_hv, ls_wsBC_hv)
        # self._encourage_outer_to_inner(ls_ihAB_hv, length_ih, p_to_inner, weaving_influence_range)
        # self._cancel_pending_changes_on_center()
        # self._restore_keepRight_outside_weaving(length_ih, weaving_influence_range)

    def move_leader_no_fol_to_inner(self, ls_leader_AV, dic_platoon_members):
        """
        Encourage an AV leader with no followers to move from the outter lane to the inner lane.

        :param ls_leader_AV: AV leader list, ascending order
        :param dic_platoon_members: Platoon membership dictionary
        """
        # avoid the first emerged AV jump to outer lane
        commands = []
        ls_leader_AV_filtered = ls_leader_AV[:-1]
        for leader_id in ls_leader_AV_filtered:
            if leader_id == 'mav38':
                pass
            # if "b_av" in leader_id: # for compare tsg on/off
            #     continue
            # Check if the AV leader has no followers
            followers = dic_platoon_members.get(leader_id, [])[1:]  # Exclude the leader itself
            if not followers and leader_id not in self.leader_move_commands_sent:
                try:
                    # Get the current lane of the AV leader
                    current_lane = self.traci.vehicle.getLaneIndex(leader_id)
                    # Ensure the AV is in the outter lane (lane 0)
                    if current_lane == 0:
                        commands.extend([('change_lane', leader_id, 1, 30),
                                         ('set_max_speed', leader_id, self.max_speed)])
                        self.leader_move_commands_sent.add(leader_id)
                except Exception as e:
                    print(f"Error encouraging AV leader {leader_id} to outer lane: {e}")
        return commands

    def move_av_fol_to_inner(self, ls_leader_AV, dic_standard_platoon, lc_fol_av):
        '''
        encourage_av_fol_to_out_lane (original name)
        Parameters
        ----------
        self.std_leaders_done:
        dic_standard_platoon

        Returns
        -------
        '''
        if not lc_fol_av or not dic_standard_platoon:
            return []
        ordered = [k for k in ls_leader_AV if k in dic_standard_platoon]
        if len(ordered) < 2:
            return []

        second_to_last_leader = ordered[-2]
        if second_to_last_leader in self.std_leaders_done:
            return []
        members = dic_standard_platoon.get(second_to_last_leader, [])
        followers = members[1:]
        commands = [('change_lane', fol, 1, 30) for fol in followers if 'av' in fol]
        self.std_leaders_done.add(second_to_last_leader) # avoid repeat loop
        return commands

    def execute_av_commands(self, commands):
        """Execute AV lane-management commands."""
        active_ids = set(self.traci.vehicle.getIDList())
        for command, vid, *args in commands:
            if vid not in active_ids:
                continue
            if command == 'change_lane':
                self.traci.vehicle.changeLane(vid, *args)
            elif command == 'set_max_speed':
                self.traci.vehicle.setMaxSpeed(vid, *args)


    def _disable_keepRight_in_weaving(self, ls_ihAB_hv, ls_wsBC_hv):
        "'_' for internal use within a class or module and not part of the public API."
        all_influenced_hv = set(ls_ihAB_hv) | set(ls_wsBC_hv)
        for veh_id in all_influenced_hv:
            if veh_id in self.lcKeepRight_disabled:
                continue
            try:
                self.traci.vehicle.setParameter(veh_id, "laneChangeModel.lcKeepRight", "0")
                self.lcKeepRight_disabled.add(veh_id)
            except Exception:
                continue

    def _restore_keepRight_outside_weaving(self, length_ih, weaving_influence_range):
        """
        Restore lcKeepRight=1 for HVs that have left the weaving-influenced region.
        The weaving zone = last `weaving_influence_range` meters of ih + entire ws.
        Vehicles entering 'center' or moving upstream beyond this zone are reset.
        """
        hv_to_reset = set()
        threshold = length_ih - weaving_influence_range
        for veh_id in list(self.lcKeepRight_disabled):
            try:
                lane_pos = self.traci.vehicle.getLanePosition(veh_id)
                road_id = self.traci.vehicle.getRoadID(veh_id)
            except Exception:
                continue

            # case 1: upstream (ih, before influence zone)
            if road_id.startswith("ih") and lane_pos < threshold:
                hv_to_reset.add(veh_id)

            # case 2: downstream (entered center)
            elif road_id.startswith("center"):
                hv_to_reset.add(veh_id)

            # case 3: still inside weaving (ih tail or ws)
            else:
                continue

        for veh_id in hv_to_reset:
            try:
                self.traci.vehicle.setParameter(veh_id, "laneChangeModel.lcKeepRight", "1")
            except Exception:
                pass
            self.lcKeepRight_disabled.discard(veh_id)

    def _encourage_outer_to_inner(self, ls_ihAB_hv, length_ih, p_to_inner, weaving_influence_range):
        for veh_id in ls_ihAB_hv:
            if veh_id in self.encourage_change_mark:
                continue
            try:
                lane_pos = self.traci.vehicle.getLanePosition(veh_id)
                dis_to_weaving = length_ih - lane_pos
                if dis_to_weaving > weaving_influence_range:
                    continue
                lane_id = self.traci.vehicle.getLaneIndex(veh_id)
                if lane_id != 0:
                    continue
                self.encourage_change_mark.add(veh_id)
                if random.random() <= p_to_inner:
                    self.traci.vehicle.changeLane(veh_id, 1, 1)
                    self.pending_changes.add(veh_id)  # Record: this vehicle received a change command
            except Exception:
                continue

    def _cancel_pending_changes_on_center(self):
        """
        Cancel any pending lane-change commands that were issued in the weaving region
        but never successfully executed before entering 'center'.
        """
        for vid in list(self.pending_changes):
            try:
                road_id = self.traci.vehicle.getRoadID(vid)
            except Exception:
                self.pending_changes.discard(vid)
                continue

            if road_id.startswith("center"):
                if vid == 'mbhv1162' or vid == 'mvhv1198':
                    pass
                try:
                    current_lane = self.traci.vehicle.getLaneIndex(vid)
                    # Cancel by re-commanding the current lane (duration=0)
                    self.traci.vehicle.changeLane(vid, current_lane, 0)
                except Exception:
                    pass
                self.pending_changes.discard(vid)
