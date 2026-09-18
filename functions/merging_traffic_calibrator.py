"""Shared human-driver calibration for the motorway merging area."""


class MergingTrafficCalibrator:
    """Temporarily reduce HV keep-right tendency inside the merging area."""

    KEEP_RIGHT_PARAMETER = "laneChangeModel.lcKeepRight"

    def __init__(self, traci, entry_edge="inflow_highway", entry_pos=1200.0,
                 exit_edge="center", exit_pos=2.0, keep_right=0.1,
                 cleanup_interval=100):
        self.traci = traci
        self.entry_edge = entry_edge
        self.entry_pos = entry_pos
        self.exit_edge = exit_edge
        self.exit_pos = exit_pos
        self.keep_right = keep_right
        self.cleanup_interval = cleanup_interval

        # veh_id -> value in effect immediately before crossing the entry section.
        self._original_keep_right = {}
        self._update_count = 0

    @staticmethod
    def _looks_like_hv(veh_id, type_id):
        """Match the HV identifiers/types used by the project."""
        return "hv" in veh_id.lower() or type_id.startswith("hv_")

    def update(self):
        """Apply entry/exit calibration once per HV, independent of lane."""
        self._update_count += 1

        # Scan only the entry edge. This edge query covers all of its lanes,
        # unlike the single-lane induction loop at the same cross-section.
        try:
            entry_ids = self.traci.edge.getLastStepVehicleIDs(self.entry_edge)
        except Exception:
            entry_ids = ()

        for veh_id in entry_ids:
            if veh_id in self._original_keep_right:
                continue
            try:
                if self.traci.vehicle.getLanePosition(veh_id) < self.entry_pos:
                    continue

                type_id = self.traci.vehicle.getTypeID(veh_id)
                if not self._looks_like_hv(veh_id, type_id):
                    continue

                original_value = self.traci.vehicle.getParameter(
                    veh_id, self.KEEP_RIGHT_PARAMETER
                )
                self.traci.vehicle.setParameter(
                    veh_id, self.KEEP_RIGHT_PARAMETER, str(self.keep_right)
                )
                # Store only after a successful write so transient failures can
                # be retried during the next simulation step.
                self._original_keep_right[veh_id] = original_value
            except Exception:
                continue

        # Only the HVs changed at entry need to be checked for exit.
        for veh_id, original_value in tuple(self._original_keep_right.items()):
            try:
                if self.traci.vehicle.getRoadID(veh_id) != self.exit_edge:
                    continue
                if self.traci.vehicle.getLanePosition(veh_id) < self.exit_pos:
                    continue
                self.traci.vehicle.setParameter(
                    veh_id, self.KEEP_RIGHT_PARAMETER, original_value
                )
                self._original_keep_right.pop(veh_id, None)
            except Exception:
                continue

        # Clean exceptional removals periodically instead of scanning the whole
        # network every 0.1 s. The default interval is 10 s at a 0.1-s step.
        if (self.cleanup_interval > 0
                and self._update_count % self.cleanup_interval == 0):
            try:
                active_ids = set(self.traci.vehicle.getIDList())
            except Exception:
                return
            for veh_id in set(self._original_keep_right) - active_ids:
                self._original_keep_right.pop(veh_id, None)