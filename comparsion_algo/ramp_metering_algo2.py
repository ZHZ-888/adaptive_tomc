# ramp_metering_algo2.py
class Func:
    def __init__(self, traci):
        self.traci = traci
        # --- minimal knobs with safe defaults ---
        self.K = 70.0                 # >0 for ALINEA
        self.use_occupancy = True     # True: use occupancy on a 0-1 scale; False: veh/km
        self.target_occupancy = 0.10  # only used when use_occupancy=True

        # release-rate bounds (veh/h)
        self.r_min = 0.0
        self.r_max = 720.0
        self._r_prev = 720.0

        # signal timing
        self.yellow_time = 0          # s
        self.min_green = 2          # s 1.8
        self.min_red   = 3         # s 4

        # if TLS controls multiple links, which index is the ramp movement (0-based)
        self.ramp_conn_index = 0

    # ---------- ALINEA control ----------
    def alinea_control(self, current_occupancy):
        """
        ALINEA control law: r_k = r_{k-1} + K * (target - current)
        NOTE: use K>0. This function keeps your original signature.
        """
        # choose error source
        error = self.target_occupancy - float(current_occupancy)
        r = self._r_prev + self.K * error
        r = max(self.r_min, min(self.r_max, r))
        self._r_prev = r
        return r

    # ---------- legacy on/off (kept) ----------
    def set_ramp_signal(self, ramp_signal_id, rate):
        self.traci.trafficlight.setRedYellowGreenState(
            ramp_signal_id, "G" if rate <= 0.5 else "r"
        )

    # ---------- measurement helper (kept name) ----------
    def get_current_density(self, lane_id):
        """
        If use_occupancy=True, returns lastStepOccupancy (0~1).
        Else returns density (veh/km).
        """
        if self.use_occupancy:
            try:
                return self.traci.lane.getLastStepOccupancy(lane_id) / 100.0
            except Exception:
                # very crude fallback from veh/km -> occupancy
                veh = self.traci.lane.getLastStepVehicleNumber(lane_id)
                Lkm = max(1e-6, self.traci.lane.getLength(lane_id) / 1000.0)
                dens = veh / Lkm
                return min(1.0, dens / 140.0)
        else:
            veh = self.traci.lane.getLastStepVehicleNumber(lane_id)
            Lkm = max(1e-6, self.traci.lane.getLength(lane_id) / 1000.0)
            return veh / Lkm

    def get_detector_occupancy(self, detector_id):
        """
        Return E1 detector occupancy on a 0-1 scale.
        """
        return self.traci.inductionloop.getLastStepOccupancy(detector_id) / 100.0

    # ---------- rate -> signal timing (fixed) ----------
    def set_ramp_metering_signal(self, ramp_meter_id, release_rate):
        """
        Single-lane ramp: enforce "1 vehicle per green".
        Uses ALINEA's release_rate (veh/h) to determine the average headway,
        and adjusts the red time to meet that headway.
        The green time is fixed to allow exactly one vehicle to pass,
        based on saturation flow and a start-up lost time.
        """
        Y = float(self.yellow_time)  # yellow time (s)

        # 1) Target: one vehicle per green phase
        # Fixed green time for approximately one vehicle per cycle
        g = self.min_green

        # 2) From release_rate (veh/h) -> target average headway (s/veh)
        r = max(1e-6, float(release_rate))
        headway = 3600.0 / r  # seconds per vehicle

        # 3) Use red time to achieve the target headway:
        #    headway = green + yellow + red
        R_raw = headway - g - Y
        R = max(self.min_red, R_raw)

        # 4) Keep decimal phase durations to preserve the target headway
        G = g

        # 5) Build TLS state strings:
        #    - All-red except the ramp connection index is set to G/y/r
        num_links = len(self.traci.trafficlight.getControlledLinks(ramp_meter_id)) or 1
        idx = min(max(0, int(self.ramp_conn_index)), num_links - 1)

        def all_red():
            return "r" * num_links

        def with_char(ch):
            s = ["r"] * num_links
            s[idx] = ch
            return "".join(s)

        # 6) Create the TLS program and apply it  —— libsumo-safe
        Phase = self.traci.trafficlight.Phase
        Logic = self.traci.trafficlight.Logic  # in libsumo this is TraCILogic

        phases = [
            Phase(G, with_char('G')),
            Phase(Y, with_char('y')),
            Phase(R, all_red()),
        ]

        # IMPORTANT: use positional args, not keywords
        # signature: Logic(programID, type, currentPhaseIndex, phases)
        program = Logic('alinea_1veh', 0, 0, phases)

        self.traci.trafficlight.setCompleteRedYellowGreenDefinition(ramp_meter_id, program)
        return (G, R)

