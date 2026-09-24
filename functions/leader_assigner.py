# leader_assigner.py
class PlatoonLeaderAssigner:
    """
    Assign leader/follower roles for an ordered vehicle stream.

    Role tags:
        0: HV follower
        1: AV leader
        2: AV follower
    """

    def __init__(self, max_team_size):
        self.max_team_size = max_team_size

    def assign(
        self,
        ordered_ids,
        tags,
        old_tags,
        role_changes,
        enable_proactive_split,
        is_av,
        get_promotion_type,
    ):
        """
        Assign vehicle roles while preserving previously triggered role changes.

        Parameters
        ----------
        ordered_ids
            Vehicle IDs ordered from downstream to upstream.
        tags
            Mutable mapping from vehicle ID to its current role tag.
        old_tags
            Role tags from the previous assignment cycle.
        role_changes
            Persistent mapping of promoted AVs to promotion types.
        enable_proactive_split
            Whether an AV may be promoted before the platoon exceeds its
            maximum size.
        is_av
            Function that returns True when a vehicle is an AV.
        get_promotion_type
            Function that returns the promotion type for a promoted AV.

        Returns
        -------
        tuple
            Updated tags, updated role changes, and IDs of demoted leaders.
        """
        demoted_leaders = set()

        current_leader = None
        current_team_size = 0

        for index, vehicle_id in enumerate(ordered_ids):
            if not is_av(vehicle_id):
                # HVs can only be followers.
                tags[vehicle_id] = 0

                if current_leader is not None:
                    current_team_size += 1

                continue

            # Recover the closest leader ahead of the current AV.
            all_tagged_ids = list(tags)
            vehicle_index = all_tagged_ids.index(vehicle_id)

            for previous_index in range(vehicle_index - 1, -1, -1):
                previous_id = all_tagged_ids[previous_index]

                if tags.get(previous_id) == 1:
                    current_leader = previous_id
                    current_team_size = (
                        vehicle_index - previous_index + 1
                    )
                    break

            # Check whether consecutive HVs behind this AV would cause the
            # current platoon to exceed the maximum team size.
            too_many_hv_behind = False

            if (
                current_leader is not None
                and current_team_size <= self.max_team_size
            ):
                remaining_slots = (
                    self.max_team_size - current_team_size
                )
                hv_count = 0

                for following_id in ordered_ids[index + 1:]:
                    if is_av(following_id):
                        break

                    hv_count += 1

                too_many_hv_behind = hv_count > remaining_slots

            should_be_leader = (
                current_leader is None
                or current_team_size > self.max_team_size
                or vehicle_id in role_changes
            )

            if should_be_leader:
                tags[vehicle_id] = 1
                current_leader = vehicle_id
                current_team_size = 1

            elif enable_proactive_split and too_many_hv_behind:
                # Promote this AV before the current platoon becomes oversized.
                tags[vehicle_id] = 1
                role_changes[vehicle_id] = get_promotion_type(vehicle_id)
                current_leader = vehicle_id
                current_team_size = 1

            elif (
                old_tags.get(vehicle_id) == 1
                and current_leader in role_changes
            ):
                # Preserve the previous leader after an upstream role change.
                tags[vehicle_id] = 1
                current_leader = vehicle_id
                current_team_size = 1

            else:
                if old_tags.get(vehicle_id) == 1:
                    demoted_leaders.add(vehicle_id)

                tags[vehicle_id] = 2
                current_team_size += 1

        return tags, role_changes, demoted_leaders