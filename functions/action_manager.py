from functions import print_control as prc
from functions import v2x_disturbance as v2x
import re
import copy

class ActionManager:
    def __init__(self, instance_dr, merge_regular, loss_rate=0, comm_rng=None):
        self.data_recorder = instance_dr
        self.merge_regular = merge_regular
        self.loss_rate = loss_rate

        self.delay_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate, rng=comm_rng)
        self.delay_buffer_leader = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate, rng=comm_rng) # leader action
        self.delay_buffer_followup = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate, rng=comm_rng) # followup leader action

        self.dic_rm_leader_map = None
        self.dic_leader_action_mgr = None
        self.ls_action_leader = None
        self.dic_m_leader_followup_action = None
        self.ls_m_leaders_followup = None

        self.last_update_payload = None
        self.last_update_payload_leader = None
        self.last_update_payload_followup = None

        self.new_leader_flag = False

    def get_action_info(self, step, new_leader_flag, interval=60):
        """
        :param step: The step for which action info is requested.
        :param interval: The interval between disturbances (default is 70).
        :return: Action information based on whether there is a disturbance or not.
        """
        self.new_leader_flag = new_leader_flag
        disturb = self.loss_rate != 0
        if disturb:
            return self._get_action_disturbed(step, interval)
        else:
            return self._get_action_clean(step, interval)

    def execute_action(self, step, dic):
        try:
            ls_veh_id = self.data_recorder.dic_vid_groups['ls_vehid']
            ls = list(dic.keys())
            self.merge_regular.apply_leader_action(step, dic)
            ls_valid = [veh_id for veh_id in ls if veh_id in ls_veh_id]
            self.merge_regular.flashing_merging(step, ls_valid)
        except:
            pass

    def _update(self, dic_leader_action, dic_m_leader_followup_action):
        '''
        so important to use deepcopy here, otherwise the reference will be passed and the original dic
        will be changed by the new dic in the next update, which will cause the action info
        always be the same as the latest one, and never be stale
        '''
        self.dic_leader_action_mgr = copy.deepcopy(dic_leader_action)
        self.dic_m_leader_followup_action = copy.deepcopy(dic_m_leader_followup_action)

    def _build_action_payload(self, step, interval=60):
        """
        :param step: the step number to check if it's a multiple of the interval
        :param interval: MPC interval, default is 70
        :return: a tuple containing dictionaries and lists representing various action-related data
                 if the step is a multiple of the interval, otherwise None
        """
        update_payload = None
        if step % interval == 0 or self.new_leader_flag:
            dic_rm_leader_map, dic_rm_leader_actor = self.merge_regular.find_rm_leader_map(step)
            dic_leader_action, ls_action_leader = self.merge_regular.get_leader_action(step)
            dic_m_leader_followup_action, ls_m_leaders_followup = self.merge_regular.get_m_leader_followup_action(step)
            self._print_decision(dic_rm_leader_actor, ls_m_leaders_followup)
            update_payload = (dic_leader_action, dic_m_leader_followup_action)
        return update_payload

    def _get_action_disturbed(self, step, interval=60):
        """
        consider v2x disturbance
        :param step: time step
               interval: MPC interval, default is 60
               self.delay_buffer:
        :return: a tuple containing the current state of action info, including dictionaries and lists
        """
        # Generate action commands at each interval
        update_payload = self._build_action_payload(step, interval)

        if update_payload:
            # Step 3: Check if it's a new (non-empty) command
            # True => if payload is either a repeat of the last one or is entirely empty
            is_redundant = (update_payload == self.last_update_payload or
                            all(not x for x in update_payload))
            if not is_redundant:  # update_payload is new and non-empty; is_redundant is False
                self.last_update_payload = copy.deepcopy(update_payload)  # store for next comparison
                # delay_buffer => v2x.UpdateDelayBuffer(loss_rate)
                self.delay_buffer.push(step, update_payload)  # <<< drop or delay here
            else:
                prc.print_message("empty or redundant command")
        delayed_payload = self.delay_buffer.maybe_release(step)  # Check if delayed payload is ready to execute

        delayed_payload and self._update(*delayed_payload)  # * => unpacking operator
        # Return the current (possibly stale) action info
        return (self.dic_leader_action_mgr, self.dic_m_leader_followup_action)

    def _get_action_disturbed2(self, step, interval=60):
        """
        consider v2x disturbance
        :param step: time step
               interval: MPC interval, default is 60
               self.delay_buffer:
        :return: a tuple containing the current state of action info, including dictionaries and lists
        """
        # Generate action commands at each interval
        update_payload = self._build_action_payload(step, interval)
        leader_action = update_payload[0]
        followup_m_leader_action = update_payload[1]

        # 1. disturbance on dic_leader_action
        delay_leader_action = self._add_disturb(step, leader_action,
                                                self.last_update_payload_leader,
                                                self.delay_buffer_leader)
        # 2. disturbance on dic_leader_action
        delay_followup_m_leader_action = self._add_disturb(step, followup_m_leader_action,
                                                           self.last_update_payload_followup,
                                                           self.delay_buffer_followup)

        delayed_payload = (delay_leader_action, delay_followup_m_leader_action)
        delayed_payload and self._update(*delayed_payload)  # * => unpacking operator
        # Return the current (possibly stale) action info
        return (self.dic_leader_action_mgr, self.dic_m_leader_followup_action)

    def _add_disturb(self, step, action, last_action, buffer):
        '''
        action (leader_action; followup_m_leader_action)
        last_action (self.last_update_leader_action; self.last_update_followup_m_leader_action)
        buffer (self.delay_buffer_leader; self.delay_buffer_followup)
        '''
        action_filtered = action.copy()
        for k in action_filtered:
            action_filtered[k] = action_filtered[k][:-1]
        if action_filtered:
            # True => if payload is either a repeat of the last one or is entirely empty
            is_redundant = (action == last_action or all(not x for x in action))
            if not is_redundant:
                last_action = action
                buffer.push(step, action)
            else:
                prc.print_message("empty or redundant leader command")
        delay_action = buffer.maybe_release(step)
        return delay_action

    def _get_action_clean(self, step, interval=60):
        """
        Assume perfect V2X
        :param step:
        :param interval:
        :return:
        """
        update_payload = self._build_action_payload(step, interval)
        if update_payload:
            self._update(*update_payload)
        return (self.dic_leader_action_mgr, self.dic_m_leader_followup_action)

    def _print_decision(self, dic_rm_leader_actor, ls_m_leaders_followup):
        '''
        print the action decision detail
        :param dic_rm_leader_actor: dic of encountered r_leader and m_leader, and choose action avh
        :param ls_m_leaders_followup: action list of
        :return:
        '''
        # dic_vid_groups = self.data_recorder.record_vehinfo()
        dic_vid_groups = self.data_recorder.dic_vid_groups
        ls_mr_leader_up = dic_vid_groups["ls_mr_leader_up"] # all avh before merging
        ls_m_leader_up_asc = dic_vid_groups["ls_m_leader_up_asc"] # all m_leader before merging
        action_dic = {key: value for key, value in dic_rm_leader_actor.items() if value in ls_mr_leader_up}
        # prc.print_message(action_dic)
        # filter out duplicate r_leader, only keep the last pair
        result = {}
        for key, value in action_dic.items():
            result[key[0]] = (key, value)
        action_dic_filter = {key: value for key, value in result.values()}
        prc.print_message(f'\n{action_dic_filter}')
        for key, value in action_dic_filter.items():
            r_leader = key[0]
            m_leader = key[1]

            if r_leader == 'ravh7310':
                pass

            action_avh = value
            prc.print_message(f'STATE: {r_leader} encounter {m_leader}')
            if 'm' in action_avh:
                # type 1: only m_leader need to take action
                prc.print_message(f'DECISION: *action type 1*\n{action_avh} take action, make sure head {m_leader} > tail {r_leader}')
            else:
                # type2: r_leader take action, m_leader no action, but m_leader's next m_leader may need to take action
                prc.print_message(f'DECISION: *action type 2*\n{action_avh} take action, make sure head {r_leader} > tail {m_leader}')
                # the next m_leader of this m_leader
                m_leader_time = int(re.search(r'\d+', m_leader).group()) # int(m_leader[4:])

                m_leader_next = min(
                    (
                        (v, int(re.search(r'\d+', v).group()))
                        for v in ls_m_leader_up_asc
                        if int(re.search(r'\d+', v).group()) > m_leader_time
                    ),
                    key=lambda x: x[1],
                    default=(None, None)
                )[0] # Handle the case where no later vehicle exists

                if m_leader_next in ls_m_leaders_followup:
                    prc.print_message(f'{m_leader_next} (next avh of {m_leader}) need to take action, make sure head {m_leader_next} > tail {r_leader}')
        prc.print_message() # print one blank



