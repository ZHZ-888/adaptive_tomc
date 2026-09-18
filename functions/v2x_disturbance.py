'''
v2x_disturbance.py
This module implements a lightweight communication disturbance simulator
for V2X-based control systems. It simulates stochastic packet loss and
randomised communication latency, allowing evaluation of algorithm
robustness under imperfect communication conditions.
'''

from functions import print_control as prc
from collections import deque
import random
import math

class UpdateDelayBuffer:
    def __init__(self, loss_rate=0.2, sim_step=0.1, rng=None):
        self.loss_rate = loss_rate
        self.sim_step = sim_step
        self.rng = rng or random.Random()
        self.buffer = deque()  # store (payload, release_step)

    @staticmethod
    def _payloads_equal(left, right):
        """Compare nested payloads without failing on array-like values."""
        if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
            return (len(left) == len(right) and
                    all(UpdateDelayBuffer._payloads_equal(a, b)
                        for a, b in zip(left, right)))
        if isinstance(left, dict) and isinstance(right, dict):
            return (left.keys() == right.keys() and
                    all(UpdateDelayBuffer._payloads_equal(left[key], right[key])
                        for key in left))
        try:
            result = left == right
            if isinstance(result, bool):
                return result
            if hasattr(result, "all"):
                return bool(result.all())
        except (TypeError, ValueError):
            pass
        return False

    def push(self, current_step, payload): # only in jam_control?
        """Decide delay (release_step) and store payload if not dropped
        latency (delay): 0.05~0.2 s
        """
        if payload:
            # min_steps = int(0.05 / self.sim_step)
            # max_steps = int(0.2 / self.sim_step)
            # delay = random.randint(min_steps, max_steps)
            delay_seconds = self.rng.uniform(0.05, 0.20)
            delay = math.ceil(delay_seconds / self.sim_step)
            release_step = current_step + delay
            # this is the only difference between push_ori and push
            if any(self._payloads_equal(p, payload) for p, _ in self.buffer):
                prc.print_message(f"[SKIP] Duplicate entry: ({payload}, {release_step})")
                return

            prc.print_message(f"\n[SEND] Step {current_step}: Generated command → {payload}")
            if self.rng.random() < self.loss_rate:
                # print(f"[DROP] Payload lost: {payload}")
                prc.print_message("[DROP] Payload lost")
                return

            prc.print_message(f"[DELAY] {delay} for command → {payload}")
            self.buffer.append((payload, release_step))

    def push_timing(self, current_step, payload): # only in jam_control
        """Decide delay (release_step) and store payload if not dropped
        latency (delay): 0.05~0.2 s
        """
        if payload:
            # min_steps = int(0.05 / self.sim_step)
            # max_steps = int(0.2 / self.sim_step)
            # delay = random.randint(min_steps, max_steps)
            delay_seconds = self.rng.uniform(0.05, 0.20)
            delay = math.ceil(delay_seconds / self.sim_step)
            release_step = current_step + delay

            prc.print_message(f"\n[SEND] Step {current_step}: Generated command → {payload}")
            if self.rng.random() < self.loss_rate:
                # print(f"[DROP] Payload lost: {payload}")
                prc.print_message("[DROP] Payload lost")
                return

            prc.print_message(f"[DELAY] {delay} for command → {payload}")
            self.buffer.append((payload, release_step))

    def maybe_release(self, current_step):
        """Check if any payload is ready to be executed
           release commands at the release_step
           self.buffer[0][1] => release_step
        """
        if current_step == 601:
            pass
        if self.buffer and self.buffer[0][1] <= current_step:
            payload = self.buffer.popleft()[0]
            prc.print_message(f"[EXECUTE] Step {current_step}: AV executes → {payload}")
            return payload
        return None


if __name__ == '__main__':
    prc.PRINT_ENABLED = True
    delay_buffer = UpdateDelayBuffer(loss_rate=0.2)

    dic_command = {"command1": 2, "command2": 10, "command3": 13}
    dic_command = {"command1": 2, "command2": 10, "command3": 13, "command4": 21,
                   "command5": 23, "command6": 26, "command7": 27, "command8": 28,
                   "command9": 34, "command10": 45, "command11": 56, "command12": 57}
    dic_steps = {v: k for k, v in dic_command.items()}
    for step in range(100):
        cmd = dic_steps.get(step)
        delay_buffer.push(step, cmd) # step 1: push command into buffer
        pay_load = delay_buffer.maybe_release(step) # step 2: check if any command is ready to be executed
