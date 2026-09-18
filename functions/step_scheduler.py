# step_scheduler.py

class StepScheduler:
    """
    Lightweight step-based scheduler.
    - Maintains global step
    - Triggers registered callbacks at given intervals
    """

    def __init__(self):
        self.global_step = 0
        self.tasks = []

    def register(self, name, interval, callback):
        """
        Args:
            name (str): task name (for debug/log)
            interval (int): execute every N steps
            callback (callable): callback(step)
        """
        assert interval > 0
        self.tasks.append({
            "name": name,
            "interval": interval,
            "callback": callback
        })

    def tick(self):
        """Advance one global step and execute scheduled tasks"""
        self.global_step += 1
        step = self.global_step

        for task in self.tasks:
            if step % task["interval"] == 0:
                task["callback"](step)


def task_fast(step):

    print(f"  task_fast running at step {step}")

def task_medium(step):
    print(f"  task_medium running at step {step}")

def task_slow(step):
    print(f"  task_slow running at step {step}")


if __name__ == "__main__":
    scheduler = StepScheduler()

    # register some example tasks
    scheduler.register("fast_task", interval=1, callback=task_fast)
    scheduler.register("medium_task", interval=3, callback=task_medium)
    scheduler.register("slow_task", interval=5, callback=task_slow)

    # loop
    for _ in range(15):
        scheduler.tick()
