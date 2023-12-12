import time
from threading import Thread


class WaitMatlab(Thread):
    def __init__(self, planning_time_ref):
        Thread.__init__(self)
        self.planning_time_ref = planning_time_ref
        self.planning_time = 0
        self.running = True
        self.timeout = False

    def stop(self):
        self.running = False

    def run(self):
        while self.running and self.planning_time < self.planning_time_ref:
            time.sleep(1)
            self.planning_time += 1
        if self.planning_time >= self.planning_time_ref:
            self.timeout = True
