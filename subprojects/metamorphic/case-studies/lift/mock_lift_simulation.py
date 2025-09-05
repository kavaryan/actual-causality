import time

class MockLiftsSimulator:
    def __init__(self, max_time):
        self.max_time = max_time

    def simulate(self, num_lifts):
        time.sleep(self.max_time / num_lifts)
        return self.max_time / num_lifts