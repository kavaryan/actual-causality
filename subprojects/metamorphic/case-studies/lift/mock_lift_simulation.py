class MockLiftsSimulator:
    def __init__(self, stretch_coefficient=1.0, startup_cost=0.1):
        self.stretch_coefficient = stretch_coefficient
        self.startup_cost = startup_cost
        self.total_time = 0.0

    def simulate(self, num_lifts):
        # Calculate simulation time based on number of lifts
        sim_time = self.stretch_coefficient / num_lifts + self.startup_cost
        
        # Add to total tracked time
        self.total_time += sim_time
        
        # Return the average waiting time (inverse relationship with num_lifts)
        return sim_time
    
    def get_total_time(self):
        return self.total_time
    
    def reset_time(self):
        self.total_time = 0.0
