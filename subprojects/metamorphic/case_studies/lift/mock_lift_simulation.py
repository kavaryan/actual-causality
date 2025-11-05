class MockLiftsSimulator:
    def __init__(self, average_max_time=1.0, simulator_startup_cost=0.1):
        self.stretch_coefficient = average_max_time
        self.startup_cost = simulator_startup_cost
        self.total_time = 0.0

    def simulate(self, num_lifts):
        # Calculate simulation time (i.e., AWT) based on number of active lifts
        sim_time = ((self.stretch_coefficient / num_lifts) if num_lifts else float('inf')) + self.startup_cost
        
        # Add to total tracked time
        self.total_time += sim_time
        
        # Return the average waiting time (inverse relationship with num_lifts)
        return sim_time
    
    def get_total_time(self):
        return self.total_time
    
    def reset_time(self):
        self.total_time = 0.0
