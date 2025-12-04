import math

class MockAVSimulator:
    def __init__(self, distance=10.0, speed=5.0, average_max_time=1.0, simulator_startup_cost=0.1):
        self.distance = distance
        self.speed = speed
        self.stretch_coefficient = average_max_time
        self.startup_cost = simulator_startup_cost
        self.total_time = 0.0

    def simulate(self, num_obstacles):
        # Calculate simulation time (i.e., TD) based on number of obstacles
        # TD = base_travel_time * stretch_coefficient / num_obstacles + startup_cost
        base_travel_time = self.distance / self.speed
        sim_time = (base_travel_time * self.stretch_coefficient / num_obstacles if num_obstacles else float('inf')) + self.startup_cost
        
        # Add to total tracked time
        self.total_time += sim_time
        
        # Return the time to destination (inverse relationship with num_obstacles)
        return sim_time
    
    def get_total_time(self):
        return self.total_time
    
    def reset_time(self):
        self.total_time = 0.0
