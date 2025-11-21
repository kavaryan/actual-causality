import random
from math import inf

class MockLiftsSimulator:
    def __init__(
        self,
        average_max_time=1.0,
        simulator_startup_cost=0.1,
        # Deterministic lift speed (e.g. relative speed, > 0)
        speed=1.0,
        # Real-world fluctuation parameters: N(mu, sigma) each
        traffic_intensity_params=(0.0, 0.05),
        door_delay_params=(0.0, 0.05),
        loading_time_params=(0.0, 0.05),
        # Ensures fluctuations are ~1 order of magnitude smaller than base time
        fluctuation_scale=0.1,
    ):
        # Interpret average_max_time as the travel time at speed=1 and 1 lift
        self.stretch_coefficient = average_max_time
        self.startup_cost = simulator_startup_cost

        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = speed

        self.traffic_intensity_params = traffic_intensity_params
        self.door_delay_params = door_delay_params
        self.loading_time_params = loading_time_params

        self.fluctuation_scale = fluctuation_scale
        self.total_time = 0.0

    def _sample_time_fluctuation(self):
        """Sample combined dimensionless fluctuation from 3 real-world factors."""
        t_mu, t_sigma = self.traffic_intensity_params
        d_mu, d_sigma = self.door_delay_params
        l_mu, l_sigma = self.loading_time_params

        delta_traffic = random.gauss(t_mu, t_sigma)
        delta_door = random.gauss(d_mu, d_sigma)
        delta_loading = random.gauss(l_mu, l_sigma)

        # Sum gives a small dimensionless offset around 0
        return delta_traffic + delta_door + delta_loading

    def simulate(self, num_lifts):
        # Deterministic base travel time:
        #   - inversely proportional to number of lifts
        #   - inversely proportional to speed
        if num_lifts:
            base_travel_time = (self.stretch_coefficient / num_lifts) / self.speed
        else:
            base_travel_time = inf

        # Add startup overhead
        base_time = base_travel_time + self.startup_cost

        # Small multiplicative perturbation from traffic/door/loading
        fluct = self._sample_time_fluctuation()
        sim_time = base_time * (1.0 + self.fluctuation_scale * fluct)

        # Prevent negative or zero times in pathological cases
        sim_time = max(sim_time, 1e-9)

        self.total_time += sim_time
        return sim_time

    def get_total_time(self):
        return self.total_time

    def reset_time(self):
        self.total_time = 0.0
