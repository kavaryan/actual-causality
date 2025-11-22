import random
from math import inf

class MockLiftsSimulator:
    def __init__(
        self,
        average_max_time=1.0,
        simulator_startup_cost=0.1,
        speed=1.0,
        call_density=0.0,
        traffic_intensity_params=(0.0, 0.05),
        door_delay_params=(0.0, 0.05),
        loading_time_params=(0.0, 0.05),
        fluctuation_scale=0.1,
        call_density_scale=1.0,
    ):
        # Interpret average_max_time as the travel time at speed=1, 1 lift, no congestion
        self.stretch_coefficient = average_max_time
        self.startup_cost = simulator_startup_cost

        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = speed

        # Call density: higher => more congestion => more time
        self.call_density = max(call_density, 0.0)
        self.call_density_scale = call_density_scale

        self.traffic_intensity_params = traffic_intensity_params
        self.door_delay_params = door_delay_params
        self.loading_time_params = loading_time_params

        self.fluctuation_scale = fluctuation_scale
        self.total_time = 0.0

    @staticmethod
    def _truncated_normal(mu, sigma, low, high, max_tries=100):
        """
        Sample from N(mu, sigma) truncated to [low, high].
        If sigma <= 0, just return mu.
        """
        if sigma <= 0:
            return mu

        for _ in range(max_tries):
            x = random.gauss(mu, sigma)
            if low <= x <= high:
                return x

        # Fallback: clamp a last sample if we somehow failed max_tries times
        x = random.gauss(mu, sigma)
        return min(max(x, low), high)

    def _sample_truncated_param(self, mu, sigma):
        """
        Convenience: sample from N(mu, sigma) truncated to [mu - sigma, mu + sigma].
        """
        low = mu - sigma
        high = mu + sigma
        return self._truncated_normal(mu, sigma, low, high)

    def _sample_time_fluctuation(self):
        """Sample combined dimensionless fluctuation from 3 real-world factors."""
        t_mu, t_sigma = self.traffic_intensity_params
        d_mu, d_sigma = self.door_delay_params
        l_mu, l_sigma = self.loading_time_params

        delta_traffic = self._sample_truncated_param(t_mu, t_sigma)
        delta_door = self._sample_truncated_param(d_mu, d_sigma)
        delta_loading = self._sample_truncated_param(l_mu, l_sigma)

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

        # Apply congestion due to call density
        # call_density >= 0, call_density_scale controls how strong the effect is.
        congestion_multiplier = 1.0 + self.call_density_scale * self.call_density
        base_travel_time *= congestion_multiplier

        # Add startup overhead
        base_time = base_travel_time + self.startup_cost

        # Small multiplicative perturbation from traffic/door/loading (truncated normal)
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
