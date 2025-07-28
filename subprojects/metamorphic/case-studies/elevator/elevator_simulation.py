import time
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import copy
import random
import numpy as np

class ElevatorState(Enum):
    IDLE = 'idle'
    MOVING = 'moving'
    WAITING = 'waiting'
    OPEN_DOOR = 'open_door'
    CLOSE_DOOR = 'close_door'

@dataclass
class ElevatorConfig:
    min_floor_num: int
    max_floor_num: int
    idle_floor_num: int
    max_load: int
    speed: float  # m/s
    stop_duration: float  # seconds
    toggle_door_duration: float  # seconds

@dataclass
class FloorsConfig:
    min_floor: int
    max_floor: int
    heights: List[float]  # height of each floor

@dataclass
class RouteData:
    total_time: float
    enter_idx: int
    leave_idx: int
    from_floor_num: int
    to_floor_num: int

@dataclass
class NextFloorData:
    floor_num: int
    details: Optional[Dict] = None

@dataclass
class Call:
    from_floor: int
    to_floor: int
    call_time: float
    passenger_id: int
    
@dataclass
class CompletedCall:
    call: Call
    delivery_time: float
    waiting_time: float

class TimeService:
    MIN_TIME_MULTIPLIER = 0.25
    MAX_TIME_MULTIPLIER = 10.0
    
    def __init__(self, time_ratio: float = 2.5):
        self._time_ratio = time_ratio
        self._simulation_time = 0.0
    
    @property
    def time_ratio(self) -> float:
        return self._time_ratio
    
    @time_ratio.setter
    def time_ratio(self, ratio: float):
        if self.MIN_TIME_MULTIPLIER <= ratio <= self.MAX_TIME_MULTIPLIER:
            self._time_ratio = ratio
    
    def convert_duration(self, real_duration: float) -> float:
        return real_duration / self._time_ratio
    
    def get_elapsed_time(self, start_time: float) -> float:
        return (self._simulation_time - start_time) * self._time_ratio
    
    def get_time(self) -> float:
        return self._simulation_time
    
    def set_time(self, simulation_time: float):
        self._simulation_time = simulation_time

class Elevator:
    MAX_BUTTONS_PER_SCREEN = 16
    
    def __init__(self, config: ElevatorConfig, elevator_service, time_service: TimeService, idx: int):
        self.config = config
        self.elevator_service = elevator_service
        self.time_service = time_service
        self.idx = idx
        
        self._state = ElevatorState.IDLE
        self._next_floors: List[NextFloorData] = [NextFloorData(float('nan'))]
        self._current_floor_num = config.idle_floor_num
        self.bottom = self._calc_distance_from_bottom(self._current_floor_num)
        self.stopped_start_time = 0
        self.door_toggle_start_time = 0
        
    @property
    def state(self) -> ElevatorState:
        return self._state
    
    @property
    def next_floor(self) -> Optional[NextFloorData]:
        return self._next_floors[1] if len(self._next_floors) > 1 else None
    
    @property
    def current_floor_num(self) -> int:
        return self._current_floor_num
    
    @property
    def total_stop_duration(self) -> float:
        return 2 * self.config.toggle_door_duration + self.config.stop_duration
    
    def update(self, delta_time: float, current_time: float):
        if self._state == ElevatorState.MOVING:
            sign = -1 if self.next_floor.floor_num < self.current_floor_num else 1
            self._move(sign * self._calc_move_distance(delta_time))
            self._update_current_floor_num()
            
        elif self._state == ElevatorState.WAITING:
            elapsed = current_time - self.stopped_start_time
            if elapsed >= self.config.stop_duration:
                if not self.next_floor:
                    self._state = ElevatorState.IDLE
                else:
                    self._state = ElevatorState.CLOSE_DOOR
                    self.door_toggle_start_time = current_time
                    
        elif self._state in [ElevatorState.OPEN_DOOR, ElevatorState.CLOSE_DOOR]:
            elapsed = current_time - self.door_toggle_start_time
            if elapsed >= self.config.toggle_door_duration:
                self._toggle_door(current_time)
                
        elif self._state == ElevatorState.IDLE:
            if self.next_floor:
                self._state = ElevatorState.CLOSE_DOOR
                self.door_toggle_start_time = current_time
            elif self.current_floor_num != self.config.idle_floor_num:
                self._next_floors.append(NextFloorData(self.config.idle_floor_num))
    
    def add_route(self, route: RouteData):
        enter_idx, leave_idx = route.enter_idx, route.leave_idx
        from_floor_num, to_floor_num = route.from_floor_num, route.to_floor_num
        
        # Remove idle floor if present
        if (self.next_floor and not self.next_floor.details and 
            self.next_floor.floor_num == self.config.idle_floor_num):
            self._next_floors.pop(1)
        
        # Add floors at the end if needed
        if enter_idx == len(self._next_floors):
            self._next_floors.append(NextFloorData(
                from_floor_num,
                {'no_leaving': 0, 'no_entering': 1, 'no_inside': 0}
            ))
            self._next_floors.append(NextFloorData(
                to_floor_num,
                {'no_leaving': 1, 'no_entering': 0, 'no_inside': 0}
            ))
            return
        
        # Handle entering
        if (enter_idx == 0 or self._next_floors[enter_idx].floor_num != from_floor_num):
            if (self.current_floor_num == from_floor_num and 
                self._state != ElevatorState.MOVING):
                self._handle_immediate_pickup()
            else:
                self._next_floors.insert(enter_idx + 1, NextFloorData(
                    from_floor_num,
                    {'no_leaving': 0, 'no_entering': 1, 'no_inside': 0}
                ))
                leave_idx += 1
        else:
            if not self._next_floors[enter_idx].details:
                self._next_floors[enter_idx].details = {'no_leaving': 0, 'no_entering': 0, 'no_inside': 0}
            self._next_floors[enter_idx].details['no_entering'] += 1
        
        # Handle leaving
        if leave_idx >= len(self._next_floors) or self._next_floors[leave_idx].floor_num != to_floor_num:
            self._next_floors.insert(leave_idx + 1, NextFloorData(
                to_floor_num,
                {'no_leaving': 1, 'no_entering': 0, 'no_inside': 0}
            ))
        else:
            if not self._next_floors[leave_idx].details:
                self._next_floors[leave_idx].details = {'no_leaving': 0, 'no_entering': 0, 'no_inside': 0}
            self._next_floors[leave_idx].details['no_leaving'] += 1
        
        # Update inside counts
        for i in range(enter_idx + 1, leave_idx + 1):
            if i < len(self._next_floors) and self._next_floors[i].details:
                self._next_floors[i].details['no_inside'] += 1
    
    def find_best_route(self, from_floor_num: int, to_floor_num: int) -> RouteData:
        possible_stops = self._find_possible_stops_indexes(from_floor_num, to_floor_num)
        total_time, enter_idx, leave_idx = self._find_lowest_time_route(
            from_floor_num, to_floor_num, possible_stops
        )
        
        return RouteData(
            total_time=total_time,
            enter_idx=enter_idx,
            leave_idx=leave_idx,
            from_floor_num=from_floor_num,
            to_floor_num=to_floor_num
        )
    
    def _handle_immediate_pickup(self):
        if self._state == ElevatorState.CLOSE_DOOR:
            self._state = ElevatorState.OPEN_DOOR
            self.door_toggle_start_time = self.time_service.get_time()
        elif self._state in [ElevatorState.IDLE, ElevatorState.WAITING]:
            self._state = ElevatorState.WAITING
            self.stopped_start_time = self.time_service.get_time()
    
    def _find_possible_stops_indexes(self, from_floor: int, to_floor: int) -> List[Tuple[int, int]]:
        res = []
        self._next_floors[0].floor_num = self.current_floor_num
        
        if from_floor < to_floor:  # Going up
            self._next_floors.append(NextFloorData(float('inf')))
            for s in range(len(self._next_floors) - 1):
                if (self._next_floors[s].floor_num <= from_floor < 
                    self._next_floors[s + 1].floor_num):
                    t = s
                    while (t + 1 < len(self._next_floors) and 
                           self._next_floors[t].floor_num < self._next_floors[t + 1].floor_num):
                        if (self._next_floors[t].floor_num <= to_floor < 
                            self._next_floors[t + 1].floor_num):
                            res.append([s, t])
                            break
                        t += 1
        elif from_floor > to_floor:  # Going down
            self._next_floors.append(NextFloorData(float('-inf')))
            for s in range(len(self._next_floors) - 1):
                if (self._next_floors[s].floor_num >= from_floor > 
                    self._next_floors[s + 1].floor_num):
                    t = s
                    while (t + 1 < len(self._next_floors) and 
                           self._next_floors[t].floor_num > self._next_floors[t + 1].floor_num):
                        if (self._next_floors[t].floor_num >= to_floor > 
                            self._next_floors[t + 1].floor_num):
                            res.append([s, t])
                            break
                        t += 1
        
        # Always allow waiting for elevator to complete all routes
        last_idx = len(self._next_floors) - 2
        if not res or res[-1] != [last_idx, last_idx]:
            res.append([last_idx, last_idx])
        
        self._next_floors[0].floor_num = float('nan')
        self._next_floors.pop()
        
        return res
    
    def _find_lowest_time_route(self, from_floor: int, to_floor: int, 
                               possible_stops: List[Tuple[int, int]]) -> Tuple[float, int, int]:
        lowest_time = float('inf')
        best_enter_idx = len(self._next_floors)
        best_leave_idx = len(self._next_floors)
        
        for enter_idx, leave_idx in possible_stops:
            cost = self._calc_total_time(from_floor, to_floor, enter_idx, leave_idx)
            if cost < lowest_time:
                lowest_time = cost
                best_enter_idx = enter_idx
                best_leave_idx = leave_idx
        
        return lowest_time, best_enter_idx, best_leave_idx
    
    def _calc_total_time(self, from_floor: int, to_floor: int, 
                        enter_idx: int, leave_idx: int) -> float:
        if not self._has_free_space(from_floor, to_floor, enter_idx, leave_idx):
            return float('inf')
        
        # Calculate travel time
        total_time = (self.total_stop_duration + 
                     self._calc_distance_between_floors(from_floor, to_floor) / self.config.speed)
        
        # Add time to reach pickup floor
        if not self.next_floor or enter_idx == 1:
            total_time += self._calc_distance_to_floor(from_floor) / self.config.speed
        else:
            total_time += self._calc_distance_to_floor(self.next_floor.floor_num) / self.config.speed
        
        # Add intermediate travel times
        for i in range(1, min(enter_idx, len(self._next_floors) - 1)):
            distance = self._calc_distance_between_floors(
                self._next_floors[i].floor_num,
                self._next_floors[i + 1].floor_num
            )
            total_time += distance / self.config.speed
        
        # Add waiting time for affected passengers
        if self.next_floor:
            no_passengers = 0
            if self._next_floors[enter_idx].floor_num != from_floor:
                if self._next_floors[enter_idx].details:
                    details = self._next_floors[enter_idx].details
                    no_passengers = details['no_entering'] + details['no_inside']
                
                for i in range(enter_idx + 1, len(self._next_floors)):
                    if self._next_floors[i].details:
                        no_passengers += self._next_floors[i].details['no_entering']
            
            if (leave_idx + 1 < len(self._next_floors) and 
                self._next_floors[leave_idx].floor_num != to_floor):
                details = self._next_floors[leave_idx + 1].details
                no_passengers = details['no_inside']
                for i in range(leave_idx + 1, len(self._next_floors)):
                    if self._next_floors[i].details:
                        no_passengers += self._next_floors[i].details['no_entering']
            
            total_time += no_passengers * self.total_stop_duration
        
        return total_time
    
    def _has_free_space(self, from_floor: int, to_floor: int, 
                       enter_idx: int, leave_idx: int) -> bool:
        if not self.next_floor or not self.next_floor.details:
            return True
        
        # Check space at pickup floor
        if self._next_floors[enter_idx].floor_num == from_floor:
            details = self._next_floors[enter_idx].details
            if details['no_entering'] + details['no_inside'] >= self.config.max_load:
                return False
        
        # Check space at dropoff floor
        if self._next_floors[leave_idx].floor_num == to_floor:
            details = self._next_floors[leave_idx].details
            if details['no_leaving'] + details['no_inside'] >= self.config.max_load:
                return False
        
        # Check intermediate floors
        for i in range(enter_idx + 1, leave_idx):
            details = self._next_floors[i].details
            if details['no_entering'] + details['no_inside'] >= self.config.max_load:
                return False
        
        return True
    
    def _calc_move_distance(self, time_delta: float) -> float:
        return self.config.speed * time_delta
    
    def _calc_distance_from_bottom(self, floor_num: int) -> float:
        return (self.elevator_service.get_floor_height(floor_num) - 
                self.elevator_service.get_floor_height(self.config.min_floor_num))
    
    def _calc_distance_to_floor(self, floor_num: int) -> float:
        return abs(self._calc_distance_from_bottom(floor_num) - self.bottom)
    
    def _calc_distance_between_floors(self, from_floor: int, to_floor: int) -> float:
        return abs(self._calc_distance_from_bottom(from_floor) - 
                  self._calc_distance_from_bottom(to_floor))
    
    def _move(self, distance: float):
        if not self.next_floor:
            self._state = ElevatorState.OPEN_DOOR
            self.door_toggle_start_time = self.time_service.get_time()
            return
        
        self.bottom += distance
        next_bottom = self._calc_distance_from_bottom(self.next_floor.floor_num)
        
        if abs(self.bottom - next_bottom) < abs(1.25 * distance):
            self._state = ElevatorState.OPEN_DOOR
            self.door_toggle_start_time = self.time_service.get_time()
            self.bottom = next_bottom
            self._current_floor_num = self.next_floor.floor_num
            
            # Check for completed calls - handle passenger delivery
            if self.next_floor.details and self.next_floor.details.get('no_leaving', 0) > 0:
                self.elevator_service.handle_passenger_delivery(
                    self.next_floor.floor_num, 
                    self.next_floor.details['no_leaving'],
                    self.time_service.get_time()
                )
            
            self._next_floors.pop(1)
    
    def _update_current_floor_num(self):
        if (self.current_floor_num < self.config.max_floor_num and 
            self._calc_distance_to_floor(self.current_floor_num + 1) < 1e-6):
            self._current_floor_num += 1
        elif (self.current_floor_num > self.config.min_floor_num and 
              self._calc_distance_to_floor(self.current_floor_num - 1) < 1e-6):
            self._current_floor_num -= 1
    
    def _toggle_door(self, current_time: float):
        if self._state == ElevatorState.OPEN_DOOR:
            self._state = ElevatorState.WAITING
            self.stopped_start_time = current_time
        else:
            self._state = ElevatorState.MOVING

class ElevatorService:
    def __init__(self, floors_config: FloorsConfig, elevator_configs: List[ElevatorConfig], 
                 time_service: TimeService, verbose: bool = False):
        self.floors = floors_config
        self.elevator_configs = elevator_configs
        self.time_service = time_service
        self.elevators: List[Elevator] = []
        self.floor_heights: Dict[int, float] = {}
        self.pending_calls: List[Call] = []
        self.active_calls: Dict[int, Call] = {}  # passenger_id -> Call
        self.completed_calls: List[CompletedCall] = []
        self.verbose = verbose
        
        self._calc_floor_height_sums()
        self._create_elevators()
    
    def _calc_floor_height_sums(self):
        self.floor_heights[self.floors.min_floor] = 0
        for i in range(1, len(self.floors.heights)):
            prev_height = self.floor_heights[self.floors.min_floor + i - 1]
            curr_height = self.floors.heights[i - 1]
            self.floor_heights[self.floors.min_floor + i] = prev_height + curr_height
    
    def _create_elevators(self):
        for i, config in enumerate(self.elevator_configs):
            elevator = Elevator(config, self, self.time_service, i)
            self.elevators.append(elevator)
    
    def get_floor_height(self, floor_num: int) -> float:
        if floor_num not in self.floor_heights:
            raise ValueError(f"Wrong floor number {floor_num}")
        return self.floor_heights[floor_num]
    
    def add_call(self, call: Call):
        self.pending_calls.append(call)
    
    def process_pending_calls(self, current_time: float):
        calls_to_process = [call for call in self.pending_calls if call.call_time <= current_time]
        
        for call in calls_to_process:
            self.pending_calls.remove(call)
            self._assign_call_to_elevator(call)
            self.active_calls[call.passenger_id] = call
    
    def _assign_call_to_elevator(self, call: Call):
        # Find available elevators
        available_elevators = [
            elevator for elevator in self.elevators
            if (elevator.config.min_floor_num <= call.from_floor <= elevator.config.max_floor_num and
                elevator.config.min_floor_num <= call.to_floor <= elevator.config.max_floor_num)
        ]
        
        if not available_elevators:
            print(f"No available elevator for call from {call.from_floor} to {call.to_floor}")
            return
        
        # Find best elevator
        best_time = float('inf')
        best_route = None
        best_elevator = None
        
        for elevator in available_elevators:
            route = elevator.find_best_route(call.from_floor, call.to_floor)
            if route.total_time < best_time:
                best_time = route.total_time
                best_route = route
                best_elevator = elevator
        
        if best_elevator and best_route:
            best_elevator.add_route(best_route)
    
    def handle_passenger_delivery(self, floor_num: int, num_passengers: int, delivery_time: float):
        # Find completed calls for this destination floor
        delivered_count = 0
        calls_to_remove = []
        
        for passenger_id, call in self.active_calls.items():
            if call.to_floor == floor_num and delivered_count < num_passengers:
                waiting_time = delivery_time - call.call_time
                completed_call = CompletedCall(call, delivery_time, waiting_time)
                self.completed_calls.append(completed_call)
                calls_to_remove.append(passenger_id)
                delivered_count += 1
                if self.verbose:
                    print(f"Passenger {passenger_id} delivered to floor {floor_num}, waiting time: {waiting_time:.2f}s")
        
        # Remove delivered passengers from active calls
        for passenger_id in calls_to_remove:
            del self.active_calls[passenger_id]
    
    def get_average_waiting_time(self) -> float:
        if not self.completed_calls:
            return 0.0
        return sum(call.waiting_time for call in self.completed_calls) / len(self.completed_calls)
    
    def all_calls_completed(self) -> bool:
        has_pending = len(self.pending_calls) > 0
        has_active = len(self.active_calls) > 0
        elevators_busy = any(elevator.next_floor for elevator in self.elevators)
        
        return not (has_pending or has_active or elevators_busy)

class ElevatorSimulation:
    def __init__(self, floors_config: FloorsConfig, elevator_configs: List[ElevatorConfig], 
                 calls: List[Dict], time_ratio: float = 2.5, verbose: bool = False):
        self.time_service = TimeService(time_ratio)
        self.elevator_service = ElevatorService(floors_config, elevator_configs, self.time_service, verbose=verbose)
        self.calls = self._create_calls(calls)
        self.current_time = 0.0
        self.time_step = 0.1  # 100ms steps
        self.verbose = verbose
        
    def _create_calls(self, call_configs: List[Dict]) -> List[Call]:
        calls = []
        passenger_id = 0
        for config in call_configs:
            for _ in range(config.get('noPassengers', 1)):
                call = Call(
                    from_floor=config['from'],
                    to_floor=config['to'],
                    call_time=config.get('callTime', 0),
                    passenger_id=passenger_id
                )
                calls.append(call)
                passenger_id += 1
        return calls
    
    def run_simulation(self, max_time: float = 1000.0, verbose: bool = False) -> float:
        # Add all calls to the service
        for call in self.calls:
            self.elevator_service.add_call(call)
        
        if verbose:
            print(f"Starting simulation with {len(self.calls)} calls and {max_time=}")
        
        # Run simulation
        while self.current_time < max_time:
            # Update time service
            self.time_service.set_time(self.current_time)
            
            # Process pending calls
            self.elevator_service.process_pending_calls(self.current_time)
            
            # Update all elevators
            for elevator in self.elevator_service.elevators:
                elevator.update(self.time_step, self.current_time)
            
            # Check if simulation is complete
            if self.elevator_service.all_calls_completed():
                if verbose:
                    print(f"Simulation completed at time {self.current_time:.2f}s")
                break
                
            self.current_time += self.time_step
        
        avg_waiting_time = self.elevator_service.get_average_waiting_time()
        if verbose:
            print(f"Completed calls: {len(self.elevator_service.completed_calls)}")
            print(f"Average waiting time: {avg_waiting_time:.2f}s")
            
        return avg_waiting_time


def run_elevator_simulation_for_lifts(num_lifts: int, verbose: bool = False) -> float:
    floors_config = FloorsConfig(
        min_floor=-1,
        max_floor=6,
        heights=[3.2, 5.0, 4.75, 3.7, 2.75, 4.45, 3.8, 4.25]
    )
    max_passengers = 10
    T = 200
    
    # Define elevator configurations
    elevator_configs = [
        ElevatorConfig(-1, 6, 0, 8, 1.5, 6, 1.5)
        for _ in range(num_lifts)]
    
    
    # Define test calls
    def generate_poisson_calls(num_calls, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        calls = []
        call_times = np.cumsum(np.random.exponential(T / num_calls, num_calls))
        call_times = np.clip(call_times, 0, T)
        for i in range(num_calls):
            from_floor = random.randint(floors_config.min_floor, floors_config.max_floor)
            to_floor = random.randint(floors_config.min_floor, floors_config.max_floor)
            while to_floor == from_floor:
                to_floor = random.randint(floors_config.min_floor, floors_config.max_floor)
            call = {
                'from': from_floor,
                'to': to_floor,
                'noPassengers': random.randint(1, 4),  # Random number of passengers
                'callTime': float(call_times[i])
            }
            calls.append(call)
        return calls

    num_calls = 10 * num_lifts
    test_calls = generate_poisson_calls(num_calls)

    if verbose:
        print(f"Running simulation with {num_lifts} elevators and {num_calls} calls...")
    sim = ElevatorSimulation(floors_config, elevator_configs, test_calls, verbose=verbose)
    awt = sim.run_simulation()
    return awt
    
if __name__ == "__main__":
    max_num_lifts = 400
    verbose = False
    awts = []
    for num_lifts in range(1, max_num_lifts + 1):
        if verbose:
            print(f"\nTesting with {num_lifts} elevators:")
        awts.append(run_elevator_simulation_for_lifts(num_lifts, verbose=verbose))
    

    coefs = []
    # Print header
    header = ["i\\j"] + [str(j) for j in range(2, max_num_lifts + 1)]
    print("\t".join(header))
    for i, awt in enumerate(awts, start=1):
        if i == max_num_lifts:
            continue  # No j > i
        row = [str(i)]
        # Add empty cells for columns before j = i+1
        row += [""] * (i - 1)
        for j in range(i + 1, max_num_lifts + 1):
            coef = j / i * awts[i - 1] / awts[j - 1]
            row.append(f"{coef:.3f}")
            coefs.append(coef)
        print("\t".join(row))
    
    print('Tight bound', min(coefs))
    
