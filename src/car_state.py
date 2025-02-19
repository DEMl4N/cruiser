

class CarState:
    def __init__(self):
        self.current_speed = 0
        self.target_speed = 40
        self.target_lane = 0
        self.front_vehicle_dist = 0
        self.lane_offset = 0
        self.front_vehicle_speed = 0
        self.aeb_state = 0

    def update_current_speed(self, speed):
        self.current_speed = speed

    def update_target_speed(self, speed):
        self.target_speed = speed

    def update_target_lane(self, lane):
        self.target_lane = lane

    def update_front_vehicle_dist(self, dist):
        self.front_vehicle_dist = dist

    def update_lane_offset(self, offset):
        self.lane_offset = offset

    def update_front_vehicle_speed(self, speed):
        self.front_vehicle_speed = speed

    def update_aeb_state(self, state):
        self.aeb_state = state
