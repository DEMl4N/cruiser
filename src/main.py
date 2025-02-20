from neural_engine import NeuralEngine
from in_vehicle_communication import InVehicleCommunication
from car_state import CarState
from lane_detector import LaneDetectionModule
# from v2v_communication import V2VCommunication
import camera_module
import numpy as np
import cv2


def prepare_state_modules():
    car_state = CarState()
    return car_state

def prepare_neural_modules():
    lane_detector = LaneDetectionModule()
    # detection_engine = NeuralEngine("./obj_detect_v1.engine")
    # prediction_engine = NeuralEngine("./predict_v1.engine")
    control_engine = NeuralEngine("./control_v1.engine")

    # return front_camera, lane_detector, detection_engine, prediction_engine, control_engine
    return lane_detector, None, None, control_engine

def prepare_communcation_modules(car_state: CarState):
    in_vehicle_communication = InVehicleCommunication(car_state=car_state)
    # v2v_communication = V2VCommunication()
    # return in_vehicle_communication, v2v_communication
    return in_vehicle_communication, None

def execute_hdp():
    car_state = prepare_state_modules()
    lane_detector, detection_engine, prediction_engine, control_engine = prepare_neural_modules()
    in_vehicle_communication, v2v_communication = prepare_communcation_modules(car_state)

    # init camera
    pipeline = camera_module.gstreamer_pipeline()
    print("[+] Initializing camera...")
    front_camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if front_camera.isOpened():
        print("[+] Camera initialized successfully.")
    else:
        print("[-] Failed to initialize camera.")
        exit()

    if front_camera.isOpened():
        print("[+] Camera initialized successfully.")
        print("[+] Starting HDP...")
        # run every 30ms
        while True:
            cam_ret, frame = front_camera.read()
            if not cam_ret:
                print("[-] Camera read failed.")
                break

            frame_array = np.array(frame)
            frame_resized = cv2.resize(frame, dsize=(816, 462))
            # todo!: lane detection
            lane_offset = lane_detector.getLaneCurve(frame_resized, laneDiff=0)
            # detection_engine.infer(frame_array)
            # detection_result = detection_engine.get_output()

            # class_id = detection_result[0][0]
            # box = detection_result[1][0]
            # confidence = detection_result[2][0]
            # prediction_engine.infer(box)

            # prediction_result = prediction_engine.get_output()
            # front_vehicle_next_x = prediction_result[0][0]
            front_vehicle_next_x = 0
            # front_vehicle_next_dist = prediction_result[1][0]
            front_vehicle_next_dist = 1

            # lane_offset = lane_result[0][0]

            front_vehicle_x_delta = -1
            distance_delta = 1
            if car_state.front_vehicle_dist != 0:
                distance_delta = min(front_vehicle_next_dist / car_state.front_vehicle_dist, 1)

            input_data = np.array([
                front_vehicle_next_x,
                front_vehicle_x_delta,
                car_state.current_speed,
                car_state.target_speed,
                lane_offset,
                distance_delta,
            ], dtype=np.float32
            )

            control_engine.infer(input_data)
            control_result = control_engine.get_output()
            steering_angle = control_result[0][0]
            speed = car_state.current_speed + control_result[0][1]

            in_vehicle_communication.send_data(steering_angle, speed)



def main():
    execute_hdp()
    return

if __name__ == "__main__":
    main()