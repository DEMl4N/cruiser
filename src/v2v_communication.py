import socket
import threading
import struct
import time
from src.car_state import CarState


class V2VCommunication:
    def __init__(self, car_state: CarState):
        self.local_host = '0.0.0.0'
        self.local_port = 3333
        self.dest_ip = '192.168.200.62'
        self.dest_port = 3333

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.local_host, self.local_port))

        self.running = True
        self.car_state = car_state

        self.handlers = {
            0x30: self.handle_car_state,
            0x50: self.handle_lane_change
        }

        self.recv_thread = threading.Thread(target=self.receiver)
        # self.send_thread = threading.Thread(target=self.sender)

    def start(self):
        self.recv_thread.start()
        # self.send_thread.start()

    def stop(self):
        self.running = False
        self.sock.close()
        self.recv_thread.join()
        # self.send_thread.join()

    def receiver(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                print(f"{addr} : {data.hex()}")

                msg_id = data[0]
                handler = self.handlers.get(msg_id)

                if handler:
                    received_data = handler(data)

                else:
                    print(f"Unknown msg_id")

            except Exception as e:
                print("receive fail:", e)
                break

    # 사용할시 교체
    def sender(self):
        while self.running:
            vehicle_no = 3
            lane_no = 1
            speed = 30.123

            # message = struct.pack('<BBBf', 0x40, self.vehicle_status.vehicle_no, self.vehicle_status.lane_no, self.vehicle_status.speed)
            try:
                # self.sock.sendto(message, (self.dest_ip, self.dest_port))
            except Exception as e:
                break

            time.sleep(1)


    def handle_car_state(self, data):
        if len(data) != 6:
            return

        vehicle_speed = round(struct.unpack('<f', data[1:5])[0] ,3)
        aeb_flag = data[5]

        self.car_state.update_front_vehicle_speed(vehicle_speed)
        self.car_state.update_aeb_state(aeb_flag)


    def handle_lane_change(self, data):
        if data[1] == 0 or data[1] == 1:
            self.car_state.update_lane(data[1])
        print("[+] Lane changed to", data[1])

