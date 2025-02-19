import time
import serial
import threading
import struct
from car_state import CarState

class InVehicleCommunication:
    def __init__(self, car_state: CarState):
        self.serial_port = serial.Serial(
            port="/dev/ttyTHS1",
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        self.car_state = car_state

        time.sleep(1)

        self.read_thread = threading.Thread(target=self.uart_read_handler)
        self.read_thread.daemon = True
        self.read_thread.start()

    def read_data(self):
        if self.serial_port.inWaiting() > 0:
            return self.serial_port.read()
        else:
            return None

    def send_data(self, angle: float, velocity: float):
        """
        메시지 구조: [헤더(1바이트), angle(4바이트), velocity(4바이트)]
        """
        tx_buffer = bytearray(9)
        tx_buffer[0] = 0x20  # 헤더 값
        struct.pack_into('<f', tx_buffer, 1, angle)
        struct.pack_into('<f', tx_buffer, 5, velocity)
        self.serial_port.write(tx_buffer)

    def uart_read_handler(self):
        """
        메시지 구조: [헤더(1바이트), velocity(4바이트)]
        """
        while True:
            if self.serial_port.in_waiting >= 5:
                rx_buffer = self.serial_port.read(5)
                # 각 바이트를 16진수 문자열로 변환 (디버깅 용)
                hex_str = ' '.join('{:02x}'.format(b) for b in rx_buffer)
                # print("Received Bytes:", hex_str)

                try:
                    # rx_buffer[1:5]에 저장된 4바이트를 float형 velocity로 변환
                    velocity = struct.unpack('<f', rx_buffer[1:5])[0]
                    self.car_state.update_current_speed(velocity)
                except struct.error as e:
                    print("Unpack Error:", e)
                    continue

    def close(self):
        self.serial_port.close()
