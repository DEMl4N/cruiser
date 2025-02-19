import cv2

class CameraModule:
    def __init__(
            self,
            sensor_id=0,
            capture_width=3264,
            capture_height=1848,
            display_width=816,
            display_height=462,
            framerate=28,
            flip_method=0,
    ):
        self.sensor_id = sensor_id
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.display_width = display_width
        self.display_height = display_height
        self.framerate = framerate
        self.flip_method = flip_method
        self.video_capture = None

        self.init_camera()

    def gstreamer_pipeline(
            self
    ):
        return (
                "nvarguscamerasrc sensor-id=%d ! "
                "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    self.sensor_id,
                    self.capture_width,
                    self.capture_height,
                    self.framerate,
                    self.flip_method,
                    self.display_width,
                    self.display_height,
                )
        )

    def init_camera(self):
        print("[+] Initializing camera...")
        video_capture = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)

        if video_capture.isOpened():
            print("[+] Camera initialized successfully.")
            self.video_capture = video_capture
        else:
            print("[-] Failed to initialize camera.")
            exit()

    def get_frame(self):
        frame = None
        if self.video_capture.isOpened():
            try:
                _, frame = self.video_capture.read()

            except Exception as e:
                print(f"[-] Error reading frame: {e}")

            finally:
                self.video_capture.release()

        return frame
