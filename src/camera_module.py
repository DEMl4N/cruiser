import cv2


def gstreamer_pipeline(
        sensor_id=0,
        capture_width=3264,
        capture_height=1848,
        display_width=640,
        display_height=640,
        framerate=28,
        flip_method=0
):
    return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


# def init_camera(self):
#     print("[+] Initializing camera...")
#     video_capture = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
#
#     if video_capture.isOpened():
#         print("[+] Camera initialized successfully.")
#         self.video_capture = video_capture
#     else:
#         print("[-] Failed to initialize camera.")
#         exit()


# def get_frame(self):
#     frame = None
#     if self.video_capture.isOpened():
#         try:
#             _, frame = self.video_capture.read()
#
#         except Exception as e:
#             print(f"[-] Error reading frame: {e}")
#
#         finally:
#             self.video_capture.release()
#
#     return frame
