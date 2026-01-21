import serial
import serial.tools.list_ports
from cv2 import WINDOW_NORMAL, VideoCapture, imwrite, imshow, moveWindow, namedWindow, resizeWindow, waitKey, destroyAllWindows
import os
import msvcrt


# List all COM ports




class CameraController:
    def __init__(self, port="COM3", baudrate=9600, LogLevel="INFO",cameraIdx=1):
        self.port = port
        self.baudrate = baudrate
        self.LogLevel = LogLevel
        self.cam = VideoCapture(cameraIdx)

    def capture_image(self, filename="image.jpg", path=os.path.dirname(__file__) + "/images"):
        if not os.path.exists(path):
            os.makedirs(path)
        ret, frame = self.cam.read()
        if ret:
            imwrite(f"{path}/{filename}", frame)
            destroyAllWindows()
            if self.LogLevel == "INFO":
                print(f"Image saved as {path}/{filename}")
        else:
            if self.LogLevel == "ERROR":
                print("Failed to capture image")

    def get_image(self, filename="image.jpg", path=os.path.dirname(__file__) + "/images"):
        ret, frame = self.cam.read()
        if ret:
            return frame
        else:
            if self.LogLevel == "ERROR":
                print("Failed to capture image")
    
    def camera_release(self):
        self.cam.release()
    
    def save_preset_images(self, filename_prefix="good/bad", path=os.path.dirname(__file__) + "/images"):
        loop = True
        while loop:
            self.send_command("#rpre1")
            frame1 = self.get_image()
            namedWindow("win1", WINDOW_NORMAL)
            resizeWindow("win1", 800 , 450)
            moveWindow("win1", 140, - 1080 + 200)  # x, y in pixels from top-left of screen

            self.send_command("#rpre2")
            frame2 = self.get_image()
            namedWindow("win2", WINDOW_NORMAL)
            resizeWindow("win2", 800 , 450)
            moveWindow("win2", 940, - 1080 + 200)  # x, y in pixels from top-left of screen

            imshow("win1", frame1)
            imshow("win2", frame2)
            key = waitKey(0)
            if key == ord('a'):
                print ("pressed a")
                filename_prefix += "good_"
                path += "/good"
                loop = False

            elif key == ord('r'):
                destroyAllWindows()
                continue
            else:
                print("pressed other key")
                filename_prefix += "bad_"
                path += "/bad"
                loop = False

        imwrite(f"{path}/{filename_prefix}focus_1.jpg", frame1)
        imwrite(f"{path}/{filename_prefix}focus_2.jpg", frame2)
        destroyAllWindows()
    
    def send_command(self, command):
        with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
            ser.write(f"{command} \r".encode())
            if self.LogLevel == "INFO":
                print(f"Sent command: {command}")

    def clear_presets(self,preset_number):
        self.send_command(f"#cpre{preset_number}")

    def save_preset(self, preset_number):
        self.send_command(f"#spre{preset_number}")

    def recall_preset(self, preset_number):
        self.send_command(f"#rpre{preset_number}")
    






# print("Available COM ports:")
# for port in serial.tools.list_ports.comports():
#     print(f"{port.device} - {port.description}")

# PORT = "COM3"
# BAUD = 9600

# with serial.Serial(PORT, BAUD, timeout=1) as ser:
#     # ser.write(b"#targetzpos7000 \r")
#     # ser.write(b"#targetzpos16384 \r")
#     # ser.write(b"#af0 \r")
#     ser.write(b"#fKey9,1 \r")
#     ser.write(b"#fKey9,0 \r")


# Func 0 = av 1 = på
# ("#fKey7,0 \r");
# ("#fKey7,1 \r");