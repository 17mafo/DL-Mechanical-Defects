import serial
import serial.tools.list_ports
from cv2 import VideoCapture, imwrite
import os

# List all COM ports




class CameraController:
    def __init__(self, port="COM3", baudrate=9600, LogLevel="INFO"):
        self.port = port
        self.baudrate = baudrate
        self.LogLevel = LogLevel

    def capture_image(self, cameraIdx=1, filename="image.jpg", path=os.path.dirname(__file__) + "/images"):
        if not os.path.exists(path):
            os.makedirs(path)
        cam = VideoCapture(cameraIdx)
        ret, frame = cam.read()
        if ret:
            imwrite(f"{path}/{filename}", frame)
            if self.LogLevel == "INFO":
                print(f"Image saved as {path}/{filename}")
        else:
            if self.LogLevel == "ERROR":
                print("Failed to capture image")
        cam.release()
    
    def send_command(self, command):
        with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
            ser.write(f"{command} \r".encode())
            if self.LogLevel == "INFO":
                print(f"Sent command: {command}")
