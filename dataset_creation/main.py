from camera_controller import CameraController
import serial
import serial.tools.list_ports

camera = CameraController(LogLevel="INFO")
# camera.capture_image()

camera.send_command("#fKey9,1")
camera.send_command("#fKey9,0")



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
    	