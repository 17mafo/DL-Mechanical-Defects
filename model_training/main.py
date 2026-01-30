import serial
import serial.tools.list_ports

from dataset_creation.camera_controller import CameraController

print("Available COM ports:")
for port in serial.tools.list_ports.comports():
    print(f"{port.device} - {port.description}")

PORT = "COM4"
BAUD = 9600
camera = CameraController(LogLevel="INFO", cameraIdx=1, port="COM4")

camera.save_preset(number=2)

# with serial.Serial(PORT, BAUD, timeout=1) as ser:
    # ser.write(b"#targetzpos7000 \r")
    # ser.write(b"#targetzpos16384 \r")
    # ser.write(b"#af0 \r")
    # ser.write(b"#fKey9,1 \r")
    # ser.write(b"#fKey9,0 \r")