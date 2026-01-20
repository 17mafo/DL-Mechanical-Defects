import serial
import serial.tools.list_ports
from cv2 import VideoCapture, imshow, imwrite, waitKey, destroyWindow

# List all COM ports
print("Available COM ports:")
for port in serial.tools.list_ports.comports():
    print(f"{port.device} - {port.description}")

# PORT = "COM3"
# BAUD = 9600

# with serial.Serial(PORT, BAUD, timeout=1) as ser:
#     ser.write(b"#af1 \r")

cam = VideoCapture(1)

ret, frame = cam.read()

if ret:
    # imshow("Captured",frame)
    imwrite("captured_image.jpg", frame)
    # waitKey(0)
    # destroyWindow("Captured")
else:
    print("Failed to capture image")