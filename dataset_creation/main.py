from camera_controller import CameraController
import serial
import serial.tools.list_ports
import time

camera = CameraController(LogLevel="INFO")
# camera.capture_image()

# camera.send_command("#fKey9,1")
# camera.send_command("#fKey9,0")
# camera.clear_presets(1)j
# camera.save_preset(1)
camera.save_preset_images(filename_prefix="125_",path="dataset_creation/images/good")



    	