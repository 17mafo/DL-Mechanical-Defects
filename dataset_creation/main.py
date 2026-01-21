from camera_controller import CameraController
import serial
import serial.tools.list_ports
import time
import msvcrt

camera = CameraController(LogLevel="INFO")


while True:
    # get active count from json file
    active_count = 0
    try :
        read_file = open("dataset_creation/active_count.json", "r")
        active_count = int(read_file.read())
        read_file.close()
    except:
        pass
    print(f"Current active count: {active_count}")
    
    try:
        print('Press s or n to continue:\n')
        key = msvcrt.getch()
        if key.lower() == 'c':
            active_count += 1
            camera.save_preset_images(filename_prefix=f"{active_count}_",path="dataset_creation/images")
            # update active count in json file
            write_file = open("dataset_creation/active_count.json", "w")
            write_file.write(str(active_count))
            write_file.close()
        elif key.lower() == 'q':
            camera.camera_release()
            exit(1)
            break
    except Exception as e:
        print(f"Error: {e}")
        camera.camera_release()
        break
    # camera.save_preset_images(filename_prefix="125_",path="dataset_creation/images")



    	