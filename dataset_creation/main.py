from camera_controller import CameraController
import msvcrt

camera = CameraController(LogLevel="INFO")


while True:
    # get active count from json file
    active_count = 0
    try :
        read_file = open("dataset_creation/active_count.json", "r")
        active_count = int(read_file.read())
        read_file.close()
    except FileNotFoundError:
        write_file = open("dataset_creation/active_count.json", "w")
        write_file.write("0")
        write_file.close()
    print(f"Current active count: {active_count}")
    
    try:
        print('Press c to capture image, and q to quit:\n')
        key = msvcrt.getch()
        if key.lower() == b'c':
            active_count += 1
            camera.save_preset_images(filename_prefix=f"{active_count}_",path="dataset_creation/images")
            write_file = open("dataset_creation/active_count.json", "w")
            write_file.write(str(active_count))
            write_file.close()
        elif key.lower() == b'q':
            camera.camera_release()
            exit(1)
            break
    except Exception as e:
        print(f"Error: {e}")
        camera.camera_release()
        break



    	