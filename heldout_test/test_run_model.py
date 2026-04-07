import tensoflow as tf
import time
import msvcrt
from tensorflow.keras.models import load_model
from ..dataset_creation.image_cutting import ImagePreprocessor
from ..dataset_creation.camera_controller import CameraController
# path to finished_models
dir_models = "C:\\Programmering\\Masters\\DL-Mechanical-Defects\\model_training\\finished_models"

# load models
model1 = load_model(f"{dir_models}\\name_focus1.h5")
model2 = load_model(f"{dir_models}\\name_focus2.h5")

def run_model():
    # initialize camera controller and image preprocessor
    cam_controller = CameraController()
    img_preprocessor = ImagePreprocessor()

    # get image from camera
    cam_controller.send_command("#rpre1")
    time.sleep(0.6)
    frame1 = cam_controller.get_image()

    cam_controller.send_command("#rpre2")
    time.sleep(0.6)
    frame2 = cam_controller.get_image()

    # preprocess image
    preprocessed_image1 = img_preprocessor.preprocess(frame1, focus="1")
    preprocessed_image2 = img_preprocessor.preprocess(frame2, focus="2")
    # run models on preprocessed images
    prediction1 = model1.predict(preprocessed_image1)
    prediction2 = model2.predict(preprocessed_image2)

    # print predictions
    print(f"Prediction for focus 1: {prediction1}")
    print(f"Prediction for focus 2: {prediction2}")

    cam_controller.camera_release()

if __name__ == "__main__":
    while True:
        key = msvcrt.getch()
        if key.lower() == b'c':
            run_model()
        elif key.lower() == b'q':
            break
        



