from ml_pipeline import MLPipeline
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as ResNet50_preprocess_input

from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as ResNet101V2_preprocess_input

from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as EfficientNetV2_preprocess_input

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as VGG16_preprocess_input

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as VGG19_preprocess_input

from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as EfficientNetV2B3_preprocess_input



# preprocess input function for ResNet50

def main():
    path_to_data = "/mnt/c/Users/Marti/Documents/DL-Mechanical-Defects/dataset_creation"
    pipeline = MLPipeline(path_to_data)

    # Example of adding a model to the pipeline

    # params include (Default values):
    # dense_units : 32
    # data_limit : 32
    # batch_size : 32
    # img_size : (300,300)
    # val_split : 0.2
    # epochs : 10ls
    # augmentation : True

    # pipeline.add_model(ResNet101V2, 
    #                    image_type=["initial"],                       
    #                    focus=["1", "2"],
    #                    preprocess_input=ResNet101V2_preprocess_input,
    #                    dense_units = 256,
    #                    data_limit=600,
    #                    val_split=0.3,
    #                    epochs=50,
    #                    batch_size=32,
    #                    augmentation=True,)

    pipeline.add_model(EfficientNetV2B3, 
                       image_type=["initial"],                       
                       focus=["1", "2"],
                       preprocess_input=EfficientNetV2B3_preprocess_input,
                       dense_units = 256,
                       data_limit=600,
                       val_split=0.3,
                       epochs=50,
                       batch_size=32,
                       augmentation=True,
                       include_preprocessing=False)

    # pipeline.print_models()
    pipeline.run_pipeline()
    pipeline.plot_histories()

if __name__ == "__main__":
    main()