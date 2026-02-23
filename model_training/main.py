from ml_pipeline import MLPipeline
import tensorflow as tf
import argparse
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as ResNet50_preprocess_input

from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as ResNet101V2_preprocess_input

from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as EfficientNetV2S_preprocess_input

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as VGG16_preprocess_input

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as VGG19_preprocess_input

from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as EfficientNetV2B3_preprocess_input

from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as ResNet152V2_preprocess_input

# preprocess input function for ResNet50

def main(data_path = None, gpu_index=None):
    # Set GPU if specified, for example python main.py --gpu 0
    if gpu_index is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPUs found!")
        elif gpu_index < 0 or gpu_index >= len(gpus):
            print(f"GPU index {gpu_index} out of range. Available GPUs: {len(gpus)}")
        else:
            tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            print(f"Using GPU {gpu_index}: {gpus[gpu_index]}")
    if data_path is not None:
        path_to_data = data_path

    pipeline = MLPipeline(path_to_data)
    pipeline_cv = MLPipeline(path_to_data)

    # Example of adding a model to the pipeline

    # params include (Default values):
    # dense_units : 32
    # data_limit : 32
    # batch_size : 32
    # img_size : (300,300)
    # val_split : 0.2
    # epochs : 10ls
    # augmentation : True

    pipeline.add_model(ResNet152V2, 
                       image_type=["initial","background","outer_rim"],                       
                       focus=["1", "2"],
                       preprocess_input=ResNet152V2_preprocess_input,
                       dense_units = 256,
                       data_limit=700,
                       val_split=0.3,
                       epochs=50,
                       batch_size=32,
                       augmentation=True,)
    
    pipeline.add_model(VGG19, 
                       image_type=["initial","background","outer_rim"],                       
                       focus=["1", "2"],
                       preprocess_input=VGG19_preprocess_input,
                       dense_units = 256,
                       data_limit=700,
                       val_split=0.3,
                       epochs=50,
                       batch_size=32,
                       augmentation=True,)


    pipeline.add_model(EfficientNetV2B3, 
                       image_type=["initial","background","outer_rim"],                       
                       focus=["1", "2"],
                       preprocess_input=EfficientNetV2B3_preprocess_input,
                       dense_units = 256,
                       data_limit=700,
                       val_split=0.3,
                       epochs=50,
                       batch_size=32,
                       augmentation=True,
                       include_preprocessing=True)

    # pipeline.print_models()

    # Normal running of the pipeline (train all models sequentially)
    pipeline.run_pipeline()
    pipeline.plot_histories()


    # Cross validation
    # pipeline.run_cross_validation(folds=5)
    # pipeline.plot_cross_validation_results()


    # pipeline_cv.add_model(ResNet152V2, 
    #                    image_type=["initial"],                       
    #                    focus=["1", "2"],
    #                    preprocess_input=ResNet152V2_preprocess_input,
    #                    dense_units = 256,
    #                    data_limit=700,
    #                    val_split=0,
    #                    epochs=50,
    #                    batch_size=32,
    #                    augmentation=True,)
    # pipeline_cv.run_cross_validation(folds=5)
    # pipeline_cv.plot_cross_validation_results()
    # pipeline_cv.run_pipeline() # Train on all data after cross validation
    # pipeline_cv.plot_histories()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML models with GPU selection')
    parser.add_argument('--gpu', type=int, default=None, help='GPU index to use (e.g., 0, 1, 2...)')
    parser.add_argument('--data_path', type=str, default=None, help='Path to the dataset')
    args = parser.parse_args()
    
    main(data_path=args.data_path, gpu_index=args.gpu)