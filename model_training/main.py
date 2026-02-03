from ml_pipeline import MLPipeline
from tensorflow.keras.applications import ResNet50
# preprocess input function for ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as ResNet50_preprocess_input

def main():
    path_to_data = "C:\\Users\\marti\\Documents\\DL-Mechanical-Defects\\dataset_creation"
    pipeline = MLPipeline(path_to_data)

    # Example of adding a model to the pipeline

    # params include (Default values):
    # dense_units : 32
    # data_limit : 32
    # batch_size : 32
    # img_size : (300,300)
    # val_split : 0.2
    # epochs : 10

    pipeline.add_model(ResNet50, 
                       image_type=["initial"],                       
                       focus=["1", "2"],
                       preprocess_input=ResNet50_preprocess_input,
                       params={'dense_units': 512,
                               'data_limit':32,
                               'epochs':50,
                               'batch_size':32,})

    pipeline.print_models()
    # pipeline.run_pipeline()

if __name__ == "__main__":
    main()