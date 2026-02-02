from ml_pipeline import MLPipeline
from tensorflow.keras.applications import ResNet50

def main():
    path_to_data = "C:\\Users\\marti\\Documents\\DL-Mechanical-Defects\\dataset_creation"
    pipeline = MLPipeline(path_to_data)

    # Example of adding a model to the pipeline
    path = "C:\\Users\\marti\\Documents\\DL-Mechanical-Defects\\dataset_creation"
    pipeline.add_model(ResNet50, path, focusmodels=True, bad_but_looks_good_equals_bad=None)
    
    pipeline.print_models()
if __name__ == "__main__":
    main()