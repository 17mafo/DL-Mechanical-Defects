import copy
import tensorflow as tf
from models.baseModel import BaseModel as bm

class MLPipeline:
    def __init__(self, path_to_data):
        print("Initializing ML Pipeline")
        self.path_to_data = path_to_data
        self.models = []

    
    def add_model(self, model, path, focusmodels = True, bad_but_looks_good_equals_bad = None, **params):
        # This code is not understandable and creates all the needed models depending on the "focusmodels and bad_but_looks_good_equals_bad", Good luck :)
        def create_paths(path, bad_but_looks_good_equals_bad, image_type):
            good_paths = []
            bad_paths = []
            good_paths.append(path + f"/processed_images/{image_type}/good")
            bad_paths.append(path + f"/processed_images/{image_type}/bad")
            if(bad_but_looks_good_equals_bad):
                bad_paths.append(path + f"/processed_images/{image_type}/bad_but_looks_good")
            else:
                good_paths.append(path + f"/processed_images/{image_type}/bad_but_looks_good")

            return good_paths, bad_paths

        for image_type in (["background", "initial", "outer_rim"]):
            temp_focusModel = []
            if(bad_but_looks_good_equals_bad is None):
                # Model 1
                good_paths, bad_paths = create_paths(path, False, image_type)
                temp_focusModel.append([good_paths, bad_paths])

                # Model 2
                good_paths2, bad_paths2 = create_paths(path, True, image_type)
                temp_focusModel.append([good_paths2, bad_paths2])

            elif(bad_but_looks_good_equals_bad):
                good_paths, bad_paths = create_paths(path, True, image_type)
                temp_focusModel.append([good_paths, bad_paths])
            else:
                good_paths, bad_paths = create_paths(path, False, image_type)
                temp_focusModel.append([good_paths, bad_paths])

            for good_path, bad_path in temp_focusModel:
                if(focusmodels is None):
                    all_good_focus1 = good_path.copy()
                    all_bad_focus1 = bad_path.copy()
                    all_good_focus1 = [p + "/1" for p in good_path]
                    all_bad_focus1  = [p + "/1" for p in bad_path]

                    all_good_focus2 = good_path.copy()
                    all_bad_focus2 = bad_path.copy()
                    all_good_focus2 = [p + "/2" for p in good_path]
                    all_bad_focus2  = [p + "/2" for p in bad_path]


                    # Model 1
                    train_ds, val_ds = self.create_datasets(all_good_focus1, all_bad_focus1)
                    self.models.append([model, params, train_ds, val_ds, all_good_focus1, all_bad_focus1])
                    # Model 2
                    train_ds, val_ds = self.create_datasets(all_good_focus2, all_bad_focus2)
                    self.models.append([model, params, train_ds, val_ds, all_good_focus2, all_bad_focus2])
                
                    all_good_focus1 = good_path.copy()
                    all_bad_focus1 = bad_path.copy()
                    all_good_focus1 = [p + "/1" for p in good_path]
                    all_bad_focus1  = [p + "/1" for p in bad_path]

                    all_good_focus2 = good_path.copy()
                    all_bad_focus2 = bad_path.copy()
                    all_good_focus2 = [p + "/2" for p in good_path]
                    all_bad_focus2  = [p + "/2" for p in bad_path]
                    
                    # Model 1
                    all_good_focus1.extend(all_good_focus2)
                    all_bad_focus1.extend(all_bad_focus2)
                    train_ds, val_ds = self.create_datasets(all_good_focus1, all_bad_focus1)
                    self.models.append([model, params, train_ds, val_ds, all_good_focus1, all_bad_focus1])

                elif(focusmodels):
                    all_good_focus1 = good_path.copy()
                    all_bad_focus1 = bad_path.copy()
                    all_good_focus1 = [p + "/1" for p in good_path]
                    all_bad_focus1  = [p + "/1" for p in bad_path]

                    all_good_focus2 = good_path.copy()
                    all_bad_focus2 = bad_path.copy()
                    all_good_focus2 = [p + "/2" for p in good_path]
                    all_bad_focus2  = [p + "/2" for p in bad_path]

                    # Model 1
                    train_ds, val_ds = self.create_datasets(all_good_focus1, all_bad_focus1)
                    self.models.append([model, params, train_ds, val_ds, all_good_focus1, all_bad_focus1])
                    # Model 2
                    train_ds, val_ds = self.create_datasets(all_good_focus2, all_bad_focus2)
                    self.models.append([model, params, train_ds, val_ds, all_good_focus2, all_bad_focus2])
                else:
                    all_good_focus1 = good_path.copy()
                    all_bad_focus1 = bad_path.copy()
                    all_good_focus1 = [p + "/1" for p in good_path]
                    all_bad_focus1  = [p + "/1" for p in bad_path]

                    all_good_focus2 = good_path.copy()
                    all_bad_focus2 = bad_path.copy()
                    all_good_focus2 = [p + "/2" for p in good_path]
                    all_bad_focus2  = [p + "/2" for p in bad_path]
                    
                    # Model 1
                    all_good_focus1.extend(all_good_focus2)
                    all_bad_focus1.extend(all_bad_focus2)
                    train_ds, val_ds = self.create_datasets(all_good_focus1, all_bad_focus1)
                    self.models.append([model, params, train_ds, val_ds, all_good_focus1, all_bad_focus1])

                
    def print_models(self):
        print("Current models in the pipeline:")
        for i, model in enumerate(self.models):
            print(f"Model {i+1}:")
            print(f"  Model Type: {model[0]}")
            print(f"  Parameters: {model[1]}")
            print(f"  Training Dataset: {model[2]}")
            print(f"  Validation Dataset: {model[3]}")
            print(model)



    def create_datasets(self, good_paths, bad_paths, BATCH_SIZE=32, IMG_SIZE=(300,300), VAL_SPLIT=0.2, DATA_LIMIT=500):
        # Load GOOD images (label = 0)
        for i, path in enumerate(good_paths):
            good_ds_part = self.get_data(path, BATCH_SIZE, IMG_SIZE)
            good_ds_part = good_ds_part.map(
                lambda x: (x, tf.zeros((tf.shape(x)[0], 1))),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            if i == 0:
                good_ds = good_ds_part
            else:
                good_ds = good_ds.concatenate(good_ds_part)

        # Load BAD images (label = 1)
        for i, path in enumerate(bad_paths):
            bad_ds_part = self.get_data(path, BATCH_SIZE, IMG_SIZE)
            bad_ds_part = bad_ds_part.map(
                lambda x: (x, tf.ones((tf.shape(x)[0], 1))),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            if i == 0:
                bad_ds = bad_ds_part
            else:
                bad_ds = bad_ds.concatenate(bad_ds_part)
        
        # Limit dataset size for testing
        good_ds = good_ds.take(DATA_LIMIT // BATCH_SIZE)
        bad_ds = bad_ds.take(DATA_LIMIT // BATCH_SIZE)

        # Combine datasets
        dataset = good_ds.concatenate(bad_ds)
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=False)

        # Train / validation split
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        val_size = int(dataset_size * VAL_SPLIT)

        val_ds = dataset.take(val_size)
        train_ds = dataset.skip(val_size)
        return train_ds, val_ds

    def get_data(self,path_to_data, batch_size, img_size = (300,300)):
        return tf.keras.utils.image_dataset_from_directory(
            path_to_data,
            image_size=img_size,
            batch_size=batch_size,
            label_mode=None,
            shuffle=True
        )

    def run_pipline (self):
        for model in self.models:
            bm
            pass

    

    