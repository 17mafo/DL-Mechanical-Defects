import copy
import tensorflow as tf
from models.baseModel import BaseModel as bm

class MLPipeline:
    def __init__(self, path_to_data):
        print("Initializing ML Pipeline")
        self.path_to_data = path_to_data
        self.models = []

    def add_model(self, model, image_type, focus, preprocess_input, **params):
        # Create model instances for each image_type and focus combination
        for type in image_type:
            for f in focus:
                good_paths = [self.path_to_data + f"/processed_images/{type}/good/{f}"]
                bad_paths = [self.path_to_data + f"/processed_images/{type}/bad/{f}"]
                # if(augmented):
                #     good_paths = [self.path_to_data + f"/processed_images_augmented/{type}/good/{f}"]
                #     bad_paths = [self.path_to_data + f"/processed_images_augmented/{type}/bad/{f}"]
                
                train_ds, val_ds = self.create_datasets(good_paths, bad_paths,
                                                        batch_size = params.get('batch_size', 32),
                                                        img_size = params.get('img_size', (300,300)),
                                                        val_split= params.get('val_split', 0.2),
                                                        data_limit= params.get('data_limit', 500))
                self.models.append([f"{model().name}_{type}_{f}",model, params, train_ds, val_ds, preprocess_input])

    def print_models(self):
        print("Current models in the pipeline:")
        for i, model in enumerate(self.models):
            print(f"Model {i+1}:")
            print(f"  Model Name: {model[0]}")
            print(f"  Parameters: {model[2]}")
            print(f"  Training Dataset: {model[3]}")
            print(f"  Validation Dataset: {model[4]}")
            print(model)

    def create_datasets(self, good_paths, bad_paths, batch_size=32, img_size=(300,300), val_split=0.2, data_limit=500):
        # Load GOOD images (label = 0)
        for i, path in enumerate(good_paths):
            good_ds_part = self.get_data(path, batch_size, img_size)
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
            bad_ds_part = self.get_data(path, batch_size, img_size)
            bad_ds_part = bad_ds_part.map(
                lambda x: (x, tf.ones((tf.shape(x)[0], 1))),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            if i == 0:
                bad_ds = bad_ds_part
            else:
                bad_ds = bad_ds.concatenate(bad_ds_part)
        
        # Limit dataset size for testing
        good_ds = good_ds.take(data_limit // batch_size)
        bad_ds = bad_ds.take(data_limit // batch_size)

        # Combine datasets
        dataset = good_ds.concatenate(bad_ds)
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=False)

        # Data augmentation pipeline
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.028),
        ])

        def augment(image, label):
            return data_augmentation(image, training=True), label

        # Create augmented copy
        augmented_ds = dataset.map(
            augment,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Double dataset size
        dataset = dataset.concatenate(augmented_ds)

        # Train / validation split
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        val_size = int(dataset_size * val_split)

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

    def run_pipeline(self):
        for model in self.models:
            mod = bm(model[0], **model[1])
            mod.preprocess(model[2], model[3], model[4])
            mod.train(**model[1])
            mod.summary()


    