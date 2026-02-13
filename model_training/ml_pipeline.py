import tensorflow as tf
from models.baseModel import BaseModel as bm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os

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
                
                self.models.append({ "name": f"{model().name}_{type}_{f}", "model_cls": model,
                "params": params, "good_paths": good_paths, "bad_paths": bad_paths, "preprocess": preprocess_input})

    def print_models(self):
        print("Current models in the pipeline:")
        for i, model in enumerate(self.models):
            print(f"Model {i+1}:")
            print(f"  Model Name: {model['name']}")
            print(f"  Parameters: {model['params']}")
            print(f"  Training Dataset: {model['good_paths']}")
            print(f"  Validation Dataset: {model['bad_paths']}")
            print(model)

    def create_full_dataset(self, good_paths, bad_paths,
                            img_size=(300,300), data_limit=500):
        
        good_ds = None
        bad_ds = None
        # GOOD = 0
        for i, path in enumerate(good_paths):
            ds_part = self.get_data(path, label=0.0, batch_size=1, img_size=img_size)
            ds_part = ds_part.unbatch()
            if good_ds is None:
                good_ds = ds_part
            else:
                good_ds = good_ds.concatenate(ds_part)

        # BAD = 1
        for i, path in enumerate(bad_paths):
            ds_part = self.get_data(path, label=1.0, batch_size=1, img_size=img_size)
            ds_part = ds_part.unbatch()
            if bad_ds is None:
                bad_ds = ds_part
            else:
                bad_ds = bad_ds.concatenate(ds_part)

        ds = good_ds.concatenate(bad_ds)
        ds = ds.take(data_limit)

        return ds


    def create_datasets(self, good_paths, bad_paths, 
                        batch_size=32, img_size=(300,300), 
                        val_split=0.2, data_limit=500, 
                        augmentation=False, cross_validation=False, k_folds=10, fold_index=0):
        
        full_ds = self.create_full_dataset(good_paths, bad_paths, 
                                           img_size=img_size, data_limit=data_limit)

        # Extract labels for stratification
        samples = list(full_ds)
        labels = np.array([y.numpy() for _, y in samples])
        n = len(samples)

        if cross_validation:
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            train_idx, val_idx = list(skf.split(np.zeros(n), labels))[fold_index]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
            train_idx, val_idx = next(sss.split(np.zeros(n), labels))

        train_ds = tf.data.Dataset.from_tensor_slices([samples[i] for i in train_idx])
        val_ds = tf.data.Dataset.from_tensor_slices([samples[i] for i in val_idx])

        if augmentation:
            aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.028),
            ])

            train_ds = train_ds.map(
                lambda x, y: (aug(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        train_ds = (train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE))

        val_ds = (val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE))

        return train_ds, val_ds

    def get_data(self, path_to_data, label, batch_size, img_size=(300,300)):
        ds = tf.keras.utils.image_dataset_from_directory(
            path_to_data,
            image_size=img_size,
            batch_size=batch_size,
            label_mode=None,
            shuffle=True
        )
        return ds.map(
            lambda x: (x, tf.fill((tf.shape(x)[0], 1), label)),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    def run_pipeline(self):
        self.hists = []

        for model in self.models:
            train_ds, val_ds = self.create_datasets(model["good_paths"], model["bad_paths"], **model["params"])

            mod = bm(model["model_cls"], **model["params"])
            mod.preprocess(train_ds, val_ds, model["preprocess"])
            history = mod.train(**model["params"])

            self.hists.append([model["name"], history])


    def run_cross_validation(self, folds=10):
        self.hists = []

        for model in self.models:
            fold_histories = []

            for fold in range(folds):
                train_ds, val_ds = self.create_datasets(model["good_paths"],model["bad_paths"],
                    cross_validation=True, k_folds=folds, fold_index=fold, **model["params"])

                mod = bm(model["model_cls"], **model["params"])
                mod.preprocess(train_ds, val_ds, model["preprocess"])

                history = mod.train(**model["params"])
                fold_histories.append(history)

            self.hists.append([model["name"], fold_histories])

    
    def save_models(self):
        for model in self.models:
            # get path from current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # if path does not exist, create it
            if not os.path.exists(f"{current_dir}/finished_models"):
                os.makedirs(f"{current_dir}/finished_models")
            model['model_cls'].save(f"{current_dir}/finished_models/{model['name']}_model.h5")


    def plot_histories(self):

        for hist in self.hists:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(hist[1].history['loss'], label='Training Loss')
            plt.plot(hist[1].history['val_loss'], label='Validation Loss')
            plt.title(f'Loss for {hist[0]}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(hist[1].history['accuracy'], label='Training Accuracy')
            plt.plot(hist[1].history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Accuracy for {hist[0]}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            # save figure
            plt.savefig(f"{hist[0]}_training_history.png")

            # plt.show()




    