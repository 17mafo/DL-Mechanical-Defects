import copy
import tensorflow as tf
from models.baseModel import BaseModel as bm
import matplotlib.pyplot as plt

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
                                                        val_split = params.get('val_split', 0.2),
                                                        data_limit = params.get('data_limit', 500),
                                                        augmentation = params.get('augmentation', False))
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

    def create_datasets(self, good_paths, bad_paths, batch_size=32, img_size=(300,300), val_split=0.2, data_limit=500, augmentation=False):
        # Load GOOD images (label = 0)
        for i, path in enumerate(good_paths):
            good_ds_part = self.get_data(path, label=0.0, batch_size=batch_size, img_size=img_size)
            # good_ds_part = good_ds_part.map(
            #     lambda x: (x, tf.zeros((tf.shape(x)[0], 1))),
            #     num_parallel_calls=tf.data.AUTOTUNE
            # )
            if i == 0:
                good_ds = good_ds_part
            else:
                good_ds = good_ds.concatenate(good_ds_part)

        # Load BAD images (label = 1)
        for i, path in enumerate(bad_paths):
            bad_ds_part = self.get_data(path, label=1.0, batch_size=batch_size, img_size=img_size)
            # bad_ds_part = bad_ds_part.map(
            #     lambda x: (x, tf.ones((tf.shape(x)[0], 1))),
            #     num_parallel_calls=tf.data.AUTOTUNE
            # )
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
        

        # Train / validation split
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        val_size = int(dataset_size * val_split)

        val_ds = dataset.take(val_size)
        train_ds = dataset.skip(val_size)

        if(augmentation):
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.028),
            ])

            def augment(image, label):
                return data_augmentation(image, training=True), label

            # Create augmented copy
            augmented_ds = train_ds.map(
                augment,
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # Double dataset size
            train_ds = train_ds.concatenate(augmented_ds)

        return train_ds, val_ds

    #  Old and wrong???
    # def get_data(self,path_to_data, batch_size, img_size = (300,300)):
    #     return tf.keras.utils.image_dataset_from_directory(
    #         path_to_data,
    #         image_size=img_size,
    #         batch_size=batch_size,
    #         label_mode=None,
    #         shuffle=True
    #     )

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
            mod = bm(model[1], **model[2])
            mod.preprocess(model[3], model[4], model[5])
            history = mod.train(**model[2])
            self.hists.append([model[0], history])

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




    