import gc

import tensorflow as tf
from models.baseModel import BaseModel as bm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import os

class MLPipeline:
    def __init__(self, path_to_data):
        print("Initializing ML Pipeline")
        self.path_to_data = path_to_data
        self.models = []

    def add_model(self, model, image_type, focus, preprocess_input, **params):
        # Create model instances for each image_type and focus combination
        for t in image_type:
            if params.get("two_inputs", False):
                # for a two‑input network we want one entry containing
                # *all* focus folders – the pairing logic lives in
                # create_full_dataset/create_image_pairs.
                good_paths = [self.path_to_data + f"/processed_images/{t}/good/{f}" for f in focus]
                bad_paths = []
                for f in focus:
                    bad_paths.append(self.path_to_data + f"/processed_images/{t}/bad/{f}")
                    bad_paths.append(self.path_to_data + f"/processed_images/{t}/bad_but_looks_good/{f}")
                name = f"{model().name}_{t}_{'_'.join(focus)}"
                self.models.append({"name": name,
                                    "model_cls": model,
                                    "params": params,
                                    "good_paths": good_paths,
                                    "bad_paths": bad_paths,
                                    "preprocess": preprocess_input,})

            else:
                for f in focus:
                    good_paths = [self.path_to_data + f"/processed_images/{t}/good/{f}"]
                    bad_paths = [self.path_to_data + f"/processed_images/{t}/bad/{f}"]
                    self.models.append({"name": f"{model().name}_{t}_{f}",
                                        "model_cls": model,
                                        "params": params,
                                        "good_paths": good_paths,
                                        "bad_paths": bad_paths,
                                        "preprocess": preprocess_input,})

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
                        img_size=(300,300), data_limit=500, two_inputs=False):
        
        per_class_limit = data_limit

        if two_inputs:
        # For two-input model: pair images from focus 1 and focus 2
            good_pairs = self.create_image_pairs(good_paths, img_size, label=0.0)
            bad_pairs = self.create_image_pairs(bad_paths, img_size, label=1.0)
            
            good_pairs = good_pairs.take(per_class_limit)
            bad_pairs = bad_pairs.take(per_class_limit)
            
            ds = good_pairs.concatenate(bad_pairs)
        else:

            good_ds = None
            bad_ds = None

            for i, path in enumerate(good_paths):
                ds_part = self.get_data(path, label=0.0, batch_size=1, img_size=img_size)
                ds_part = ds_part.unbatch()
                good_ds = ds_part if good_ds is None else good_ds.concatenate(ds_part)

            for i, path in enumerate(bad_paths):
                ds_part = self.get_data(path, label=1.0, batch_size=1, img_size=img_size)
                ds_part = ds_part.unbatch()
                bad_ds = ds_part if bad_ds is None else bad_ds.concatenate(ds_part)

            good_ds = good_ds.take(per_class_limit)
            bad_ds = bad_ds.take(per_class_limit)

            ds = good_ds.concatenate(bad_ds)

        return ds
    
    def create_image_pairs(self, paths, img_size=(300, 300), label=None):
    
        import os
        
        images_by_focus = {}
        
        for path in paths:
            if not os.path.exists(path):
                continue
                
            focus_num = os.path.basename(path)  # Extract "1" or "2"
            images = {}
            
            for img_file in os.listdir(path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(path, img_file)
                    try:
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        images[img_file] = img_array
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
            
            if focus_num not in images_by_focus:
                images_by_focus[focus_num] = {}
            
            # Merge images from this path into the focus dictionary
            images_by_focus[focus_num].update(images)
        
        # Pair images by filename across focuses
        paired_images = []
        focus_keys = sorted(images_by_focus.keys())
        
        if len(focus_keys) >= 2:
            focus1_images = images_by_focus[focus_keys[0]]
            focus2_images = images_by_focus[focus_keys[1]]
            
            # Match pairs by filename - only include if both focuses have the image
            common_files = set(focus1_images.keys()) & set(focus2_images.keys())
            
            for fname in sorted(common_files):
                paired_images.append((
                    (focus1_images[fname], focus2_images[fname]),
                    label if label is not None else 1.0
                ))
                # print(f"Paired image: {fname} from focus {focus_keys[0]} and {focus_keys[1]}")

        
        if not paired_images:
            raise ValueError(f"No matching image pairs found in paths: {paths}")
        
        return tf.data.Dataset.from_generator(
            lambda: iter(paired_images),
            output_signature=(
                (tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32)),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        )


    def create_datasets(
        self,
        good_paths,
        bad_paths,
        cross_validation=False,
        k_folds=5,
        fold_index=0,
        **kwargs
    ):
        data_limit = kwargs.get("data_limit")
        val_split = kwargs.get("val_split", 0.2)
        batch_size = kwargs.get("batch_size", 32)
        img_size = kwargs.get("img_size", (300,300))
        augmentation = kwargs.get("augmentation", False)
        two_inputs = kwargs.get("two_inputs", False)


        
        full_ds = self.create_full_dataset(
            good_paths, bad_paths, img_size=img_size, data_limit=data_limit, two_inputs=two_inputs
        )

        # Extract labels for stratification
        samples = list(full_ds)

        if two_inputs:
            # x is a tuple (img1, img2)
            images_a = np.stack([x[0].numpy() for x, _ in samples])
            images_b = np.stack([x[1].numpy() for x, _ in samples])
            labels = np.array([y.numpy() for _, y in samples]).reshape(-1)
            print("Two-input dataset shapes:")
            print(images_a.shape, images_b.shape)
        else:
            images_a = np.stack([x.numpy() for x, _ in samples])
            images_b = None
            labels = np.array([y.numpy() for _, y in samples]).reshape(-1)


        n = len(images_a)
        if cross_validation:
                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                train_idx, val_idx = list(skf.split(np.zeros(n), labels))[fold_index]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
            train_idx, val_idx = next(sss.split(np.zeros(n), labels))

        if two_inputs:
            train_images = (images_a[train_idx], images_b[train_idx])
            val_images = (images_a[val_idx], images_b[val_idx])
        else:
            train_images = images_a[train_idx]
            val_images = images_a[val_idx]


        train_labels = labels[train_idx]
        val_labels   = labels[val_idx]

        del images_a, images_b, labels, samples
        gc.collect()

        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        
        print(len(train_ds), len(val_ds))
        if augmentation:
            aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.028),
            ])

            def augment_fn(x, y):
                # two-input case
                if isinstance(x, (tuple, list)):
                    x1, x2 = x
                    return (aug(x1, training=True), aug(x2, training=True)), y
                # single-input case
                else:
                    return aug(x, training=True), y

            augmented_ds = train_ds.map(
                augment_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )

            train_ds = train_ds.concatenate(augmented_ds)
        train_ds = (train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE))

        val_ds = (val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE))
        print(len(train_ds), len(val_ds))
        return train_ds, val_ds

    def get_data(self, path_to_data, label, batch_size, img_size=(300, 300)):
        ds = tf.keras.utils.image_dataset_from_directory(
            path_to_data,
            image_size=img_size,
            batch_size=batch_size,
            label_mode=None,
            shuffle=False
        )

        return ds.map(
            lambda x: (x, tf.cast(tf.fill([tf.shape(x)[0]], label), tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    def run_pipeline(self):
        self.hists = []
        print("Running Normal Model training")

        for model in self.models:
            print(f"  Model Name: {model['name']}")
            train_ds, val_ds = self.create_datasets(model["good_paths"], model["bad_paths"], **model["params"])

            mod = bm(model["model_cls"], **model["params"])
            
            mod.preprocess(train_ds, val_ds, model["preprocess"])
            history = mod.train(**model["params"])

            self.hists.append([model["name"], history])

            del mod, train_ds, val_ds
            tf.keras.backend.clear_session()
            gc.collect()

            df = pd.DataFrame(history.history)
            df.index.name = "epoch"
            if not os.path.exists("histories"):
                os.makedirs("histories")
            df.to_csv(f"histories/{model['name']}_hist.csv", index=True)

            # Plot history for the model
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Loss for {model["name"]}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Accuracy for {model["name"]}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            # save figure but in folder /plots if it does not exist, create it
            if not os.path.exists("plots"):
                os.makedirs("plots")
            plt.savefig(f"plots/{model['name']}_training_history.png")
            plt.close()


    def run_cross_validation(self, folds=5):
        self.hists = []
        print(f"Running cross-validation with {folds} folds...")

        for model in self.models:
            fold_histories = []
            fold_rows = []
            print(f"  Model Name: {model['name']}")

            for fold in range(folds):
                print(f"    Fold {fold+1}/{folds}")
                train_ds, val_ds = self.create_datasets(model["good_paths"],model["bad_paths"],
                    cross_validation=True, k_folds=folds, fold_index=fold, **model["params"])

                mod = bm(model["model_cls"], **model["params"])
                mod.preprocess(train_ds, val_ds, model["preprocess"])

                history = mod.train(**model["params"])
                fold_histories.append(history)

                del mod
                del train_ds
                del val_ds
                tf.keras.backend.clear_session()
                gc.collect()

                # calc average of metrics and save
                avg_metrics = {metric: float(np.mean(values))
                              for metric, values in history.history.items()}

                avg_metrics["fold"] = fold
                fold_rows.append(avg_metrics)
                tf.keras.backend.clear_session()
                gc.collect()

            df = pd.DataFrame(fold_rows).set_index("fold")

            # mean across folds
            mean_row = df.mean(numeric_only=True)
            mean_row.name = "mean"
            df = pd.concat([df, mean_row.to_frame().T])
            if not os.path.exists("cv_histories"):
                os.makedirs("cv_histories")
            df.to_csv(f"cv_histories/{model['name']}_cv_hist.csv")

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
            # save figure but in folder /plots if it does not exist, create it
            if not os.path.exists("plots"):
                os.makedirs("plots")
            plt.savefig(f"plots/{hist[0]}_training_history.png")

            

            # plt.show()

    def plot_cross_validation_results(self):
        for hist in self.hists:
            fold_histories = hist[1]

            # Samla data från alla folds (kan ha olika längd p.g.a. early stopping)
            val_loss_lists = [h.history['val_loss'] for h in fold_histories]
            val_acc_lists = [h.history['val_accuracy'] for h in fold_histories]

            max_epochs = max(len(values) for values in val_loss_lists)

            val_losses = np.full((len(val_loss_lists), max_epochs), np.nan, dtype=np.float32)
            val_accs = np.full((len(val_acc_lists), max_epochs), np.nan, dtype=np.float32)

            for i, values in enumerate(val_loss_lists):
                val_losses[i, :len(values)] = values

            for i, values in enumerate(val_acc_lists):
                val_accs[i, :len(values)] = values

            # Medelvärde och standardavvikelse per epoch
            avg_val_loss = np.nanmean(val_losses, axis=0)
            std_val_loss = np.nanstd(val_losses, axis=0)

            avg_val_acc = np.nanmean(val_accs, axis=0)
            std_val_acc = np.nanstd(val_accs, axis=0)

            epochs = range(len(avg_val_loss))

            plt.figure(figsize=(12, 4))

            # --- Validation Loss ---
            plt.subplot(1, 2, 1)
            plt.plot(epochs, avg_val_loss, label='Mean Validation Loss')
            plt.fill_between(
                epochs,
                avg_val_loss - std_val_loss,
                avg_val_loss + std_val_loss,
                alpha=0.3,
                label='±1 std'
            )
            plt.title(f'Validation Loss (mean ± std) for {hist[0]}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            # --- Validation Accuracy ---
            plt.subplot(1, 2, 2)
            plt.plot(epochs, avg_val_acc, label='Mean Validation Accuracy')
            plt.fill_between(
                epochs,
                avg_val_acc - std_val_acc,
                avg_val_acc + std_val_acc,
                alpha=0.3,
                label='±1 std'
            )
            plt.title(f'Validation Accuracy (mean ± std) for {hist[0]}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()

            if not os.path.exists("cv_plots"):
                os.makedirs("cv_plots")
            plt.savefig(f"cv_plots/{hist[0]}_cross_validation_results.png")
            plt.close()





    