# image augmentations, brightness, rotation, saturation
import tensorflow as tf
from tensorflow.keras import layers
import math

def flip(image, flip):
    if flip == "horizontal":
        return tf.image.flip_left_right(image)
    elif flip == "vertical":
        return tf.image.flip_up_down(image)
    elif flip == "both":
        return tf.image.flip_up_down(tf.image.flip_left_right(image))
    else:
        return image

def rotate(image, degrees):
    radians = degrees * (math.pi / 180)
    # random rotation
    
    return tf.image.rotate(image, radians)

def augment_image(image, path_save=None):
    flip_options = ["horizontal", "vertical", "both"]
    num_rotations = 3

    for flip_type in flip_options:
        for i in range(num_rotations):
            angle = tf.random.uniform(
                shape=(),
                minval=-10.0,
                maxval=10.0
            )

            augmented_image = rotate(image, angle)
            augmented_image = flip(augmented_image, flip_type)

            tf.keras.utils.save_img(
                f"{path_save}/aug_{flip_type}_{i}.jpg",
                augmented_image
            )

def augment_rand_once(image): # if image is path to image file, then uncomment code below
    # image = tf.io.read_file("cat.jpg")
    # image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    data_augmentation = tf.keras.Sequential([ layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.028) ])
    return data_augmentation(image, training=True)


if __name__ == "__main__":
    image = tf.io.read_file("C:\\Programmering\\Masters\\DL-Mechanical-Defects\\dataset_creation\\processed_images\\initial\\bad\\1\\8.jpg")
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    augmented = augment_rand_once(image)
    tf.keras.utils.save_img("C:\\Programmering\\Masters\\DL-Mechanical-Defects\\dataset_creation\\augmented1.jpg", augmented)