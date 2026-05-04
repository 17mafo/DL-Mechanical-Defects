# image augmentations, brightness, rotation, saturation
import tensorflow as tf
from tensorflow.keras import layers


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