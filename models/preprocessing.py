import tensorflow as tf
from keras.utils import image_dataset_from_directory
import numpy as np
import os
from warnings import filterwarnings


ROOT_DIR = "data/train_test_dataset"
TRAIN_PATH = os.path.abspath(ROOT_DIR + "/train")
TEST_PATH = os.path.abspath(ROOT_DIR + "/test")

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
COLOR_MODE = "grayscale"
VALIDATION_SPLIT = 0.2
SPLIT_SEED = 42


class DataManager:
    def __init__(self):
        self.data: tf.data.Dataset = None
        self.train_data: tf.data.Dataset = None
        self.valid_data: tf.data.Dataset = None
        self.test_data: tf.data.Dataset = None

    def __read(self):
        # TODO: Fix "data" to support COLOR_MODE (doesn't now because it is used to visualize data)
        self.data = image_dataset_from_directory(TRAIN_PATH, image_size=IMG_SIZE)
        self.train_data = image_dataset_from_directory(TRAIN_PATH,
                                                       subset='training',
                                                       validation_split=VALIDATION_SPLIT,
                                                       seed=SPLIT_SEED,
                                                       image_size=IMG_SIZE,
                                                       batch_size=BATCH_SIZE,
                                                       color_mode=COLOR_MODE)

        self.valid_data = image_dataset_from_directory(TRAIN_PATH,
                                                       subset='validation',
                                                       validation_split=VALIDATION_SPLIT,
                                                       seed=SPLIT_SEED,
                                                       image_size=IMG_SIZE,
                                                       batch_size=BATCH_SIZE,
                                                       color_mode=COLOR_MODE)

        self.test_data = image_dataset_from_directory(TEST_PATH,
                                                      seed=42,
                                                      image_size=IMG_SIZE,
                                                      batch_size=BATCH_SIZE,
                                                      color_mode=COLOR_MODE)

    def load_data(self):
        self.__read()

    def preprocess_single(self, img):
        img = tf.image.resize(img, IMG_SIZE)
        if COLOR_MODE == "grayscale":
            img = tf.image.rgb_to_grayscale(img)

        img = np.expand_dims(img, axis=0)  # Format as a batch

        return img
