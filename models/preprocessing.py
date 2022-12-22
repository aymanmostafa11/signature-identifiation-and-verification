import tensorflow as tf
from keras.utils import image_dataset_from_directory
import numpy as np
import os

ROOT_DIR = "data/train_test_dataset"
TRAIN_PATH = os.path.abspath(ROOT_DIR + "/train")
TEST_PATH = os.path.abspath(ROOT_DIR + "/test")

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
COLOR_MODE = "rgb"
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


class Preprocessor:
    MODEL_CLASSIFIER = "classifier"
    MODEL_VERIFIER = "verifier"
    MODEL_DETECTOR = "detector"

    __model_types = [MODEL_CLASSIFIER, MODEL_VERIFIER, MODEL_CLASSIFIER]
    @staticmethod
    def preprocess_single(img, model_type: str):
        """
        :param img: image as numpy array or any tensorflow compatible format
        :param model_type: apply which model preprocessing, one of "classifier", "verifier", "detector"
        :return:
        """
        assert model_type in Preprocessor.__model_types , f"Choosen model type not available, must be in " \
                                                          f"{Preprocessor.__model_types}"

        img = tf.image.resize(img, IMG_SIZE)

        if model_type == Preprocessor.MODEL_CLASSIFIER:
            img = tf.image.rgb_to_grayscale(img)

        img = np.expand_dims(img, axis=0)
        return img
