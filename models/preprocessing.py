import os
import tensorflow as tf
from keras.utils import image_dataset_from_directory
import numpy as np


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

    def __read(self, subset="all"):
        """
        :param subset: one of ["train", "valid", "test", "all"]
        """
        if subset in ["train", "all"]:
            self.data = image_dataset_from_directory(TRAIN_PATH, image_size=IMG_SIZE)
            self.train_data = image_dataset_from_directory(TRAIN_PATH,
                                                           subset='training',
                                                           validation_split=VALIDATION_SPLIT,
                                                           seed=SPLIT_SEED,
                                                           image_size=IMG_SIZE,
                                                           batch_size=BATCH_SIZE,
                                                           color_mode=COLOR_MODE)
        if subset in ["valid", "all"]:
            self.valid_data = image_dataset_from_directory(TRAIN_PATH,
                                                           subset='validation',
                                                           validation_split=VALIDATION_SPLIT,
                                                           seed=SPLIT_SEED,
                                                           image_size=IMG_SIZE,
                                                           batch_size=BATCH_SIZE,
                                                           color_mode=COLOR_MODE)

        if subset in ["test", "all"]:
            self.test_data = image_dataset_from_directory(TEST_PATH,
                                                          seed=42,
                                                          image_size=IMG_SIZE,
                                                          batch_size=BATCH_SIZE,
                                                          color_mode=COLOR_MODE)

    def load_data(self, subset="all"):
        """
        load subsets of data or all of them
        :param subset: one of (train, test, validation) to load specific dataset, if "all" is provided will load all data
        """
        self.__read(subset)

    @staticmethod
    def read_image(path: str, as_array=True):
        """
        Reads an image from disk and returns is
        :param path: absolute path to image
        :param as_array: whether to return the image as a numpy.ndarray or a PIL image
        :return: PIL image or numpy.ndarray
        """
        img = tf.keras.utils.load_img(path)
        if not as_array:
            return img

        img = tf.keras.utils.img_to_array(img)
        return img

    @staticmethod
    def read_bulk(path: str):
        """
        Read bulk of images from a directory
        :param path: absolute path to directory
        :return: tf.data.Dataset
        """
        assert os.path.isdir(path), f"Please provide a path to a directory containing images, provided: {path}"
        data = image_dataset_from_directory(path,
                                            shuffle=False,
                                            label_mode=None,
                                            seed=42,
                                            image_size=IMG_SIZE,
                                            batch_size=BATCH_SIZE,
                                            color_mode=COLOR_MODE)

        assert data.cardinality().numpy() > 0, "No data found in the provided directory"

        return data


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
        :return: preprocessed image
        """
        assert model_type in Preprocessor.__model_types , f"Choosen model type not available, must be in " \
                                                          f"{Preprocessor.__model_types}"

        img = tf.image.resize(img, IMG_SIZE)

        if model_type == Preprocessor.MODEL_CLASSIFIER:
            img = tf.image.rgb_to_grayscale(img)

        img = np.expand_dims(img, axis=0)
        return img

    @staticmethod
    def preprocess_bulk(data: tf.data.Dataset, model_type: str, external_data=False):
        """
        Preprocess a dataset for one of the available models
        :param data: a tensorflow dataset to be process
        :param model_type: apply which model preprocessing, one of "classifier", "verifier", "detector"
        :param external_data : a flag whether this data is from the datasets already processed by DataManager
        (will resize images if this flag is provided)
        :return: preprocessed data
        """
        assert model_type in Preprocessor.__model_types, f"Choosen model type not available, must be in " \
                                                         f"{Preprocessor.__model_types}"

        if external_data:
            data = data.map(Preprocessor.__resize)

        if model_type == Preprocessor.MODEL_CLASSIFIER:
            data = data.map(Preprocessor.__to_grayscale)

        return data

    # tf.dataset specific functions
    @staticmethod
    def __to_grayscale(img, label=None):
        img = tf.image.rgb_to_grayscale(img)
        return img, label if label else img

    @staticmethod
    def __resize(img, label):
        img = tf.image.resize(img, IMG_SIZE)
        return img, label if label else img

