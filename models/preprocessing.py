import tensorflow as tf
from keras.utils import image_dataset_from_directory
import numpy as np
import os
from warnings import filterwarnings


ROOT_DIR = "data/train_test_dataset"
TRAIN_PATH = os.path.abspath(ROOT_DIR + "/train")
TEST_PATH = os.path.abspath(ROOT_DIR + "/test")


class DataManager:
    def __init__(self):
        self.data: tf.data.Dataset = self.__read()

    def __read(self):
        return image_dataset_from_directory(TRAIN_PATH)
