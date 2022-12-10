import tensorflow as tf
import keras

PRETRAINED_DIR = "./models/pretrained/"


class SignatureClassifier:

    def __init__(self, from_pretrained=False):
        self.model: keras.Sequential = self.__load_pretrained() if from_pretrained else self.__fit_new()

    def __load_pretrained(self):
        try:
            model = keras.models.load_model(PRETRAINED_DIR + "best_signature.hdf5")
            print("Loaded Model:")
            model.summary()
        except FileNotFoundError:
            print("Please add the corresponding hdf5 file to 'models/pretrained/'")
        return model

    def __fit_new(self):
        pass


