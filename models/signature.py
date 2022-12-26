import pickle
import tensorflow as tf
import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import cv2
import numpy as np

PRETRAINED_DIR = "./models/pretrained/"

IMG_SIZE = (256, 256)
COLOR_MODE = "grayscale"

label_map = {0: "personA", 1: "personB", 2: "personC", 3: "personD", 4: "personE"}

class SignatureClassifier:
    def __init__(self, from_pretrained=False):
        self.__model: keras.Sequential = self.__load_pretrained() if from_pretrained else self.__fit_new()

    def __load_pretrained(self):
        try:
            model = keras.models.load_model(PRETRAINED_DIR + "best_signature.hdf5")
            print("Loaded Model:")
            model.summary()

            if model.layers[-1].get_config()["activation"] != "softmax":
                model.add(tf.keras.layers.Softmax())
            return model
        except OSError:
            print("Please add the corresponding hdf5 file to 'models/pretrained/'")
            exit()


    def __fit_new(self):
        pass

    def predict(self, data, as_proba=False, as_class_name=False):
        """
        :param data: an image or batch of images of shape (m,IMG_SIZE[0], IMG_SIZE[1], channels)
        :param as_proba: return predicted probabilities instead of class index
        :return: Single Class number or np.ndarray depicting the probability of each class
        """
        pred = self.__model.predict(data)

        if as_proba:
            return pred

        class_indx = pred.argmax(axis=1)[0]

        if as_class_name:
            return label_map[class_indx]

        return class_indx

    def evaluate(self, test_data):
        return self.__model.evaluate(test_data)


class ClassicalClassifier:
    def __init__(self):
        self.__predictor: RandomForestClassifier = pickle.load(open("models/pretrained/classical_predictor.p", "rb"))
        self.__kmeans: KMeans = pickle.load(open("models/pretrained/classical_clusterer.p", "rb"))
        self.__sift = cv2.SIFT_create()
        self.__no_clusters = 50

    def predict(self, data):
        data = tf.image.resize(data, IMG_SIZE).numpy().astype(np.uint8)
        bow = self.__img_to_bow(data).reshape(-1, 1)
        pred = self.__predictor.predict(bow)
        return pred

    def __extract_sift_features(self, img):
        try:
            kps, desc = self.__sift.detectAndCompute(img, None)
            return desc
        except Exception as e:
            print(f"img: {img.shape}, {e}")

    def __img_to_bow(self, img):
        descriptors = self.__extract_sift_features(img)

        pred = self.__kmeans.predict(descriptors)
        bow = np.histogram(pred, bins=self.__no_clusters)[0]
        return bow
