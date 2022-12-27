from models.preprocessing import DataManager
from tensorflow import keras
import pickle
import numpy as np
import cv2
import os

IMG_SHAPE = (256, 256)
ROOT = os.path.abspath(os.path.curdir)


class SignatureVerifier:
    threshold = 0.7
    def __init__(self):
        self.encoder: keras.models.Model = keras.models.load_model(ROOT+'\saved models\saved-encoder.h5', compile=False)
        self.database = pickle.load(open(ROOT+'\saved models\saved_embeddings.p', 'rb'))

    # still experimenting with the threshold value
    def predict_single(self, img: np.ndarray, id: str, threshold: float = threshold):
        """
        :param img: image
        :param id: The ID of the signature i.e: 'PersonA'
        :param threshold: threshold to use for classification
        :return: False if signature is forged, True if genuine
        """
        anchor_embedding = self.database[id]
        #print(img_path)
        #img = self.__read_image(image_path = img_path)
        img_embedding = self.encoder.predict(img)
        distance = np.linalg.norm(np.subtract(anchor_embedding, img_embedding))
        prediction = np.where(distance < threshold, 1, 0)

        return prediction == 1

    def __read_image(self, image_path):
        """
        :param image_path: Absolute path for image
        :return: read image with shape (1, 256, 256, 3)
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SHAPE)
        image = np.expand_dims(image, 0)

        return image

    def predict_bulk(self, images, IDs: list, threshold: float = threshold):
        """
        :param images: images
        :param IDs: The list of IDs of the signatures'
        :param threshold: threshold to use for classification
        :return: list of predictions, False if signature is forged, True if genuine
        """
        predictions = []

        for img_no, img in enumerate(images.unbatch()):
            anchor_embedding = self.database[IDs[img_no]]
            img_embedding = self.encoder.predict(np.expand_dims(img, axis=0), verbose=0)  # TODO: implement a cleaner sol
            distance = np.linalg.norm(np.subtract(anchor_embedding, img_embedding))
            pred = np.where(distance < threshold, 1, 0)
            predictions.append(pred == 1)

        return predictions    

