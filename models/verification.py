from preprocessing import DataManager
from tensorflow import keras
import pickle
import numpy as np
import cv2

IMG_SHAPE = (256,256)
ROOT = 'D:/FCIS/4th\CV\Project\signature-identifiation-and-verification'

class SignatureVerifier:

    def __init__(self):
        self.encoder = keras.models.load_model(ROOT+'\saved models\saved-encoder.h5')
        self.database = pickle.load(open(ROOT+'\saved models\saved_embeddings.p', 'rb'))

    # still experimenting with the threshold value
    def predict_single(self, img_path: str, id: str, threshold: float):
        """
        :param img_path: path for image
        :param id: The ID of the signature i.e: 'PersonA'
        :param threshold: threshold to use for classification
        :return: False if signature is forged, True if genuine
        """
        anchor_embedding = self.database[id]
        print(img_path)
        img = self.__read_image(image_path = img_path)
        img_embedding = self.encoder.predict(img)
        distance = np.linalg.norm(np.subtract(anchor_embedding, img_embedding))
        prediction = np.where(distance<threshold, 1, 0)

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

