from models.signature import SignatureClassifier
from models.forgery import ForgeryDetector
from models.preprocessing import DataManager


class CNN_Siamese_Pipeline:
    def __init__(self, cnn_classifier: SignatureClassifier = None, siamese_verifier: ForgeryDetector = None):
        self.cnn_classifier = cnn_classifier if cnn_classifier is not None else SignatureClassifier(from_pretrained=True)
        self.siamese_verifier = siamese_verifier if siamese_verifier is not None else ForgeryDetector()
        self.data_manager = DataManager()

    def run_single(self, img, verbose=True):
        img = self.data_manager.preprocess_single(img)

        cnn_output = self.cnn_classifier.predict(img)

        siamese_output = self.siamese_verifier.predict_single(img, cnn_output)

        if verbose:
            print(f"This img belongs to class :{cnn_output} and is {'Verified' if siamese_output else 'Not Verified'}")

        return cnn_output, siamese_output
