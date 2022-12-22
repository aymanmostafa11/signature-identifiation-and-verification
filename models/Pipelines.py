from models.signature import SignatureClassifier
from models.verification import SignatureVerifier
from models.preprocessing import DataManager
from models.preprocessing import Preprocessor


class CNN_encoder_Pipeline: # NOQA
    def __init__(self, cnn_classifier: SignatureClassifier = None, verifier: SignatureVerifier = None):
        self.cnn_classifier = cnn_classifier if cnn_classifier is not None else SignatureClassifier(from_pretrained=True)
        self.verifier = verifier if verifier is not None else SignatureVerifier()
        self.data_manager = DataManager()

    def run_single(self, img, verbose=True):

        cnn_img = Preprocessor.preprocess_single(img, Preprocessor.MODEL_CLASSIFIER)
        cnn_output = self.cnn_classifier.predict(cnn_img, as_class_name=True)

        encoder_img = Preprocessor.preprocess_single(img, Preprocessor.MODEL_VERIFIER)
        encoder_output = self.verifier.predict_single(encoder_img, cnn_output)

        if verbose:
            print(f"This img belongs to class :{cnn_output} and is {'Verified' if encoder_output else 'Not Verified'}")

        return cnn_output, encoder_output


