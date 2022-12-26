from models.signature import SignatureClassifier
from models.verification import SignatureVerifier
from models.preprocessing import DataManager
from models.preprocessing import Preprocessor

import warnings

class CNN_encoder_Pipeline: # NOQA
    def __init__(self, cnn_classifier: SignatureClassifier = None, verifier: SignatureVerifier = None):
        self.cnn_classifier = cnn_classifier if cnn_classifier is not None else SignatureClassifier(from_pretrained=True)
        self.verifier = verifier if verifier is not None else SignatureVerifier()
        self.data_manager = DataManager()

    def run_single(self, path, verbose=True):
        img = DataManager.read_image(path)

        if verbose:
            print("Classifying Image..")
        cnn_img = Preprocessor.preprocess_single(img, Preprocessor.MODEL_CLASSIFIER)
        cnn_output = self.cnn_classifier.predict(cnn_img, as_class_name=True)

        if verbose:
            print("Verifying Image..")
        encoder_img = Preprocessor.preprocess_single(img, Preprocessor.MODEL_VERIFIER)
        encoder_output = self.verifier.predict_single(encoder_img, cnn_output)

        if verbose:
            print(f"This image belongs to class : {cnn_output} and is {'Verified' if encoder_output else 'Not Verified'}")

        return cnn_output, encoder_output

    def run_bulk(self, path, verbose=True):
        """
        Predict on unlabeled data
        :param path: abs path to a folder containing images to be classified and verified
        :param verbose: print messages
        """
        pass

    def evaluate(self, subset="test"):
        """
        Evaluate the pipeline on predefined data (train or test set)
        :param subset: one of ["train", "test", "other"]
        """
        assert subset in ["train", "test"], "Please provide a valid subset to evaluate the data on ('train' or 'test')"

        self.data_manager.load_data(subset)

        data = self.data_manager.data if subset == "train" else self.data_manager.test_data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            print("\n#### Classifier Evaluation ####")
            cnn_data = Preprocessor.preprocess_bulk(data, Preprocessor.MODEL_CLASSIFIER, external_data=False)
            self.cnn_classifier.evaluate(cnn_data)
            print('#' * 30)





