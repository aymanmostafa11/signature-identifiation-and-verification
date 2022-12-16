from preprocessing import DataManager
from signature import SignatureClassifier

dataManager = DataManager()
dataManager.load_data()

classifier = SignatureClassifier(from_pretrained=True)

classifier.eveluate(test_data=dataManager.test_data)

