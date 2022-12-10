from models.preprocessing import DataManager
from models.signature import SignatureClassifier
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_manager = DataManager()
    plt.imshow(data_manager.data.as_numpy_iterator().next()[0][0])
    plt.show()

    sig = SignatureClassifier(from_pretrained=True)

    sig.model.evaluate(data_manager.test_data)

