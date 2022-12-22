from models.preprocessing import DataManager
from models.signature import SignatureClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
from models.Pipelines import CNN_encoder_Pipeline
from PIL import Image

if __name__ == "__main__":
    # data_manager = DataManager()
    # plt.imshow(data_manager.data.as_numpy_iterator().next()[0][0])
    # plt.show()
    #
    # sig = SignatureClassifier(from_pretrained=True)
    #
    # sig.model.evaluate(data_manager.test_data)
    img = tf.keras.utils.load_img("data/train_test_dataset/test/personD/personD_31.png")
    img = tf.keras.utils.img_to_array(img)

    pipeline = CNN_encoder_Pipeline()
    pipeline.run_single(img)

