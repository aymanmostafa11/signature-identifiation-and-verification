from models.preprocessing import DataManager
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_manager = DataManager()
    plt.imshow(data_manager.data.as_numpy_iterator().next()[0][0])
    plt.show()
