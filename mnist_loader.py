import numpy as np
from mnist import MNIST

def one_hot(labels, num_classes=10):
    """Convert a list of labels of the number-images to *one-hot encoded* format.
    **Example:** 3 produces `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
    Args:
        labels: List or array of integer labels.
        num_classes: Total number of classes."""
    labels = np.array(labels, dtype=int)
    out = np.zeros((labels.size, num_classes), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1.0
    return out

class MNISTLoader:
    def __init__(self, folder="data"):
        self.folder = folder
        self.mndata = MNIST(folder)

    def load(self):
        train_images, train_labels = self.mndata.load_training()
        test_images, test_labels   = self.mndata.load_testing()

        train_images = np.array(train_images, dtype=np.float32) / 255.0 #improve brightness contrast DOES NOT WORK OTHERWISE
        train_labels = one_hot(train_labels)

        test_images = np.array(test_images, dtype=np.float32) / 255.0
        test_labels = one_hot(test_labels)

        return train_images, train_labels, test_images, test_labels
