from mnist_loader import MNISTLoader
import mvp_nn as nn

loader = MNISTLoader("data")
train_images, train_labels, test_images, test_labels = loader.load()

nn.train(train_images, train_labels, epochs=1000, batch_size=32, lr=0.01)

print("Test accuracy:", nn.accuracy(test_images, test_labels))

# at 1000 epochs, it reaches 97.5% accuracy :D