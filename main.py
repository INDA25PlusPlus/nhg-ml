from mnist_loader import MNISTLoader
import numpy as np

loader = MNISTLoader("data")
train_images, train_labels, test_images, test_labels = loader.load()

print("Min pixel:", train_images.min())
print("Max pixel:", train_images.max())
print("Mean pixel:", train_images.mean())
print("Std pixel:", train_images.std())

""" Expected Output:
Min pixel: 0.0
Max pixel: 1.0
Mean pixel: 0.130...
Std pixel: 0.308...
"""

print("Train images:", train_images.shape)
print("Train labels:", train_labels.shape)
print("Test images:", test_images.shape)
print("Test labels:", test_labels.shape)

""" Expected Output:
Train images: (60000, 784)
Train labels: (60000, 10)
Test images: (10000, 784)
Test labels: (10000, 10) """

def print_digit(img):
    img = img.reshape(28, 28)
    chars = " .:-=+*#%@"
    for row in img:
        line = "".join(chars[int(pixel * 9)] for pixel in row)
        print(line)

# label 5 looks WHACK but i guess it's just like that
for i in range(5): # not ordered ?
    print("Label:", train_labels[i].argmax())
    print_digit(train_images[i])
    print("---")

