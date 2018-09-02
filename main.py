from mnist import MNIST
import matplotlib.pyplot as plt
mndata = MNIST("Z:\data")
tr_images, tr_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
img_shape = (28, 28)
import numpy as np

for i in range(0, len(test_images)):
    test_images[i] = np.array(test_images[i]) / 255

for i in range(0, len(tr_images)):
    tr_images[i] = np.array(tr_images[i]) / 255
def relu(x):
    return np.maximum(x, 0)


def nn_calculate(img):
    resp = list(range(0, 10))
    for i in range(0, 10):
        r = w[:, i] * img
        r = relu(np.sum(r) + b[i])
        resp[i] = r
    return np.argmax(resp)


w = (2 * np.random.rand(10, 784) - 1) / 10
b = (2 * np.random.rand(10) - 1) / 10

for n in range(len(tr_images)):
    img = tr_images[n]
    cls = tr_labels[n]
    # forward propagation
    resp = np.zeros(10, dtype=np.float32)
    for i in range(0, 10):
        r = w[i] * img
        r = relu(np.sum(r) + b[i])
        resp[i] = r

    resp_cls = np.argmax(resp)
    resp = np.zeros(10, dtype=np.float32)
    resp[resp_cls] = 1.0

    # back propagation
    true_resp = np.zeros(10, dtype=np.float32)
    true_resp[cls] = 1.0

    error = resp - true_resp

    delta = error * ((resp >= 0) * np.ones(10))
    for i in range(0, 10):
        w[i] -= np.dot(img, delta[i])
        b[i] -= delta[i]


def nn_calculate(img):
    resp = list(range(0, 10))
    for i in range(0, 10):
        r = w[i] * img
        r = np.maximum(np.sum(r) + b[i], 0)  # relu
        resp[i] = r
    return np.argmax(resp)


total = len(test_images)
valid = 0
invalid = []

for i in range(0, total):
    img = test_images[i]
    predicted = nn_calculate(img)
    true = test_labels[i]
    if predicted == true:
        valid = valid + 1
    else:
        invalid.append({"image": img, "predicted": predicted, "true": true})

print("accuracy {}".format(valid / total))