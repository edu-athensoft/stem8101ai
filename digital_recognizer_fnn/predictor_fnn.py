"""
predict API for FNN model
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from digital_recognizer_fnn.model_fnn import FFNN
from keras.datasets import mnist
import matplotlib.image as img
from digital_recognizer_fnn import settings
import json


def load_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(len(train_X), 784)
    test_X = test_X.reshape(len(test_X), 784)

    train_X = train_X / 255.0
    test_X = test_X / 255.0
    train_y = pd.get_dummies(train_y).values
    test_y = pd.get_dummies(test_y).values
    return train_X, test_X, train_y, test_y


def predict_fnn(test_X, n_in, n_out, hidden_layers):
    pred_proba, pred_label = nn.predict(test_X=test_X)
    return pred_proba, pred_label


def predict_single_img(my_img, n_in, n_out, hidden_layers):
    image = img.imread(str(my_img))
    image = image / 255.0
    image = image.reshape(1, settings.n_in)
    nn = FFNN(n_in=settings.n_in, n_out=settings.n_out, hidden_layers=settings.hidden_layers, reg_type='multiclass',
              optimizer=tf.train.AdamOptimizer)
    pred_proba, pred_label = nn.predict(test_X=image)
    proba = np.max(pred_proba)
    # print(proba, pred_label)
    res = {'predProbability': str(proba), 'predNumber': str(pred_label[0])}
    json_res = json.dumps(res)
    return json_res


if __name__ == '__main__':
    my_img = settings.saved_imgs + 'img_356.jpg'
    # pred_proba, pred_label = predict_single_img(my_img, settings.n_in, settings.n_out, settings.hidden_layers)
    res = predict_single_img(my_img, settings.n_in, settings.n_out, settings.hidden_layers)
    print(json.loads(res))
    # proba = np.max(pred_proba)
    # print("I recognize this image is {} with probability {}%".format(pred_label[0], proba*100))

# train_X, test_X, train_y, test_y = load_data()
# n_in = train_X.shape[1]
# n_out = train_y.shape[1]
# print(n_in, n_out)
# hidden_layers = [200, 300]
# pred_proba, pred_label = predict_fnn(test_X, n_in, n_out, hidden_layers)
# print(pred_proba[:5], pred_label[:5])
