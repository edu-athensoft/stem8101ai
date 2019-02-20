"""
this is module is to demo how to use tensorflow to build basic NN from the begining
Author: Yimin Nie
Correspondence: ymnie888@gmail.com
All copyright reserved

SITEM/Athensoft
"""
from keras.datasets import mnist
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from digital_recognizer_fnn import settings
from sklearn.model_selection import train_test_split


class HiddenLayer(object):

    def __init__(self, n_in, n_out, inputs, scope, activation=None, W=None, b=None):
        self.n_in = n_in
        self.n_out = n_out
        self.inputs = inputs
        self.scope = scope

        if W is None:
            with tf.variable_scope('weight_' + self.scope):
                W_init = tf.truncated_normal(
                    shape=[self.n_in, self.n_out],
                    stddev=1e-5,
                )
                self.W = tf.get_variable(name='weight_' + self.scope, initializer=W_init, dtype=tf.float32)

        if b is None:
            with tf.variable_scope('bias_' + self.scope):
                b_init = tf.constant(0.0, shape=[self.n_out])
                self.b = tf.get_variable(name='bias_' + self.scope, initializer=b_init, dtype=tf.float32)

        self.output = tf.matmul(self.inputs, self.W) + self.b
        if activation:
            self.output = activation(self.output)


class FFNN(object):

    def __init__(self, n_in, n_out, hidden_layers, reg_type=None, optimizer=None):

        self.n_in = n_in  # first layer input from external data
        self.n_out = n_out  # last layer output
        self.hidden_layers = hidden_layers  # hidden layers with sizes
        self.layers = []
        self.Ws = []
        self.bs = []
        # regression type: regular regression, binary classification or multi-class classification
        self.reg_type = reg_type
        # optimizer we will use
        self.optimizer = optimizer

        # input data as placeholder, tensorflow create placeholder to feed data dynamically
        self.x = tf.placeholder(name="x", shape=[None, self.n_in], dtype=tf.float32)
        self.y = tf.placeholder(name='y', shape=[None, self.n_out], dtype=tf.float32)

        n_layers = len(self.hidden_layers)
        assert n_layers != 0

        for i in range(n_layers):  # stacking layers

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.layers[-1].output

            if i == 0:
                layer_size = n_in
            else:
                layer_size = self.hidden_layers[i - 1]

            hidden_layer = HiddenLayer(n_in=layer_size, n_out=self.hidden_layers[i], inputs=layer_input,
                                       scope='hidden_layer_' + str(i), activation=tf.nn.relu)
            self.layers.append(hidden_layer)
            self.Ws.append(hidden_layer.W)
            self.bs.append(hidden_layer.b)

        with tf.variable_scope("last_layer", reuse=tf.AUTO_REUSE):

            self.W = tf.get_variable(
                name="W",
                initializer=tf.truncated_normal(shape=(self.hidden_layers[-1], self.n_out)),
                dtype=tf.float32,
            )
            self.b = tf.get_variable(
                name='b',
                initializer=tf.constant(0.0, shape=[self.n_out]),
                dtype=tf.float32
            )
        self.Ws.append(self.W)
        self.bs.append(self.b)
        self.output = tf.matmul(self.layers[-1].output, self.W) + self.b
        self.optimizer = optimizer

        self.sess = tf.InteractiveSession()  # open session for graph computation

        save_dir = "checkpoints/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_path = os.path.join(save_dir, 'model.ckpt')

        # save the model
        self.saver = tf.train.Saver()

        if self.reg_type == "multiclass":
            self.output = tf.nn.softmax(self.output)
        if self.reg_type == "binary":
            self.output = tf.nn.sigmoid(self.output)


    def get_loss_function(self):
        cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.output), reduction_indices=1))
        if self.reg_type == "binary":
            cost = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(self.output) + (1 - self.y) * tf.log(1 - self.output),
                                                 reduction_indices=1))
        if self.reg_type == "multiclass":
            cost = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(self.output), reduction_indices=1))
        return cost


    def train(self, train_X, train_y, valid_X, valid_y):

        cost = self.get_loss_function()
        optimizer = self.optimizer(learning_rate=settings.learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        best_validation_score = np.inf
        iteration = 0

        for epoch in range(settings.epochs):
            print("...processing {}th epoch".format(epoch + 1))
            n_batches = int(train_X.shape[0] / settings.batch_size)
            avg_train_cost = 0.0

            for batch_idx in range(n_batches):
                iteration += 1
                input_batch = train_X[batch_idx * settings.batch_size:(batch_idx + 1) * settings.batch_size]
                target_batch = train_y[batch_idx * settings.batch_size:(batch_idx + 1) * settings.batch_size]
                _, c_train = self.sess.run([optimizer, cost], feed_dict={self.x: input_batch, self.y: target_batch})

                avg_train_cost += c_train / n_batches

                # evaluate the performance using validation set in every 100 steps
                if iteration % settings.eval_valid_steps == 0:
                    c_validation = self.sess.run([cost], feed_dict={self.x: valid_X, self.y: valid_y})
                    if c_validation[0] < best_validation_score:
                        best_validation_score = c_validation[0]
                        best_iter = iter
                        self.saver.save(self.sess, self.save_path)
                        print("Iter: {}, train_cost: {}, validation cost: {}".format(best_iter, c_train,
                                                                                     best_validation_score))

            if (epoch + 1) % settings.eval_epoch_steps == 0:
                print("...*** epoch: {}, avg cost: {}".format(epoch + 1, avg_train_cost))


    def predict_test(self, test_X, test_y):
        # load the best model by opening a new session
        sess = tf.Session()
        saver = tf.train.import_meta_graph("checkpoints/model.ckpt.meta")
        # restore the best model from checkpoint
        saver.restore(sess, "checkpoints/model.ckpt")

        pred_y = self.output
        pred = tf.argmax(pred_y, axis=1)
        correct_pred = tf.equal(pred, tf.argmax(self.y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        p_y, p, corr_pred, accu = sess.run([pred_y, pred, correct_pred, accuracy],
                                           feed_dict={self.x: test_X, self.y: test_y})

        return p_y, p, corr_pred, accu


    def predict(self, test_X):
        # load the best model by opening a new session
        sess = tf.Session()
        saver = tf.train.import_meta_graph("checkpoints/model.ckpt.meta")
        # restore the best model from checkpoint
        saver.restore(sess, "checkpoints/model.ckpt")

        predict_proba = self.output
        predict_label = tf.argmax(predict_proba, axis=1)

        pred_proba, pred_label = sess.run([predict_proba, predict_label], feed_dict={self.x: test_X})

        return pred_proba, pred_label


def main():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(len(train_X), 784)
    test_X = test_X.reshape(len(test_X), 784)

    train_X = train_X / 255.0
    test_X = test_X / 255.0
    train_y = pd.get_dummies(train_y).values
    test_y = pd.get_dummies(test_y).values
    n_in = train_X.shape[1]
    n_out = train_y.shape[1]
    print(n_in, n_out)
    hidden_layers = [200, 300]

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, shuffle=True, random_state=2019,
                                                          test_size=0.2)
    print(train_X.shape, valid_X.shape, train_y.shape, valid_y.shape)

    nn = FFNN(n_in=n_in, n_out=n_out, hidden_layers=hidden_layers, reg_type='multiclass',
              optimizer=tf.train.AdamOptimizer)
    nn.train(train_X, train_y, valid_X, valid_y)
    p_y, p, corr_pred, accu = nn.predict(test_X, test_y)
    print(accu)


if __name__ == '__main__':
    main()
