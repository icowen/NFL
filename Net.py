import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import backend as K

np.set_printoptions(suppress=True)


class Net:
    random.seed(1)
    np.random.seed(2)
    tf.random.set_seed(3)
    tf.keras.backend.set_floatx('float64')

    def __init__(self,
                 x_train,
                 y_train,
                 number_of_epochs=10,
                 batch_size=100,
                 load_filename=None):
        self.model = tf.keras.Sequential()
        self.x_train = x_train
        self.y_train = y_train
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.load_filename = load_filename
        if self.load_filename:
            self.load_model()
        else:
            self.set_up_model()

    def set_up_model(self):
        num_of_input_neurons = len(self.x_train[0])
        num_of_output_neurons = len(self.y_train[0])
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(num_of_input_neurons,
                                             activation=tf.nn.sigmoid))
        self.model.add(tf.keras.layers.Dense(num_of_output_neurons,
                                             activation=tf.nn.sigmoid))
        self.model.compile(optimizer='adam',
                           loss=crps_loss,
                           metrics=['accuracy'])

    def train(self):
        self.model.fit(self.x_train,
                       self.y_train,
                       epochs=self.number_of_epochs,
                       batch_size=self.batch_size)
        date = datetime.now().strftime("%m-%d-%y_%H_%M_%S")
        # self.model.save(f'net_trained_with_1000_on_{date}.h5')

    def predict(self, x_input):
        predicted = self.model.predict(x_input)
        return predicted

    def load_model(self):
        self.model = tf.keras.models.load_model(self.load_filename)
        self.model.compile(optimizer='adam',
                           loss=crps_loss,
                           metrics=['accuracy'])


@tf.function
def crps_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    # tf.print('y_true: ', y_true, summarize=-1, output_stream=sys.stderr)
    y_pred = K.cast(y_pred, dtype='float32')
    # tf.print('y_pred: ', y_pred, summarize=-1, output_stream=sys.stderr)
    yards = K.arange(-99, 100, dtype='float32')
    # tf.print('yards: ', yards, summarize=-1, output_stream=sys.stderr)
    ret = K.switch(yards >= y_true, y_pred - 1, y_pred)
    # tf.print('ret: ', ret, summarize=-1, output_stream=sys.stderr)
    ret = K.square(ret)
    per_play_loss = K.sum(ret, axis=1)
    # tf.print('per_play_loss: ', per_play_loss)
    total_loss = K.sum(per_play_loss) / 199
    # tf.print('total_loss: ', total_loss)
    return total_loss
