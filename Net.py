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
                 x_valid,
                 y_valid,
                 number_of_epochs=10,
                 batch_size=100,
                 load_filename=None):
        self.model = tf.keras.Sequential()
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
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
        self.model.add(tf.keras.layers.Flatten(input_shape=(num_of_input_neurons,)))
        self.model.add(tf.keras.layers.Dense(num_of_input_neurons,
                                             activation=tf.nn.sigmoid))
        self.model.add(tf.keras.layers.Dense(num_of_output_neurons,
                                             activation=tf.nn.sigmoid))
        self.model.compile(optimizer='adam',
                           loss=crps_loss,
                           metrics=['accuracy'])

    def train(self):
        date = datetime.now().strftime("%m-%d-%y_%H_%M_%S")
        path = f'net_configurations/crps_net_trained_with_10000_on_{date}.h5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(path,
                                                        monitor='loss',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        save_freq=100 * len(self.x_train))
        callbacks_list = [checkpoint]

        self.model.fit(self.x_train,
                       self.y_train,
                       validation_data=(self.x_valid, self.y_valid),
                       epochs=self.number_of_epochs,
                       batch_size=self.batch_size,
                       callbacks=callbacks_list)

    def predict(self, x_input):
        predicted = self.model.predict(x_input)
        return predicted

    def load_model(self):
        self.model = tf.keras.models.load_model(self.load_filename, custom_objects={'crps_loss': crps_loss})
        self.model.compile(optimizer='adam',
                           loss=crps_loss,
                           metrics=['accuracy'])


@tf.function
def crps_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    ret = K.switch(y_true >= 1, y_pred - 1, y_pred)
    ret = K.square(ret)
    per_play_loss = K.sum(ret, axis=1)
    total_loss = K.mean(per_play_loss)
    return total_loss
