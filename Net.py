import random

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
                 cumsum,
                 number_of_epochs=10,
                 batch_size=100,
                 load_filename=None):
        self.model = tf.keras.Sequential()
        self.x_train = x_train
        self.y_train = y_train
        self.cumsum = cumsum
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
                           loss=crps_loss(self.cumsum),
                           metrics=['accuracy'])

    def train(self):
        # validation_overfitting = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                                           min_delta=1,
        #                                                           patience=50,
        #                                                           verbose=0, mode='min')
        # callbacks_list = [validation_overfitting]

        self.model.fit(self.x_train,
                       self.y_train,
                       validation_split=.2,
                       epochs=self.number_of_epochs,
                       batch_size=self.batch_size)

    def predict(self, x_input):
        predicted = self.model.predict(x_input)
        for input_play, prediction in zip(x_input, predicted):
            yards_2_endzone = int(input_play[-4])
            for i in range(yards_2_endzone + 15, len(prediction)):
                prediction[i] = 1
        predicted = list(map(lambda x: np.pad(x, (84, 0), constant_values=0), predicted))
        return predicted

    def load_model(self):
        self.model = tf.keras.models.load_model(self.load_filename, custom_objects={'crps_loss': crps_loss})
        self.model.compile(optimizer='adam',
                           loss=crps_loss(self.cumsum),
                           metrics=['accuracy'])


def crps_loss(cumsum):
    @tf.function
    def crps(y_true, logit_of_y_pred):
        y_true = K.cast(y_true, dtype='float64')
        y_pred = K.cast(logit_of_y_pred, dtype='float64')
        logit_of_avg = K.log(cumsum / (1 - K.clip(cumsum, 0, 1-K.epsilon())))
        logit_of_y_pred = K.log(y_pred / (1 - K.clip(y_pred, 0, 1 - K.epsilon())))
        sum_of_logits = logit_of_avg + logit_of_y_pred
        inverse_logit = K.exp(sum_of_logits) / (1 + K.exp(sum_of_logits))
        ret = K.switch(y_true is not None and y_true >= 1, inverse_logit - 1, inverse_logit)
        ret = K.square(ret)
        per_play_loss = K.sum(ret, axis=1)
        total_loss = K.mean(per_play_loss)
        return total_loss

    return crps
