import math
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
                 avg_by_play,
                 number_of_epochs=10,
                 num_hiddden_nodes=None):
        self.model = tf.keras.Sequential()
        self.x_train = x_train
        self.y_train = y_train
        self.avg_by_play = avg_by_play
        self.number_of_epochs = number_of_epochs
        if not num_hiddden_nodes:
            self.num_hidden_nodes = len(self.x_train[0])
        else:
            self.num_hidden_nodes = num_hiddden_nodes
        self.set_up_model()

    def set_up_model(self):
        num_of_input_neurons = len(self.x_train[0])
        num_of_output_neurons = len(self.y_train[0])
        self.model.add(tf.keras.layers.Flatten(input_shape=(num_of_input_neurons,)))
        self.model.add(tf.keras.layers.Dense(self.num_hidden_nodes,
                                             activation=tf.nn.sigmoid))
        self.model.add(tf.keras.layers.Dense(num_of_output_neurons,
                                             activation=tf.nn.sigmoid))
        self.model.compile(optimizer='adam',
                           # loss=crps_loss_func,
                           loss=crps_loss(K.placeholder(shape=self.y_train.shape, dtype='float64')),
                           metrics=['accuracy'])
        # self.model.summary()

    def train(self):
        validation_overfitting = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                  min_delta=1,
                                                                  patience=50,
                                                                  verbose=0,
                                                                  mode='min')
        callbacks_list = [validation_overfitting]

        self.model.fit(self.x_train,
                       self.y_train,
                       # validation_split=.2,
                       epochs=1,
                       batch_size=len(self.x_train),
                       # callbacks=callbacks_list
                       )

    def predict(self, x_input, noise, cumsum):
        predicted = self.model.predict(x_input)
        for input_play, prediction, n in zip(x_input, predicted, noise):
            yards_2_endzone = int(input_play[-4])
            for i in range(len(prediction)):
                p = prediction[i]
                p = math.log(p / (1 - p))
                p -= cumsum[i] + n[i]
                p = math.exp(p) / (1 + math.exp(p))
                prediction[i] = 1 - p
            for i in range(yards_2_endzone + 15, len(prediction)):
                prediction[i] = 1
            for i in range(len(prediction) - 1):
                if prediction[i + 1] < prediction[i]:
                    prediction[i + 1] = prediction[i]
        predicted = list(map(lambda x: np.pad(x, (84, 0), constant_values=0), predicted))
        return predicted

    def update_loss(self, avg_of_play_no_noise):
        self.model.compile(loss=crps_loss(avg_of_play_no_noise))
        self.avg_by_play = avg_of_play_no_noise


def crps_loss(avg_of_play_no_noise):
    def crps(y_true, y_pred):
        logit_of_y_pred = K.log(y_pred / (1 - K.clip(y_pred, 0, 1 - 10 ** -16)))
        sum_of_logits = avg_of_play_no_noise + logit_of_y_pred
        inverse_logit = K.exp(sum_of_logits) / (1 + K.exp(sum_of_logits))
        inverse_logit = tf.where(tf.math.is_nan(inverse_logit), tf.ones_like(inverse_logit), inverse_logit)
        ret = tf.where(y_true >= 1, inverse_logit - 1, inverse_logit)
        ret = K.square(ret)
        per_play_loss = K.sum(ret, axis=1)
        total_loss = K.mean(per_play_loss)
        return total_loss

    return crps


def crps_loss_func(y_true, y_pred):
    ret = tf.where(y_true >= 1, y_pred - 1, y_pred)
    ret = K.square(ret)
    per_play_loss = K.sum(ret, axis=1)
    total_loss = K.mean(per_play_loss)
    return total_loss

# A(d)= logit(overall proportion of play make <=d yards)
# p(d)= logit^{-1} (A(d) + logit c(d))
# Then apply old loss function to these p(d). 
# 
# logit(p)= log(p/(1-p))  
# logit^{-1}(x)= e^x/(1+e^x)

