import random
from datetime import datetime
import numpy as np
import tensorflow as tf


class Net:
    random.seed(1)
    np.random.seed(2)
    tf.random.set_seed(3)
    tf.keras.backend.set_floatx('float64')

    def __init__(self,
                 x_train,
                 y_train,
                 number_of_epochs=10,
                 batch_size=10):
        self.model = tf.keras.Sequential()
        self.x_train = x_train
        self.y_train = y_train
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
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
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self):
        self.model.fit(self.x_train,
                       self.y_train,
                       epochs=self.number_of_epochs,
                       batch_size=self.batch_size)
        date = datetime.now().strftime("%m%d%y%H%M%S")
        # self.model.save(f'net{date}.h5')

    def predict(self, x_input):
        predicted = self.model.predict([x_input])
        predicted = [x / sum(predicted[0]) for x in predicted[0]]
        random_num = random.random()
        cutoff = 0
        for i in range(10):
            for j in range(10):
                print(format(round(np.asarray(predicted)[i*10+j], 4), '.4f') + '  ', end='')
            print()
        for i in range(len(predicted)):
            prob = predicted[i]
            cutoff += prob
            if random_num <= cutoff:
                return i
