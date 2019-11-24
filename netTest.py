import datetime
import math
import random
import sys
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

import CleanData
from Net import Net, crps_loss

random.seed(3)
pd.set_option('display.max_columns', None, 'display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # self.x, self.y, self.cumsum = CleanData.convert_data(pd.read_csv('data/train.csv'))
        # self.x_train = np.asarray(self.x.values)
        # self.y_train = np.asarray(self.y.values)
        # self.cumsum = np.asarray(self.cumsum, dtype='float')
        self.x, self.y, self.cumsum = CleanData.get_csv_data()
        self.x_train = np.asarray(self.x, dtype='float')[:-100*22]
        self.y_train = np.asarray(self.y, dtype='float')[:-100*22]
        self.cumsum = np.asarray(self.cumsum, dtype='float')
        # self.net = Net(self.x_train,
        #                self.y_train,
        #                self.cumsum,
        #                number_of_epochs=500,
        #                num_hiddden_nodes=200)
        # self.initial_net = tf.keras.models.clone_model(self.net.model)
        # self.initial_net_config = self.net.model.predict(self.x_train)
        # self.net_noise = []
        # for i in range(len(self.initial_net_config)):
        #     play = self.initial_net_config[i]
        #     play = list(map(lambda x: math.log(x / (1 - x)), play))
        #     play = self.cumsum - play
        #     self.net_noise.append(play)
        # self.net.update_loss(avg_of_play_no_noise=np.asarray(self.net_noise))


    def test_for_hidden_nodes(self):
        models = []
        for layers in range(10):
            net = Net(self.x_train,
                      self.y_train,
                      self.cumsum,
                      number_of_epochs=layers,
                      num_hiddden_nodes=5)
            initial_net = tf.keras.models.clone_model(net.model)
            initial_net_config = net.model.predict(self.x_train)
            net_noise = []
            for i in range(len(initial_net_config)):
                play = initial_net_config[i]
                play = list(map(lambda x: math.log(x / (1 - x)), play))
                play = self.cumsum - play
                net_noise.append(play)
            net.update_loss(avg_of_play_no_noise=np.asarray(net_noise))
            test_data_x = np.asarray(self.x, dtype='float')[-50*22:]
            test_data_y = np.asarray(self.y, dtype='float')[-50*22:]
            net_noise = initial_net.predict(test_data_x)
            for j in range(len(net_noise)):
                play = net_noise[j]
                play = list(map(lambda x: math.log(x / (1 - x)), play))
                play = self.cumsum - play
                net_noise[j] = play
            m = tf.keras.models.clone_model(net.model)
            m.compile(loss=crps_loss(net_noise))
            best = (0, float('inf'))
            for _ in range(200):
                net.train()
                m = tf.keras.models.clone_model(net.model)
                m.compile(loss=crps_loss(net_noise))
                score = m.evaluate(test_data_x, test_data_y)
                best = (_, score, m, layers) if score < best[1] else best
                print(f'Epoch: {_} - val_loss: {score}\n')
            m = best[2]
            models.append(best)
            predicted = m.predict(test_data_x)
            for input_play, prediction, n in zip(test_data_x, predicted, net_noise):
                yards_2_endzone = int(input_play[-4])
                for i in range(len(prediction)):
                    p = prediction[i]
                    p = math.log(p / (1 - p))
                    p -= self.cumsum[i] + n[i]
                    p = math.exp(p) / (1 + math.exp(p))
                    prediction[i] = 1 - p
                for i in range(yards_2_endzone + 15, len(prediction)):
                    prediction[i] = 1
                for i in range(len(prediction) - 1):
                    if prediction[i + 1] < prediction[i]:
                        prediction[i + 1] = prediction[i]
            prediction = list(map(lambda x: np.pad(x, (84, 0), constant_values=0), predicted))
            with open(f'out/{layers}_layers_{datetime.datetime.now().strftime("%m-%d-%y_%H_%M_%S")}.txt', 'w') as f:
                f.write(f'best: {best}\n')
                for x, y in zip(test_data_y, prediction):
                    i = -99
                    x = np.pad(x, (84, 0), constant_values=0)
                    for a, b in zip(x, y):
                        f.write('i: {: 3d}; Actual: {:f}; Predicted: {:f}\n'.format(i, a, b))
                        i += 1
                    f.write('\n-------------------------\n')

        print(models)
        models.sort(key=lambda q: q[1])
        print(models[0])

    # def test_train_and_predict(self):
    #     # self.net.train()
    #     test_data_x = np.asarray(self.x, dtype='float')[-20:]
    #     test_data_y = np.asarray(self.y, dtype='float')[-20:]
    #     net_noise = self.initial_net.predict(test_data_x)
    #     for i in range(len(net_noise)):
    #         play = net_noise[i]
    #         play = list(map(lambda x: math.log(x / (1 - x)), play))
    #         play = self.cumsum - play
    #         net_noise[i] = play
    #
    #     predicted = self.net.predict(test_data_x, net_noise, self.cumsum)
    #     # m = tf.keras.models.clone_model(self.net.model)
    #     # m.compile(loss=crps_loss(net_noise))
    #     # best = (0, float('inf'))
    #     # for _ in range(10):
    #     #     self.net.train()
    #     #     m = tf.keras.models.clone_model(self.net.model)
    #     #     m.compile(loss=crps_loss(net_noise))
    #     #     score = m.evaluate(test_data_x, test_data_y)
    #     #     best = (_, score, m) if score < best[1] else best
    #     #     print(f'Epoch: {_} - val_loss: {score}\n')
    #     # print(f'best: {best}')
    #     # m = best[2]
    #     # predicted = m.predict(test_data_x)
    #     # for input_play, prediction, n in zip(test_data_x, predicted, net_noise):
    #     #     yards_2_endzone = int(input_play[-4])
    #     #     for i in range(len(prediction)):
    #     #         p = prediction[i]
    #     #         p = math.log(p / (1 - p))
    #     #         p -= self.cumsum[i] + n[i]
    #     #         p = math.exp(p) / (1 + math.exp(p))
    #     #         prediction[i] = 1 - p
    #     #     for i in range(yards_2_endzone + 15, len(prediction)):
    #     #         prediction[i] = 1
    #     #     for i in range(len(prediction) - 1):
    #     #         if prediction[i + 1] < prediction[i]:
    #     #             prediction[i + 1] = prediction[i]
    #     # predicted = list(map(lambda x: np.pad(x, (84, 0), constant_values=0), predicted))
    #     with open(f'out/{datetime.datetime.now().strftime("%m-%d-%y_%H_%M_%S")}.txt', 'w') as f:
    #         for x, y in zip(test_data_y, predicted):
    #             i = -99
    #             x = np.pad(x, (84, 0), constant_values=0)
    #             for a, b in zip(x, y):
    #                 f.write('i: {: 3d}; Actual: {:f}; Predicted: {:f}\n'.format(i, a, b))
    #                 i += 1
    #             f.write('\n-------------------------\n')
    #     self.assertTrue(True, True)


if __name__ == '__main__':
    unittest.main()
