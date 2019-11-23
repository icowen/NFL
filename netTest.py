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
        self.x, self.y, self.cumsum = CleanData.convert_data(pd.read_csv('data/train.csv'))
        self.x_train = np.asarray(self.x.values)
        self.y_train = np.asarray(self.y.values)
        self.cumsum = np.asarray(self.cumsum, dtype='float')
        # self.x, self.y, cumsum = CleanData.get_csv_data()
        # self.x_train = np.asarray(self.x, dtype='float')[:100]
        # self.y_train = np.asarray(self.y, dtype='float')[:100]
        # self.cumsum = np.asarray(cumsum, dtype='float')
        self.net = Net(self.x_train,
                       self.y_train,
                       self.cumsum,
                       number_of_epochs=500,
                       num_hiddden_nodes=200)
        self.initial_net = tf.keras.models.clone_model(self.net.model)
        self.initial_net_config = self.net.model.predict(self.x_train)
        self.net_noise = []
        for i in range(len(self.initial_net_config)):
            play = self.initial_net_config[i]
            play = list(map(lambda x: math.log(x / (1 - x)), play))
            play = self.cumsum - play
            self.net_noise.append(play)
        self.net.update_loss(avg_of_play_no_noise=np.asarray(self.net_noise))
        # for i in range(3):
        #     num_hiddden_nodes = 195 + 5 * i
        #     print(f'Training net with {num_hiddden_nodes} hidden nodes')
        #     self.net = Net(self.x_train,
        #                    self.y_train,
        #                    self.cumsum,
        #                    batch_size=10,
        #                    number_of_epochs=300,
        #                    num_hiddden_nodes=num_hiddden_nodes)
        #     self.net.train()
        #     avg_by_play = self.net.predict(np.asarray(self.x_train[-10:]))
        #     c = np.pad(self.cumsum, (84, 0), constant_values=0)
        #     with open(f'out/{num_hiddden_nodes}_hidden_nodes.txt', 'w') as f:
        #         for x, y in zip(self.y_train[-10:], avg_by_play):
        #             i = -99
        #             x = np.pad(x, (84, 0), constant_values=0)
        #             f.write(f'LOSS: {crps_loss(c)(tf.convert_to_tensor([x]), tf.convert_to_tensor(y))}\n')
        #             # f.write(f'LOSS: {crps_loss_func(tf.convert_to_tensor([x]), tf.convert_to_tensor(y))}\n')
        #             for a, b in zip(x, y):
        #                 f.write('i: {: 3d}; Actual: {:d}; Predicted: {:f}\n'.format(i, a, b))
        #                 i += 1
        #             f.write('\n-------------------------\n')

    def test_train_and_predict(self):
        # self.net.train()
        test_data_x = np.asarray(self.x, dtype='float')[-20:]
        test_data_y = np.asarray(self.y, dtype='float')[-20:]
        net_noise = self.initial_net.predict(test_data_x)
        for i in range(len(net_noise)):
            play = net_noise[i]
            play = list(map(lambda x: math.log(x / (1 - x)), play))
            play = self.cumsum - play
            net_noise[i] = play
        prediction = self.net.predict(test_data_x, net_noise, self.cumsum)
        m = tf.keras.models.clone_model(self.net.model)
        m.compile(loss=crps_loss(net_noise))
        best = (0, float('inf'))
        for _ in range(200):
            self.net.train()
            m = tf.keras.models.clone_model(self.net.model)
            m.compile(loss=crps_loss(net_noise))
            score = m.evaluate(test_data_x, test_data_y)
            best = (_, score) if score < best[1] else best
            print(f'Epoch: {_} - val_loss: {score}\n')
        print(f'best: {best}')
        with open(f'out/{datetime.datetime.now().strftime("%m-%d-%y_%H_%M_%S")}.txt', 'w') as f:
            f.write(f'best: {best}\n')
            for x, y in zip(test_data_y, prediction):
                i = -99
                x = np.pad(x, (84, 0), constant_values=0)
                for a, b in zip(x, y):
                    f.write('i: {: 3d}; Actual: {:f}; Predicted: {:f}\n'.format(i, a, b))
                    i += 1
                f.write('\n-------------------------\n')
        self.assertTrue(True, True)


if __name__ == '__main__':
    unittest.main()
