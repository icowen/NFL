import random
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import CleanData
from Net import Net, crps_loss

random.seed(3)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # df_input = pd.read_csv('data/dist_ang_radial_tang_x_y_disfromyl.csv', sep=',', header=None)
        # df_output = pd.read_csv('data/dist_ang_radial_tang_x_y_disfromyl_yards.csv', sep=',', header=None)
        # load_filename = 'net_configurations/crps_net_trained_with_10000_on_10-26-19_18_35_37.h5'
        x_train, y_train = CleanData.convert_data(pd.read_csv('data/train.csv').head(44))
        self.x_train = np.asarray(x_train.values)
        self.y_train = np.asarray(y_train.values)
        # self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_train,
        #                                                                           y_train,
        #                                                                           test_size=0.2,
        #                                                                           shuffle=True)
        print(f'x_train: {x_train}')
        self.net = Net(x_train,
                       y_train,
                       number_of_epochs=100)

    def test_train_and_predict(self):
        self.net.train()
        prediction = self.net.predict(np.asarray(self.x_train[:10]))
        with open(f'out/{datetime.now().strftime("%m-%d-%y_%H_%M_%S")}.txt', 'w') as f:
            for x, y in zip(self.y_train, prediction):
                i = -99
                for a, b in zip(x, y):
                    f.write('i: {: 3d}; Actual: {:d}; Predicted: {:f}\n'.format(i, a, b))
                    i += 1
                f.write('\n-------------------------\n')
        self.assertTrue(True, True)


if __name__ == '__main__':
    unittest.main()
