from datetime import datetime

import numpy as np
import random
import unittest

import pandas as pd

from Net import Net

random.seed(3)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        df_input = pd.read_csv('data/dist_ang_radial_tang_x_y_disfromyl.csv', sep=',', header=None)
        df_output = pd.read_csv('data/dist_ang_radial_tang_x_y_disfromyl_yards.csv', sep=',', header=None)
        load_filename = 'net_configurations/crps_net_trained_with_10000_on_10-26-19_18_35_37.h5'
        self.x_train = df_input.values
        self.y_train = df_output.values
        self.net = Net(self.x_train,
                       self.y_train,
                       number_of_epochs=1000,
                       load_filename=load_filename)

    def test_train_and_predict(self):
        self.net.train()
        prediction = self.net.predict(np.asarray(self.x_train[:10]))
        with open(f'out/{datetime.now().strftime("%m-%d-%y_%H_%M_%S")}.txt', 'w') as f:
            for x, y in zip(self.y_train[:10], prediction[:10]):
                i = -99
                for a, b in zip(x, y):
                    f.write('i: {: 3d}; Actual: {:d}; Predicted: {:f}\n'.format(i, a, b))
                    i += 1
                f.write('\n-------------------------\n')
        self.assertTrue(True, True)


if __name__ == '__main__':
    unittest.main()
