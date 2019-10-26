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
        load_filename = 'binary_crossentropy_net_trained_with_1000_on_10-26-19_14_06_15.h5'
        self.x_train = df_input.values
        self.y_train = df_output.values
        self.net = Net(self.x_train,
                       self.y_train,
                       number_of_epochs=1000)

    def test_something(self):
        self.net.train()
        prediction = self.net.predict(np.asarray(self.x_train[:10]))
        for x, y in zip(self.y_train[:10], prediction[:10]):
            i = -99
            for a, b in zip(x, y):
                print(f'i: {i}; Actual: {round(a, 5)}; Predicted: {b}')
                i += 1
            print('\n-------------------------\n')
        self.assertEqual(all(self.y_train[0]), False)


if __name__ == '__main__':
    unittest.main()
