import random
import unittest

import numpy as np
import pandas as pd

from Net import Net

random.seed(3)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        df_input = pd.read_csv('data/dist_ang_radial_tang_x_y_disfromyl.csv', sep=',', header=None)
        df_output = pd.read_csv('data/dist_ang_radial_tang_x_y_disfromyl_yards.csv', sep=',', header=None)
        self.x_train = df_input.values
        self.y_train = df_output.values
        self.net = Net(self.x_train, self.y_train, number_of_epochs=1000)

    def test_something(self):
        self.net.train()
        prediction = self.net.predict(np.asarray(self.x_train[:1]))
        i = -99
        for x, y in zip([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1
                         ], prediction[0]):
            print(f'i: {i}; Actual: {round(x, 5)}; Predicted: {y}')
            i += 1
        self.assertEqual(self.y_train[0], prediction)


if __name__ == '__main__':
    unittest.main()
