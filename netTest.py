import random
import unittest
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import CleanData
from Net import Net, crps_loss, crps_loss_func

random.seed(3)
pd.set_option('display.max_columns', None, 'display.max_rows', None)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        x_train, y_train, self.cumsum = CleanData.convert_data(pd.read_csv('data/train.csv').head(22*20))
        self.x_train = np.asarray(x_train.values)#[:-10]
        self.y_train = np.asarray(y_train.values)#[:-10]
        print(f'x_train.values: {list(x_train.columns)}')
        print(f'x_train[0]: {self.x_train[0]}')
        self.net = Net(self.x_train,
                       self.y_train,
                       self.cumsum,
                       batch_size=10,
                       number_of_epochs=10)

    def test_train_and_predict(self):
        self.net.train()
        prediction = self.net.predict(np.asarray(self.x_train[-5:]))
        self.cumsum = np.pad(self.cumsum, (84, 0), constant_values=0)
        with open(f'out/{datetime.now().strftime("%m-%d-%y_%H_%M_%S")}.txt', 'w') as f:
            for x, y in zip(self.y_train[-5:], prediction):
                i = -99
                x = np.pad(x, (84, 0), constant_values=0)
                f.write(f'LOSS: {crps_loss(self.cumsum)(tf.convert_to_tensor([x]), tf.convert_to_tensor(y))}\n')
                # f.write(f'LOSS: {crps_loss_func(tf.convert_to_tensor([x]), tf.convert_to_tensor(y))}\n')
                for a, b in zip(x, y):
                    f.write('i: {: 3d}; Actual: {:d}; Predicted: {:f}\n'.format(i, a, b))
                    i += 1
                f.write('\n-------------------------\n')
        self.assertTrue(True, True)


if __name__ == '__main__':
    unittest.main()
