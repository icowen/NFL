import random
import unittest
from Net import Net
random.seed(3)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.x_train = [[random.random() for j in range(100)] for i in range(100)]
        self.y_train = [[random.random() for j in range(100)] for i in range(100)]
        self.net = Net(self.x_train, self.y_train)

    def test_something(self):
        self.net.train()
        prediction = self.net.predict(self.x_train[0])
        print(f'prediction: {prediction}')
        self.assertEqual(prediction, 4)


if __name__ == '__main__':
    unittest.main()
