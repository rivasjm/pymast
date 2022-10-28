import unittest
from vholistic import  jitter_matrix, successor_matrix
import numpy as np


class VHolisticTest(unittest.TestCase):
    def test_jitter_matrix(self):
        successors = np.array([2, 3, -1, 5, 6, -1], dtype=np.int32).reshape((-1, 1))
        s = successor_matrix(successors)
        R = np.array([[5, 12, 25, 5, 10, 15],
                      [6, 13, 26, 6, 11, 16]]).transpose().ravel('F').reshape(2, 6, 1)
        J = jitter_matrix(s, R)
        self.assertEqual(J[0, 0, 0], 0)
        self.assertEqual(J[0, 2, 0], 12)
        self.assertEqual(J[0, 5, 0], 10)
        self.assertEqual(J[1, 0, 0], 0)
        self.assertEqual(J[1, 2, 0], 13)
        self.assertEqual(J[1, 5, 0], 11)


if __name__ == '__main__':
    unittest.main()