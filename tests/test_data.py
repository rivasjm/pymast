import unittest

from data import to_vector
from examples import get_palencia_system


class MyTestCase(unittest.TestCase):
    def test_to_vector(self):
        system = get_palencia_system()
        to_vector(system, (2, 3, 3))


if __name__ == '__main__':
    unittest.main()
