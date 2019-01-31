import numpy as np
import unittest

from mnkutil import *

from mnk import Shape

class TestMnkUtil(unittest.TestCase):
	def test_to_dense_input(self):
		a = [[Shape.X, Shape.O, Shape.N], [Shape.N, Shape.N, Shape.N], [Shape.N, Shape.O, Shape.X]]
		np.testing.assert_equal(to_dense_input(a), np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0]))

if __name__ == "__main__":
	unittest.main()
