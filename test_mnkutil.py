import unittest

from mnkutil import *

from mnk import Shape

class TestMnkUtil(unittest.TestCase):
	def test_to_dense_input(self):
		a = [[Shape.X, Shape.O, Shape.N], [Shape.N, Shape.N, Shape.N], [Shape.N, Shape.O, Shape.X]]
		print(to_dense_input(a))

if __name__ == "__main__":
	unittest.main()
