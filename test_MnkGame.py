import unittest

from mnk import MnkGame, Shape

class TestMnkGame(unittest.TestCase):
	def test_1(self):
		game = MnkGame(11, 19)
		#Test if it stores its size correctly
		self.assertEqual(11, game.horSize)
		self.assertEqual(19, game.verSize)

		#Test if it initializes the game correctly
		game.initialize()
		self.forAllCellOn(game, lambda shape: self.assertEqual(Shape.N, shape))

		#Test nextIndex
		self.assertEqual(0, game.nextIndex)
		game.place(1, 1)
		self.assertEqual(1, game.nextIndex)

		#Test changeShape
		game.changeShape(1)
		self.assertEqual(0, game.nextIndex)
		game.changeShape(2)
		self.assertEqual(0, game.nextIndex)

	def test_checkWin(self):
		tp = [[(1, 1), (3, 3), (1, 0), (3, 2), (1, 2)], [(0, 3), (0, 0), (1, 3), (1, 1), (2, 3), (2, 2)]]
		game = MnkGame(4, 4, 3)
		for s in tp:
			game.initialize()
			for t in s:
				try:
					game.place(t)
				except ValueError:
					raise ValueError(f"s: {s}, t: {t}")
			self.assertTrue(game.checkWin(s[len(s) - 1]))

	@staticmethod
	def forAllCellOn(game, action):
		for i in game.array:
			for j in i:
				action(j)

if __name__ == "__main__":
	unittest.main()
