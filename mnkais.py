import random

from mnk import Shape

class RandomAi:
	def play(self, game):
		candidates = []
		for i in range(len(game.array)):
			for j in range(len(game.array[i])):
				if game.array[i][j] == game.empty:
					candidates.append((j, i))
		return random.choice(candidates)

class FillerAi:
	def play(self, game):
		for i in range(len(game.array)):
			for j in range(len(game.array)):
				if game.array[i][j] == game.empty:
					return j, i