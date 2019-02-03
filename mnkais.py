from collections import deque
import numpy as np
import random

from mnk import Point, Shape

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

class Human:
	def play(self, game):
		print(np.array(game.array))
		return eval(input())

class EmacsGomokuAi:
	class Mode:
		HOR = 0
		VER = 1
		SLASH = 2
		RESLASH = 3
		@classmethod
		def perpendicular(cls, mode):
			return {cls.HOR: cls.VER, cls.VER: cls.HOR, cls.SLASH: cls.RESLASH, cls.RESLASH: cls.SLASH}[mode]

	ownValues = [7, 35, 800, 15000, 800000, 4000000, 20000000, 2100000000]
	enemValues = [7, 15, 400, 1800, 100000, 600000, 1500000, 80000000]
	xP = [1, 0, -1, 1]; yP = [0, 1, 1, 1]

	def play(self, game):
		self.game = game
		self.myShape = game.shapes[game.nextIndex]
		self.enemShape = Shape.O if self.myShape == Shape.X else Shape.X
		values = np.zeros((game.verSize, game.horSize), dtype=np.int32)
		self.checkTuples(game, EmacsGomokuAi.Mode.HOR, values)
		self.checkTuples(game, EmacsGomokuAi.Mode.VER, values)
		self.checkTuples(game, EmacsGomokuAi.Mode.SLASH, values)
		self.checkTuples(game, EmacsGomokuAi.Mode.RESLASH, values)
		return self.findMax(game, values)

	def checkTuples(self, game, mode, values):
		po = Point(game.horSize - 1 if mode == EmacsGomokuAi.Mode.RESLASH else 0, 0)
		while game.inBoundary(po.y, po.x):
			count = {Shape.X: 0, Shape.O: 0}
			ktuple = deque()
			pi = Point(po)
			while game.inBoundary(pi.y, pi.x):
				if game.array[pi.y][pi.x] != Shape.N:
					count[game.array[pi.y][pi.x]] += 1
				ktuple.append(game.array[pi.y][pi.x])
				if self.bigEnough(po, pi, mode):
					def stampValue(x, y):
						if count[self.enemShape] == 0:
							values[y][x] += EmacsGomokuAi.ownValues[count[self.myShape]]
						elif count[self.myShape] == 0:
							values[y][x] += EmacsGomokuAi.enemValues[count[self.enemShape]]
					self.forBackward(pi, mode, game.winStreak, stampValue)
					removed = ktuple.popleft()
					if removed != Shape.N:
						count[removed] -= 1
				self.forward(pi, mode)
			if mode == EmacsGomokuAi.Mode.HOR or mode == EmacsGomokuAi.Mode.VER:
				self.forward(po, EmacsGomokuAi.Mode.perpendicular(mode))
			elif mode == EmacsGomokuAi.Mode.SLASH and po.x < game.horSize - 1:
				po.x += 1
			elif mode == EmacsGomokuAi.Mode.RESLASH and po.x > 0:
				po.x -= 1
			else:
				po.y += 1

	def bigEnough(self, po, pi, mode):
		if mode == EmacsGomokuAi.Mode.HOR:
			return pi.x >= self.game.winStreak - 1
		elif mode == EmacsGomokuAi.Mode.VER:
			return pi.y >= self.game.winStreak - 1
		else:
			return pi.y - po.y >= self.game.winStreak - 1

	def forward(self, p, mode):
		p.y += EmacsGomokuAi.yP[mode]
		p.x += EmacsGomokuAi.xP[mode]

	def backward(self, p, mode):
		p.y -= EmacsGomokuAi.yP[mode]
		p.x -= EmacsGomokuAi.xP[mode]

	def forBackward(self, fromp, mode, n, action):
		p = Point(fromp)
		for i in range(n):
			action(p.x, p.y)
			self.backward(p, mode)

	def findMax(self, game, values):
		maxv = -1
		maxI = -1; maxJ = -1
		for i in range(game.verSize):
			for j in range(game.horSize):
				if game.array[i][j] == game.empty and values[i][j] > maxv:
					maxv = values[i][j]
					maxI = i
					maxJ = j
		return None if maxv == -1 else (maxJ, maxI)
