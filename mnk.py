# Translated and modified from PIRI MNK 1.5.
import copy

class Shape:
	X = -1
	O = 1
	N = 0

class Move:
	def __init__(self, coord, placed, prev):
		self.coord = coord
		self.placed = placed
		self.prev = prev

class Point:
	def __init__(self, x, y=None):
		if isinstance(x, Point):
			self.x = x.x
			self.y = x.y
		else:
			self.x = x
			self.y = y

class MnkGame:
	def __init__(self, hor=15, ver=15, winStreak=5, copyFrom=None):
		if copyFrom is not None:
			self.horSize = copyFrom.horSize; self.verSize = copyFrom.verSize
			self.winStreak = copyFrom.winStreak
			self.array = copy.deepcopy(copyFrom.array)
			self.shapes = copy.deepcopy(copyFrom.shapes)
			self.empty = copyFrom.empty
			self.history = copy.deepcopy(copyFrom.history)
			self.nextIndex = copyFrom.nextIndex
		else:
			self.shapes = [Shape.X, Shape.O]
			self.empty = Shape.N
			self.nextIndex = 0
			self.horSize = hor; self.verSize = ver
			self.winStreak = winStreak
			self.array = [[self.empty] * self.horSize] * self.verSize
			self.history = []
			self.initialize()

	def place(self, x, y=0):
		"""Supply a tuple or 2 integers."""
		if isinstance(x, tuple):
			y = x[1]; x = x[0]
		if self.array[y][x] != self.empty:
			raise ValueError("Attempted to play on a filled cell")
		if not self.inBoundary(y, x):
			raise ValueError("Attempted to play outside the board")
		self.history.append(Move(Point(x, y), self.shapes[self.nextIndex], self.array[y][x]))
		self.array[y][x] = self.shapes[self.nextIndex]
		self.changeShape(1)

	def initialize(self):
		self.array = [[self.empty for j in range(self.horSize)] for i in range(self.verSize)]
		self.nextIndex = 0
		self.history.clear()

	def revertLast(self):
		if len(self.history) > 0:
			m = self.history.pop()
			self.array[m.coord.y][m.coord.x] = m.prev
			self.nextIndex = self.shapes.index(m.placed)
			return True
		return False

	def changeShape(self, steps):
		self.nextIndex = abs((self.nextIndex + steps) % len(self.shapes))

	# setSize not implemented
	def checkWin(self, x, y=0):
		"""Supply a tuple or 2 integers."""
		if isinstance(x, tuple):
			y = x[1]; x = x[0]
		i = 0; j = 0; streak = 0
		array = self.array
		# vertical check
		for i in range(self.verSize - 1):
			streak += 1
			if array[i][x] != array[i + 1][x] or array[i][x] == self.empty:
				streak = 0
			if streak == self.winStreak - 1:
				return True
		# horizontal check
		streak = 0
		for i in range(self.horSize - 1):
			streak += 1
			if array[y][i] != array[y][i + 1] or array[y][i] == self.empty:
				streak = 0
			if streak == self.winStreak - 1:
				return True
		# diagonal check / shape
		streak = 0
		i = self.verSize - 1 if x + y >= self.verSize else x + y
		j = (x + y) - i
		while self.inBoundary(i - 1, j + 1):
			streak += 1
			if array[i][j] != array[i - 1][j + 1] or array[i][j] == self.empty:
				streak = 0
			if streak == self.winStreak - 1:
				return True
			i -= 1; j += 1
		# diagonal check \ shape
		streak = 0
		i = 0 if y - x < 0 else y - x
		j = 0 if x - y < 0 else x - y
		while i + 1 < self.verSize and j + 1 < self.horSize:
			streak += 1
			if array[i][j] != array[i + 1][j + 1] or array[i][j] == self.empty:
				streak = 0
			if streak == self.winStreak - 1:
				return True
			i += 1; j += 1
		return False

	# getLinePoints not implemented
	def inBoundary(self, y, x):
		return 0 <= y < self.verSize and 0 <= x < self.horSize
