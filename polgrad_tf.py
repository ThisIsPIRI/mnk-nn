import tensorflow as tf

from mnk import MnkGame
from mnkutil import choose_cell_weighted, to_dense_input, reverseboard, needs_session

class PolgradRunnerTf:
	"""The tensors. The first element is the input placeholder and the last one the output layer."""
	layers = []
	def __init__(self, node_nums, activations):
		"""
		Builds the graph and optionally loads saved weights.
		:param node_nums: A list containing the number of neurons in each layer from input to output.
		:param activations: The list of Tensorflow activation functions to be used in each connections.
		"""
		self.node_nums = node_nums
		self.layers.append(tf.placeholder(tf.float32, shape=(None, node_nums[0])))
		for i in range(len(node_nums))[1:]:
			self.layers.append(tf.layers.dense(self.layers[i - 1], node_nums[i], activations[i - 1]))
		self.modulator_t = tf.placeholder(tf.float32)
		#loss = NotImplementedError("stub!") TODO: Implement
		#self.trainer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

	@needs_session
	def forward_propagate(self, samples, session=None):
		"""
		Propagates forward the samples and calculates the probabilities for each cells.
		:param samples: 2d np.ndarray with rows as samples and columns as cells.
		:param session: The session to use. If not supplied, a new session is created and closed.
		:return: The probabilities for each cell, flattened in C order.
		"""
		result = session.run(self.layers[len(self.layers) - 1], feed_dict={self.layers[0]: samples})
		return result

	def play(self, game, board=None, session=None):
		"""
		Determines the best cell to play on.
		:param game: An MnkGame.
		:param board: A 1d np.ndarray. Fed into the network instead of game.array if supplied.
		:param session: The tf.Session to use. If not supplied, a new Session is created.
		:return: The chosen cell's (x, y) coordinates.
		"""
		if board is None:
			input_d = to_dense_input(game.array)
		else:
			input_d = board
		return choose_cell_weighted(game, self.forward_propagate(input_d.reshape(-1, self.node_nums[0]), session)[0]) #Reshape to 2d before passing it to the network

	@needs_session
	def selfplay(self, dimen, winLen, cycles=100, session=None):
		"""
		Plays a game against itself cycles times and returns a list containing all states encountered in all games except the terminal states.
		:param dimen: A 2-tuple: (horSize, verSize) of the game.
		:param winLen: The k in m,n,k game.
		:param cycles: How many games to play.
		:param session: The tf.Session to use. If not specified, a new Session is created.
		:return: A list of lists containing 1~(dimen[0] * dimen[1]) 1d np.ndarrays.
		"""
		result = []
		game = MnkGame(dimen[0], dimen[1], winLen)
		for i in range(cycles):
			boards = []
			for j in range(game.horSize * game.verSize):
				input_d = to_dense_input(game.array)
				if j % 2 == 0: #Represent our side's stone by 1 and the opponent's by -1.
					input_d = reverseboard(input_d)
				boards.append(input_d)
				decision = self.play(game, board=input_d, session=session)
				game.place(decision)
				if game.checkWin(decision):
					break
			#Do nothing in a draw. TODO: do something?
			result.append(boards) #train() will figure out who won from len(boards) % 2
			game.initialize()
		return result

	@needs_session
	def train(self, dimen, winLen, batch_size=100, cycles=1000, session=None):
		"""
		Trains the network with policy gradients.
		:param dimen: A 2-tuple: (horSize, verSize) of the game.
		:param winLen: The k in m,n,k game.
		:param batch_size: The cycles argument in selfplay.
		:param cycles: How many batches of games to play.
		:param session: The tf.Session to use. If not specified, a new Session is created.
		"""
		for i in range(cycles):
			self.selfplay(dimen, winLen, batch_size, session)
		#TODO: the core
