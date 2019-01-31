import tensorflow as tf
import time

from metrics import evaluate_player, prepareplt, shownonblock
from mnk import MnkGame
from mnkais import RandomAi
from mnkutil import choose_cell_weighted, to_dense_input, to_dense_index, needs_session

class PolgradRunnerTf:
	"""The tensors. The first element is the input placeholder and the last one the output layer."""
	layers = []
	def __init__(self, node_nums, activations, last_logged=False):
		"""
		Builds the graph.
		:param node_nums: A list containing the number of neurons in each layer from input to output.
		:param activations: The list of Tensorflow activation functions to be used in each connections.
		:param last_logged: Whether the activation function for the output layer is "logged"(e.g. tf.nn.log_softmax).
		"""
		self.node_nums = node_nums
		self.layers.append(tf.placeholder(tf.float32, shape=(None, node_nums[0])))
		for i in range(len(node_nums))[1:]:
			self.layers.append(tf.layers.dense(self.layers[i - 1], node_nums[i], activations[i - 1]))
		self.reward_t = tf.placeholder(tf.float32)
		self.sampled_t = tf.placeholder(tf.int32)
		self.onehot_sampled_t = tf.one_hot(self.sampled_t, self.node_nums[-1])
		if not last_logged:
			self.layers[-1] = tf.log(tf.clip_by_value(self.layers[-1], 1e-9, 1.0))
		self.loss_t = -(self.reward_t * tf.reduce_sum(tf.multiply(self.onehot_sampled_t, self.layers[-1]), axis=1))
		#tf.summary.histogram("Loss", self.loss_t)
		self.trainer = tf.train.AdamOptimizer(0.001).minimize(self.loss_t)

	@needs_session
	def forward_propagate(self, samples, session=None):
		"""
		Propagates forward the samples and calculates the probabilities for each cells.
		:param samples: 2d np.ndarray with rows as samples and columns as cells.
		:param session: The session to use. If not supplied, a new session is created and closed.
		:return: The probabilities for each cell, flattened in C order.
		"""
		result = session.run(self.layers[-1], feed_dict={self.layers[0]: samples})
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
				decision = self.play(game, board=input_d, session=session)
				boards.append((input_d, to_dense_index(game, decision)))
				game.place(decision)
				if game.checkWin(decision):
					break
			result.append(boards) #train() will figure out who won from len(boards) % 2
			game.initialize()
		return result

	@needs_session
	def train(self, dimen, winLen, batch_size=100, cycles=1000, stops=100, session=None):
		"""
		Trains the network with policy gradients.
		:param dimen: A 2-tuple: (horSize, verSize) of the game.
		:param winLen: The k in m,n,k game.
		:param batch_size: The cycles argument in selfplay.
		:param cycles: How many batches of games to play.
		:param stops: At every (stops)th step, the runner will print out some statistics.
		:param session: The tf.Session to use. If not specified, a new Session is created.
		"""
		#writer = tf.summary.FileWriter("logs/")
		histo = {0.2: 0, 1: 0, 0: 0}
		performances = []
		ai = RandomAi()
		prepareplt()
		started = time.time()
		for i in range(cycles):
			if i % stops == (stops - 1):
				print(f"{i + 1}th cycle, {time.time() - started} seconds have elapsed since the training started")
				performances.append(evaluate_player(lambda g: self.play(g, to_dense_input(g.array), session=session), lambda g: ai.play(g), rules=(dimen[0], dimen[1], winLen)))
				shownonblock(performances, labels=["1st won", "2nd won", "draw"])
			plays = self.selfplay(dimen, winLen, batch_size, session)
			for play in plays:
				for turn, board in enumerate(play):
					reward_d = 0.2 if len(play) == self.node_nums[0] else 1 if len(play) % 2 == turn % 2 else 0 #The modulus == 1 if 1st won else 0
					histo[reward_d] += 1
					if reward_d != 0:
						session.run(self.trainer, feed_dict={self.layers[0]: [board[0]], self.reward_t: reward_d, self.sampled_t: board[1]})
		print(histo)
		return histo, performances