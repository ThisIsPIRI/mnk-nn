import itertools
import tensorflow as tf
import time

from metrics import evaluate_player, prepareplt, shownonblock
from mnk import MnkGame, Shape
from mnkais import RandomAi
from mnkutil import choose_cell_weighted, to_dense_input, to_dense_index, needs_session

class PolgradRunnerTf:
	"""The tensors. The first element is the input placeholder and the last one the output layer."""
	layers = []
	def __init__(self, node_nums, activations):
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
		self.onehot_sampled_t = tf.reshape(tf.one_hot(self.sampled_t, self.node_nums[-1]), (-1, 1)) #Reshape to 2d for matmul
		self.loss_t = -tf.reduce_sum(self.reward_t * tf.reduce_sum(tf.matmul(tf.log(tf.clip_by_value(self.layers[-1], 1e-8, 1.0)), self.onehot_sampled_t), axis=1))
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

	def play(self, game, board=None, session=None, returnProbs=False):
		"""
		Determines the best cell to play on.
		:param game: An MnkGame.
		:param board: A 1d np.ndarray. Fed into the network instead of game.array if supplied.
		:param session: The tf.Session to use. If not supplied, a new Session is created.
		:param returnProbs: If True, the probabilities will be returned along with the coordinates.
		:return: The chosen cell's (x, y) coordinates. ((x, y), probabilities list) if returnProbs.
		"""
		if board is None:
			input_d = to_dense_input(game.array)
		else:
			input_d = board
		return choose_cell_weighted(game, self.forward_propagate(input_d.reshape(-1, self.node_nums[0]), session)[0], returnProbs) #Reshape to 2d before passing it to the network

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
			winner = 0
			for j in range(game.horSize * game.verSize):
				input_d = to_dense_input(game.array)
				decision = self.play(game, board=input_d, session=session)
				boards.append((input_d, to_dense_index(game, decision)))

				#TODO: Remove
				try:
					game.place(decision)
				except ValueError:
					print(game.array)
					fd = {self.layers[0]: input_d.reshape(-1, self.node_nums[0])}
					for l in self.layers[1:]:
						print("layer: ", session.run(l, fd))
					session.close()
					input("What to do now?")

				if game.checkWin(decision):
					winner = Shape.O if j % 2 == 1 else Shape.X
					break
			result.append((boards, winner)) #-1: 1st won, 1: 2nd won, 0: draw
			game.initialize()
		return result

	@needs_session
	def _train_cycle(self, dimen, winLen, batch_size, rewards, histo, session=None):
		"""
		See train() for parameter documentations.
		:param histo: The histogram dict {rewards[0]: int, rewards[1]: int, rewards[2]: int}. The numbers will be incremented.
		"""
		winr, loser, drawr = rewards
		plays = self.selfplay(dimen, winLen, batch_size, session)
		def get_reward(won_side, i):
			if won_side == 0:
				histo[drawr] += 1
				return drawr
			elif (i % 2 == 0 and won_side == Shape.X) or (i % 2 == 1 and won_side == Shape.O):
				histo[winr] += 1
				return winr
			histo[loser] += 1
			return loser
		#A list of (input array, reward, sampled action).
		boards = itertools.chain(*[[(board[0], get_reward(gw[1], i), board[1]) for i, board in enumerate(gw[0])] for gw in plays])
		#Classify them by the action taken
		action_arrs = [[] for _ in range(self.node_nums[-1])]
		for board in boards:
			action_arrs[board[2]].append(board[:2])
		for action in range(self.node_nums[-1]):
			if len(action_arrs[action]) != 0:
				br = list(zip(*action_arrs[action]))
				session.run(self.trainer, feed_dict={self.layers[0]: br[0], self.reward_t: br[1], self.sampled_t: action})

	@needs_session
	def train(self, dimen, winLen, rewards, batch_size=100, cycles=1000, stops=100, interactive=False, save_path=None, session=None):
		"""
		Trains the network with policy gradients.
		:param dimen: A 2-tuple: (horSize, verSize) of the game.
		:param winLen: The k in m,n,k game.
		:param rewards: A tuple (reward for winning, reward for losing, reward for drawing). The 3 must be different from each other.
		:param batch_size: The cycles argument in selfplay.
		:param cycles: How many batches of games to play.
		:param stops: At every (stops)th cycle, the runner will print out some statistics.
		:param interactive: If True, will ask for confirmation to keep training every (stops)th cycle.
		:param save_path: If not None, will save the weights to save_path every (stops)th cycle.
		:param session: The tf.Session to use. If not specified, a new Session is created.
		"""
		#writer = tf.summary.FileWriter("logs/")
		saver = tf.train.Saver()
		histo = {rewards[0]: 0, rewards[1]: 0, rewards[2]: 0}
		performances = []
		ai = RandomAi()
		prepareplt()
		started = time.time()
		for i in range(cycles):
			if i % stops == (stops - 1):
				if save_path is not None:
					saver.save(session, save_path)
				print(f"{i + 1}th cycle, {time.time() - started} seconds have elapsed since the training started")
				print(histo)
				performances.append(evaluate_player(lambda g: ai.play(g), lambda g: self.play(g, to_dense_input(g.array), session=session), rules=(dimen[0], dimen[1], winLen)))
				print(performances[-1])
				shownonblock(performances, labels=["1st won", "2nd won", "draw"])
				if interactive and input("Continue training?(y/n): ") == 'n':
					break
			self._train_cycle(dimen, winLen, batch_size, rewards, histo, session)
		print(histo)
		return histo, performances