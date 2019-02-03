import bisect
import itertools
import os.path
import random
import tensorflow as tf
import time

from metrics import evaluate_player, prepareplt, shownonblock
from mnk import MnkGame, Shape
from mnkais import RandomAi
from mnkutil import choose_cell_weighted, to_dense_input, to_dense_index, needs_session

class PolgradRunnerTf:
	weight_format = "{}th_cycle.ckpt"
	"""The tensors. The first element is the input placeholder and the last one the output layer."""
	def __init__(self, node_nums, activations):
		"""
		Builds the graph.
		:param node_nums: A list containing the number of neurons in each layer from input to output.
		:param activations: The list of Tensorflow activation functions to be used in each connections.
		"""
		self.layers = []
		self.node_nums = node_nums
		self.layers.append(tf.placeholder(tf.float32, shape=(None, node_nums[0])))
		for i in range(len(node_nums))[1:]:
			self.layers.append(tf.layers.dense(self.layers[i - 1], node_nums[i], activations[i - 1]))
		self.reward_t = tf.placeholder(tf.float32)
		self.sampled_t = tf.placeholder(tf.int32)
		self.onehot_sampled_t = tf.one_hot(self.sampled_t, self.node_nums[-1], dtype=tf.bool, on_value=True, off_value=False)
		self.onehot_sampled_t.set_shape((None, None))
		self.loss_t = -tf.reduce_mean(self.reward_t * tf.boolean_mask(tf.log(tf.clip_by_value(self.layers[-1], 1e-8, 1.0)), self.onehot_sampled_t))
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

	def play(self, game, board=None, session=None, return_probs=False):
		"""
		Determines the best cell to play on.
		:param game: An MnkGame.
		:param board: A 1d np.ndarray. Fed into the network instead of game.array if supplied.
		:param session: The tf.Session to use. If not supplied, a new Session is created.
		:param return_probs: If True, the probabilities will be returned along with the coordinates.
		:return: The chosen cell's (x, y) coordinates. ((x, y), probabilities list) if returnProbs.
		"""
		if board is None:
			input_d = to_dense_input(game.array)
		else:
			input_d = board
		return choose_cell_weighted(game, self.forward_propagate(input_d.reshape(-1, self.node_nums[0]), session)[0], return_probs) #Reshape to 2d before passing it to the network

	@needs_session
	def selfplay(self, rules, cycles=100, teacher=None, play_first=False, epsilon=0, session=None):
		"""
		Plays a game against itself or a teacher cycles times and returns a list containing all states encountered in all games except the terminal states.
		:param rules: A tuple: (horSize, verSize, winlen) of the game.
		:param cycles: How many games to play.
		:param teacher: The teacher AI function. If None, will play against self.
		:param play_first: Whether to play first or second against the teacher.
		:param epsilon: The probability of playing a purely random move.
		:param session: The tf.Session to use. If not specified, a new Session is created.
		:return: A list of tuples (list, winning mnk.Shape), where the list contains tuples (board state, action taken) for one game.
		"""
		result = []
		randai = RandomAi()
		game = MnkGame(rules[0], rules[1], rules[2])
		for i in range(cycles):
			boards = []
			winner = 0
			for j in range(game.horSize * game.verSize):
				input_d = to_dense_input(game.array)
				if teacher is not None and (j % 2 == 0 ^ play_first):
					decision = teacher(game)
					boards.append(-1) #The teacher's actions aren't needed, but append something so finding out which moves are from the winning/losing side is easier
				else:
					if random.random() < epsilon:
						decision = randai.play(game)
					else:
						decision = self.play(game, board=input_d, session=session)
					boards.append((input_d, to_dense_index(game, decision)))
				game.place(decision)
				if game.checkWin(decision):
					winner = Shape.O if j % 2 == 1 else Shape.X
					break
			result.append((boards, winner)) #-1: 1st won, 1: 2nd won, 0: draw
			game.initialize()
		return result

	@needs_session
	def _train_cycle(self, rules, batch_size, rewards, histo, teacher=None, play_first=False, epsilon=0, session=None):
		"""
		See train() and selfplay() for parameter documentations.
		:param histo: The histogram dict {rewards[0]: int, rewards[1]: int, rewards[2]: int}. The numbers will be incremented.
		"""
		winr, loser, drawr = rewards
		plays = self.selfplay(rules, batch_size, teacher=teacher, play_first=play_first, epsilon=epsilon, session=session)
		def get_reward(won_side, i): #TODO: Occurrences of nonzero rewards should only be counted once
			if won_side == 0:
				histo[drawr] += 1
				return drawr
			elif (i % 2 == 0 and won_side == Shape.X) or (i % 2 == 1 and won_side == Shape.O):
				histo[winr] += 1
				return winr
			histo[loser] += 1
			return loser
		#A list of (input array, reward, sampled action). Ignores 0-reward and teachers' actions and discounts rewards.
		boards = itertools.chain(*[[(board[0], get_reward(gw[1], i) / (2 ** (len(gw[0]) - i - 1)), board[1]) for i, board in enumerate(gw[0]) if board != -1 and get_reward(gw[1], i) != 0] for gw in plays])
		bra = list(zip(*boards)) #[list of input arrays, list of rewards, list of sampled actions]
		if len(bra) > 0: #bra can be empty if the runner lost all games against a teacher
			session.run(self.trainer, feed_dict={self.layers[0]: bra[0], self.reward_t: bra[1], self.sampled_t: bra[2]})

	@needs_session
	def train(self, rules, rewards, batch_size=100, cycles=1000, cyclestart=0, stops=100, teachers=None, epsilon=0, interactive=False, save_path=None, session=None):
		"""
		Trains the network with policy gradients.
		:param rules: A tuple: (horSize, verSize, winlen) of the game.
		:param rewards: A tuple (reward for winning, reward for losing, reward for drawing). The 3 must be different from each other.
		:param batch_size: The cycles argument in selfplay.
		:param cycles: How many batches of games to play.
		:param cyclestart: The number at which the cycle counts start.
		:param stops: At every (stops)th cycle, the runner will print out some statistics.
		:param teachers: A list of tuples (teacher function, frequency int, probability of the teacher playing first). If None, will only play against itself.
		The teachers will be cycled in the order they are stored, each cycle using each teacher (frequency int) times.
		The saved model from the last stop will be used as a teacher if a literal "SELF" is passed as a teacher function. The save_path must be specified in this case.
		:param epsilon: The probability of playing a purely random move.
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
		if teachers is not None:
			teachers = list([list(z) for z in zip(*teachers)])
			teachersum = sum(teachers[1])
			teachers[1] = [sum(teachers[1][:i]) for i in range(1, len(teachers[1]) + 1)] #Accumulate
			if "SELF" in teachers[0]:
				frozen_sess = tf.Session()
				teachers[0][teachers[0].index("SELF")] = lambda g: self.play(g, board=to_dense_input(g.array), session=frozen_sess)
			else:
				frozen_sess = None
		started = time.time()
		for i in range(cyclestart, cyclestart + cycles):
			if i % stops == (stops - 1) or i == cyclestart:
				if save_path is not None:
					saver.save(session, os.path.join(save_path, PolgradRunnerTf.weight_format.format(i + 1)))
				if frozen_sess is not None:
					saver.restore(frozen_sess, os.path.join(save_path, PolgradRunnerTf.weight_format.format(i + 1)))
				print(f"{i + 1}({i + 1 - cyclestart})th cycle, {time.time() - started} seconds have elapsed since the training started")
				print(histo)
				performances.append(evaluate_player(lambda g: ai.play(g), lambda g: self.play(g, to_dense_input(g.array), session=session), rules=rules))
				print(performances[-1])
				shownonblock(performances, labels=["1st won", "2nd won", "draw"])
				if interactive and input("Continue training?(y/n): ") == 'n':
					break
			if teachers is not None:
				teacheridx = bisect.bisect(teachers[1], i % teachersum)
				teacher = teachers[0][teacheridx]
				play_first = teachers[2][teacheridx] < random.random()
			else:
				teacher = None
				play_first = False
			self._train_cycle(rules, batch_size, rewards, histo, teacher=teacher, play_first=play_first, epsilon=epsilon, session=session)
		print(histo)
		return histo, performances
