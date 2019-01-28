import bisect
import functools
import inspect
import numpy as np
import random
import tensorflow as tf

from mnk import MnkGame, Shape

def choose_weighted(probs):
	"""
	Randomly chooses an integer i from range(len(probs)) with probs[i] as its probability.
	:param probs: The 1d np.ndarray of probabilities.
	:return: The chosen index.
	"""
	pairs = sorted([(i, probs[i]) for i in range(len(probs))], key=lambda t: t[1]) #Sort the probabilities and pair them with their indices
	pairs = [(gamecell[0], sum([t[1] for t in pairs[:loopidx + 1]])) for loopidx, gamecell in enumerate(pairs)] #Accumulate the probabilities
	return pairs[bisect.bisect([p[1] for p in pairs], random.random())][0] #Find the chosen cell's coordinates and return it

def choose_cell_weighted(game, probs):
	"""
	Chooses a cell from the game with probs[i] as the probability for cell (i % horSize, i / verSize).
	The probabilities for non-empty cells will be changed to 0 and the rest normalized to sum to unity.
	:param game: The MnkGame to use.
	:param probs: The 1d np.ndarray of probabilities.
	:return: The coordinates of the chosen cell in (x, y)
	"""
	idx = [to_game_index(game, i) for i in range(len(probs))] #Cache 2d indices
	probs = [0 if game.array[idx[i][1]][idx[i][0]] != Shape.N else probs[i] for i in range(len(probs))] #Never choose a filled cell
	probs = [i / sum(probs) for i in probs] #Normalize to sum to unity
	return to_game_index(game, np.random.choice(len(probs), p=probs))

def find_max_valid(game, out):
	"""Pass the game and raw output from the output layer. Returns None if there isn't any empty cell."""
	for i in np.argsort(out)[::-1]:
		if game.array[int(i / game.verSize)][int(i % game.horSize)] == Shape.N:
			return to_game_index(game, i)
	return None

def to_game_index(game, data):
	"""
	Converts a flattened index to 2d index.
	:param game: The MnkGame or a tuple containing (horSize, verSize).
	:param data: The index.
	:return: (x, y)
	"""
	if isinstance(game, MnkGame):
		return int(data % game.horSize), int(data / game.verSize)
	else:
		return int(data % game[0]), int(data / game[1])

def to_dense_input(array):
	"""
	Converts a multidimensional python list to 1d np.ndarray in row-major(C-style) order.
	:param array: The
	:return: The flattened np.ndarray.
	"""
	return np.array(array).flatten('C')

def reverseboard(array):
	return np.array([-x for x in array])

def needs_session(func):
	"""
	Creates a Session if one isn't given, runs global_variables_initializer() on it, passes it to the decoratee and closes the Session it created, if any.
	The Session argument's name(both when being passed into the wrapper and the wrapee) must be 'session' and its default value None.
	:param func: The function to decorate. It must accept a 'session' as a keyword argument
	and be able to accept positional arguments from kwargs(e.g. for a function f(a, b=123), f(a=1, b=2) must be a valid call).
	:return: The decorated function.
	"""
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		created_session = False
		kwargs = inspect.signature(func).bind(*args, **kwargs).arguments
		if "session" not in kwargs or kwargs["session"] is None:
			kwargs["session"] = tf.Session()
			kwargs["session"].run(tf.global_variables_initializer())
			created_session = True
		result = func(**kwargs)
		if created_session:
			kwargs["session"].close()
		return result
	return wrapper
