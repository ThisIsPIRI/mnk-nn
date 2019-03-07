import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnk import MnkGame

def play_game(first_player, second_player, rules, print_board=False):
	"""
	Lets the players play an MnkGame. If a player returns a tuple ((x, y), justification), will print the justification and use (x, y) for the coordinates.

	:param first_player: The first player function. Must accept an MnkGame as the first positional argument.
	:param second_player: The second player function. Must accept an MnkGame as the first positional argument.
	:param rules: A tuple (horSize, verSize, winLen)
	:param print_board: Whether to print the board at the end of each turn.
	:return: "1st won" if first_player won, "2nd won" if second_player won and "draw" if they drew.
	"""
	game = MnkGame(*rules)
	for i in range(game.horSize * game.verSize):
		if i % 2 == 0:
			decision = first_player(game)
		else:
			decision = second_player(game)
		if not isinstance(decision[1], int):
			print(decision[1])
			decision = decision[0]
		game.place(decision)
		if print_board:
			print(np.array(game.array))
		if game.checkWin(decision):
			if len(game.history) % 2 == 0:
				return "2nd won"
			else:
				return "1st won"
	return "draw"

def evaluate_player(first_player, second_player, rules, games=100):
	"""
	Pits the players against each other games times and returns how many times they won, lost or drew. See play_game for parameter documentations.

	:param games: How many games to play.
	:return: A dict {"1st won": int, "2nd won": int, "draw": int}
	"""
	histo = {"1st won": 0, "2nd won": 0, "draw": 0}
	for i in range(games):
		histo[play_game(first_player, second_player, rules)] += 1
	return histo

def prepareplt():
	"""Call before calling shownonblock."""
	plt.ion()
	plt.show()

def shownonblock(data, labels=None):
	"""
	Plots the individual elements in data as separate lines and shows them with pyplot.

	:param data: The list of lines to plot. Can be a list of dicts if the keys are supplied as labels.
	:param labels: The labels for each line.
	"""
	if isinstance(data, list) and isinstance(data[0], dict) and labels is not None:
		data = pd.DataFrame(data)
		data = [data[l] for l in labels]

	plt.clf()

	if labels is None:
		for i in range(len(data)):
			plt.plot(data[i])
	else:
		for i in range(len(data)):
			plt.plot(data[i], label=labels[i])
	plt.legend()
	plt.draw()
	plt.pause(0.001)
