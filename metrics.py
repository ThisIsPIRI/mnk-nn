import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnk import MnkGame

def play_game(first_player, second_player, game=None, print_board=False):
	if game is None:
		game = MnkGame(3, 3, winStreak=3)
	for i in range(game.horSize * game.verSize):
		if i % 2 == 0:
			decision = first_player(game)
		else:
			decision = second_player(game)
		game.place(decision)
		if print_board:
			print(np.array(game.array))
		if game.checkWin(decision):
			break
	if len(game.history) == 9:
		return "draw"
	elif len(game.history) % 2 == 0:
		return "O won"
	else:
		return "X won"

def evaluate_player(first_player, second_player):
	histo = {"O won": 0, "X won": 0, "draw": 0}
	for i in range(100):
		histo[play_game(first_player, second_player)] += 1
	return histo

def prepareplt():
	plt.ion()
	plt.show()

def shownonblock(data, labels=None):
	if isinstance(data, list) and isinstance(data[0], dict) and labels is not None:
		print(data)
		data = pd.DataFrame(data)
		data = [data[l] for l in labels]
		print(data)
	plt.clf()
	if labels is None:
		for i in range(len(data)):
			plt.plot(data[i])
	else:
		for i in range(len(data)):
			plt.plot(data[i], label=labels[i])

	#if plt.gca().get_legend() is not None:
		#plt.gca().get_legend().remove()
	plt.legend()
	plt.draw()
	plt.pause(0.001)
	#plt.show(block=False)