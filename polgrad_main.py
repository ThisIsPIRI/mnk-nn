import numpy as np
import tensorflow as tf

from mnk import MnkGame
from polgrad_tf import PolgradRunnerTf
from mnkais import FillerAi, RandomAi

def playGame(first_player, second_player, game=None, print_board=False):
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

def main():
	LOAD = True
	WEIGHT_PATH = "weight/weight.ckpt"
	DIMEN = (3, 3); BOARD_SIZE = DIMEN[0] * DIMEN[1]
	targetAi = FillerAi()

	runner = PolgradRunnerTf([BOARD_SIZE, BOARD_SIZE * 2, 20, BOARD_SIZE], [tf.nn.relu, tf.nn.relu, tf.nn.softmax])
	saver = tf.train.Saver()
	with tf.Session() as sess:
		if LOAD:
			tf.train.Saver().restore(sess, WEIGHT_PATH)
		else:
			sess.run(tf.global_variables_initializer())
		if input("train the network?(y/n): ") == "y":
			runner.train(dimen=DIMEN, winLen=3, batch_size=50, cycles=100, session=sess)
		histo = {"O won": 0, "X won": 0, "draw": 0}
		for i in range(100):
			histo[playGame(lambda g: runner.play(g, session=sess), lambda g: targetAi.play(g))] += 1
		print(histo)
		playGame(lambda g: runner.play(g, session=sess), lambda g: eval(input()), print_board=True)
		saver.save(sess, WEIGHT_PATH)

if __name__ == "__main__":
	main()
