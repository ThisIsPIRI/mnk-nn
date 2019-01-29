import tensorflow as tf

from mnk import MnkGame
from mnkutil import needs_session
from polgrad_tf import PolgradRunnerTf
from mnkais import FillerAi, RandomAi

@needs_session
def testAgainst(runner, ai, game=None, session=None):
	if game is None:
		game = MnkGame(3, 3)
	for i in range(game.horSize * game.verSize):
		if i % 2 == 0:
			game.place(ai.play(game))
		else:
			game.place(runner.play(game, session=session))
	if len(game.history) == 9:
		print("draw")
	elif len(game.history) % 2 == 0:
		print("O won")
	else:
		print("X won")

def main():
	LOAD = False
	WEIGHT_PATH = "weight/weight.ckpt"
	DIMEN = (3, 3); BOARD_SIZE = DIMEN[0] * DIMEN[1]

	training_ordered = input("train the network?(y/n): ") == "y"

	runner = PolgradRunnerTf([BOARD_SIZE, BOARD_SIZE * 2, 20, BOARD_SIZE], [tf.nn.relu, tf.nn.relu, tf.nn.softmax])
	saver = tf.train.Saver()
	with tf.Session() as sess:
		if LOAD:
			tf.train.Saver().restore(sess, WEIGHT_PATH)
		else:
			sess.run(tf.global_variables_initializer())
		if training_ordered:
			runner.train(dimen=DIMEN, winLen=3, batch_size=100, cycles=100, session=sess)
		for i in range(100):
			testAgainst(runner, FillerAi(), session=sess)
		saver.save(sess, WEIGHT_PATH)

if __name__ == "__main__":
	main()
