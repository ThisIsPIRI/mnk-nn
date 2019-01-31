import tensorflow as tf

from polgrad_tf import PolgradRunnerTf
from metrics import evaluate_player, play_game
from mnkais import FillerAi, RandomAi
from mnkutil import to_dense_input

def main():
	LOAD = True
	WEIGHT_PATH = "weight_bn_sn/weight_bn_sn.ckpt"
	DIMEN = (3, 3); BOARD_SIZE = DIMEN[0] * DIMEN[1]
	targetai = RandomAi()

	runner = PolgradRunnerTf([BOARD_SIZE * 2, BOARD_SIZE * 2, BOARD_SIZE * 2, BOARD_SIZE], [tf.nn.relu, tf.nn.relu, tf.nn.softmax])
	saver = tf.train.Saver()
	with tf.Session() as sess:
		if LOAD:
			tf.train.Saver().restore(sess, WEIGHT_PATH)
		else:
			sess.run(tf.global_variables_initializer())
		if input("train the network?(y/n): ") == "y":
			runner.train(dimen=DIMEN, winLen=3, batch_size=100, cycles=1000, stops=100, session=sess)
		print(evaluate_player(lambda g: runner.play(g, board=to_dense_input(g.array), session=sess), lambda g: targetai.play(g), rules=(3, 3, 3)))
		saver.save(sess, WEIGHT_PATH)
		print(play_game(lambda g: runner.play(g, board=to_dense_input(g.array), session=sess), lambda g: eval(input()), rules=(3, 3, 3), print_board=True))

if __name__ == "__main__":
	main()
