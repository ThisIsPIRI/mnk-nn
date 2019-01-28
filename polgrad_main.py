import tensorflow as tf

from polgrad_tf import PolgradRunnerTf

def main():
	LOAD = False
	WEIGHT_PATH = "weight/weight.ckpt"
	DIMEN = (3, 3); BOARD_SIZE = DIMEN[0] * DIMEN[1]

	runner = PolgradRunnerTf([BOARD_SIZE, BOARD_SIZE * 2, 20, BOARD_SIZE], [tf.nn.relu, tf.nn.relu, tf.nn.softmax])
	saver = tf.train.Saver()
	with tf.Session() as sess:
		if LOAD:
			tf.train.Saver().restore(sess, WEIGHT_PATH)
		else:
			sess.run(tf.global_variables_initializer())
		runner.train(dimen=DIMEN, winLen=3, batch_size=2, cycles=2, session=sess)
		saver.save(sess, WEIGHT_PATH)

if __name__ == "__main__":
	main()
