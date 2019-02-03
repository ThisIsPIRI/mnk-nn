import tensorflow as tf

from polgrad_tf import PolgradRunnerTf
from metrics import evaluate_player, play_game
from mnkais import EmacsGomokuAi, FillerAi, RandomAi, Human
from mnkutil import to_dense_input

def main():
	LOAD = False
	WEIGHT_PATH = "weights/weight_bn_ln/weight_bn_ln.ckpt"
	rules = (3, 3, 3); BOARD_SIZE = rules[0] * rules[1]
	randai = RandomAi(); emacai = EmacsGomokuAi()
	human = Human()
	runner = PolgradRunnerTf([BOARD_SIZE * 2, BOARD_SIZE * 2, BOARD_SIZE, BOARD_SIZE * 2, BOARD_SIZE * 2, BOARD_SIZE], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.softmax])
	saver = tf.train.Saver()
	with tf.Session() as sess:
		if LOAD:
			tf.train.Saver().restore(sess, WEIGHT_PATH)
		else:
			sess.run(tf.global_variables_initializer())
		if input("train the network?(y/n): ") == "y":
			runner.train(rules=rules, rewards=(1, 0, 0.3), batch_size=100, cycles=2000, stops=100, interactive=False, save_path=WEIGHT_PATH, session=sess)
			saver.save(sess, WEIGHT_PATH)
		print(evaluate_player(lambda g: runner.play(g, board=to_dense_input(g.array), session=sess), randai.play, rules=rules, games=200))
		print(evaluate_player(randai.play, lambda g: runner.play(g, board=to_dense_input(g.array), session=sess), rules=rules, games=200))
		#runner.train(rules=rules, rewards=(1, -1, 0.3), batch_size=2, cycles=2, stops=100, teachers=[(human.play, 1, 1)], interactive=True, save_path=WEIGHT_PATH, session=sess)
		#saver.save(sess, WEIGHT_PATH)
		#print(play_game(lambda g: runner.play(g, board=to_dense_input(g.array), session=sess, returnProbs=True), lambda g: eval(input()), rules=rules, print_board=True))
		#print(play_game(lambda g: eval(input()), lambda g: runner.play(g, board=to_dense_input(g.array), session=sess, returnProbs=True), rules=rules, print_board=True))

if __name__ == "__main__":
	main()
