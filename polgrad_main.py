import os.path
import tensorflow as tf

from polgrad_tf import PolgradRunnerTf
from metrics import evaluate_player, play_game
from mnkais import EmacsGomokuAi, RandomAi, Human

def main():
	LOAD = True
	SAVE_TFLITE = True
	WEIGHT_PATH = "weights/weight_bn_ln/"
	cyctorun = 2000
	try:
		with open(os.path.join(WEIGHT_PATH, "cycran.txt")) as cycfile: cycran = int(cycfile.read())
	except FileNotFoundError:
		cycran = 0
		with open(os.path.join(WEIGHT_PATH, "cycran.txt"), 'w') as cycfile: cycfile.write("0")
	rules = (3, 3, 3); BOARD_SIZE = rules[0] * rules[1]
	randai = RandomAi()
	emacai = EmacsGomokuAi()
	human = Human()
	teachers = [(randai.play, 2, 0.6), (emacai.play, 2, 0.6), ("SELF", 2, 0.6), (None, 6, 0.5)]
	runner = PolgradRunnerTf([BOARD_SIZE * 2, BOARD_SIZE * 2, BOARD_SIZE, BOARD_SIZE * 2, BOARD_SIZE * 2, BOARD_SIZE], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.softmax])
	saver = tf.train.Saver()
	with tf.Session() as sess:
		#Load or initialize the model
		if LOAD:
			tf.train.Saver().restore(sess, os.path.join(WEIGHT_PATH, PolgradRunnerTf.weight_format.format(cycran)))
		else:
			sess.run(tf.global_variables_initializer())
		#Train the model and save the weights
		if input("train the network?(y/n): ") == "y":
			runner.train(rules=rules, rewards=(1, -0.5, 0.3), batch_size=100, cycles=cyctorun, cyclestart=cycran, stops=100, teachers=teachers, epsilon=0.05, interactive=False, save_path=WEIGHT_PATH, session=sess)
			saver.save(sess, os.path.join(WEIGHT_PATH, PolgradRunnerTf.weight_format.format(cyctorun + cycran)))
			with open(os.path.join(WEIGHT_PATH, "cycran.txt"), 'w') as cycfile: cycfile.write(str(cyctorun + cycran))
		#Export .tflite file
		if SAVE_TFLITE:
			with open("model.tflite", "wb") as f:
				f.write(tf.lite.TFLiteConverter.from_session(sess, [runner.layers[0]], [runner.layers[-1]]).convert())
		#Test the model against other AIs and humans
		print(evaluate_player(lambda g: runner.play(g, session=sess), randai.play, rules=rules, games=100))
		print(evaluate_player(randai.play, lambda g: runner.play(g, session=sess), rules=rules, games=100))
		print(evaluate_player(lambda g: runner.play(g, session=sess), emacai.play, rules=rules, games=1))
		print(evaluate_player(emacai.play, lambda g: runner.play(g, session=sess), rules=rules, games=1))
		print(play_game(human.play, lambda g: runner.play(g, session=sess, return_probs=True), rules=rules, print_board=True))

if __name__ == "__main__":
	main()
