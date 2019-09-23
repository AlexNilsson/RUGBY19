""" DATA """
SCORE_DECAY_FACTOR = 0.5 # How much a goal from the first year is worth
USE_TEST_DATA = False

""" NETWORK ARCHITECTURE """
INPUT_SHAPE = 40 # = 2 * number of unique teams in train/test data
DROPOUT = 0.4
LEAKY_RELU_ALPHA = 0.1

""" TRAINING """
LEARNING_RATE = 0.00005
EPOCHS = 500
BATCH_SIZE = 100

WRONG_WINNER_PENALTY = 10
