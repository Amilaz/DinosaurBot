import numpy as np

DO_NOTHING = np.array([1, 0])

#game parameters
ACTIONS = 2
INPUT_SIZE = (32, 72, 5)
SCORE_INPUT_SIZE = (1, )
SCORE_RATIO = 100
IMAGE_SIZE = (72, 32)
DALAY = 10
GAMMA = 0.99 # decay rate of past observations original 0.99
OBSERVATION = 100 # timesteps to observe before training
GAME_OBSERVATION = 10 # game to observe before training
EXPLORE = 10000
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.01 # starting value of epsilon
REPLAY_MEMORY = 100 # number of previous transitions to remember
BATCH = 128 # size of minibatch
TRAINING_LOG_INTERVAL = 50