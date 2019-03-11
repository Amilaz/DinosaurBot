from utils import const
import collections
import numpy as np
import random
from IPython.display import clear_output
import time
from termcolor import colored


def train_batch(model, sample_data_list):

    ##print("----------  Batch ---------- ")
    for sample_data in sample_data_list:
        image_sample = sample_data[0]['frame_t_0']
        inputs_frame = np.zeros(
            (len(sample_data), image_sample.shape[0], image_sample.shape[1], image_sample.shape[2]))
        inputs_score = np.zeros(
            (inputs_frame.shape[0], const.SCORE_INPUT_SIZE[0]))
        targets = np.zeros((inputs_frame.shape[0], const.ACTIONS))
        loss = 0

        for i in range(0, len(sample_data)):
            state_t = sample_data[i]["frame_t_0"]    # 4D stack of images
            action_t = sample_data[i]["action_index"]  # This is action index
            # reward at state_t due to action_t
            reward_t = sample_data[i]["reward_t_1"]
            state_t_1 = sample_data[i]["frame_t_1"]  # next state
            # wheather the agent died or survided due the action
            is_over = sample_data[i]["is_over"]
            score_t_0 = sample_data[i]["score_t_0"]
            score_t_1 = sample_data[i]["score_t_1"]
            score_t_0 = [score_t_0[1] - score_t_0[0]]
            score_t_1 = [score_t_1[1] - score_t_1[0]]
            score_t_0 = np.array([x / const.SCORE_RATIO for x in score_t_0])
            score_t_1 = np.array([x / const.SCORE_RATIO for x in score_t_1])
            inputs_frame[i:i + 1] = state_t
            inputs_score[i:i + 1] = score_t_0
            targets[i] = model.predict([np.reshape(state_t, (1, state_t.shape[0], state_t.shape[1], state_t.shape[2])),
                                        np.reshape(score_t_0, (1, score_t_0.shape[0]))])  # predicted q values
            q_sa = model.predict([np.reshape(state_t_1, (1, state_t_1.shape[0], state_t_1.shape[1], state_t_1.shape[2])),
                                  np.reshape(score_t_1, (1, score_t_1.shape[0]))])  # predict q values for next step
            if is_over:
                # if terminated, only equals reward
                targets[i, action_t] = reward_t / 2
            else:
                targets[i, action_t] = reward_t + \
                    const.GAMMA * np.max(targets[i])

            loss += model.train_on_batch([inputs_frame, inputs_score], targets)

    return model, q_sa, loss


def create_sample_data(game_stack):
    sample_data = []
    game_stack_copy = game_stack.copy()
    game_stack_copy = sorted(game_stack_copy, key=len, reverse=True)
    random_game = random.sample(game_stack, k=min(len(game_stack), 100))
    sample_data_index = 0
    # Best game
    for frame_stack in game_stack_copy:
        for frame in frame_stack:
            if len(sample_data) == sample_data_index:
                sample_data.append([])
            if len(sample_data[sample_data_index]) >= const.BATCH:
                if len(sample_data) == 20:
                    break
                else:
                    sample_data_index += 1
            else:
                sample_data[sample_data_index].append(frame)
    # Random
    for frame_stack in random_game:
        for frame in frame_stack:
            if len(sample_data) == sample_data_index:
                sample_data.append([])
            if len(sample_data[sample_data_index]) >= const.BATCH:
                if len(sample_data) == 40:
                    break
                else:
                    sample_data_index += 1
            else:
                sample_data[sample_data_index].append(frame)
    return sample_data


def train_model(model, game_state, model_exist=False):
    # Initial variable
    frame_image_list = []  # List for store frame stack
    epsilon = const.INITIAL_EPSILON  # Initial epsilon
    in_memory_running = []  # Game store in memory
    do_nothing = const.DO_NOTHING  # Do nothing
    t = 0  # Process index
    delay = 0  # Delay counting for training model
    game_index = 0  # Game index
    game_index_d = 0

    game_state.initial_image_state()  # Start game
    image_t_0, reward_t_0, score_t_0, _ = game_state.get_state(
        do_nothing)  # Get first frame of game
    frame_image_list = [image_t_0] * \
        (const.INPUT_SIZE[2])  # Create 5 frame list
    # Combine 5 frame into 1 image as input of model
    stack_frame_t_0 = np.stack(tuple(frame_image_list), axis=2)
    score_stack_t_0 = [score_t_0, score_t_0]

    while True:  # Forever running
        loss = 0
        q_sa = 0
        action_index = 0
        reward_t_0 = 0
        action_t = np.zeros([const.ACTIONS])
        action_source = ""

        if  random.random() <= epsilon or ( not model_exist and (t <= const.OBSERVATION or game_index < const.GAME_OBSERVATION)):
            action_source = "Explore"
            action_index = random.randrange(const.ACTIONS)
            action_t[action_index] = 1
        else:
            action_source = "Exploit"
            speed = [score_stack_t_0[1] - score_stack_t_0[0]]
            q = model.predict([np.reshape(
                stack_frame_t_0, (1, stack_frame_t_0.shape[0], stack_frame_t_0.shape[1], stack_frame_t_0.shape[2])),
                np.reshape(np.array(speed), (1, np.array(speed).shape[0]))])
            action_index = np.argmax(q)
            action_t[action_index] = 1

        # Reduced the exploration epsilon
        if epsilon > const.FINAL_EPSILON and t > const.OBSERVATION:
            epsilon -= (const.INITIAL_EPSILON -
                        const.FINAL_EPSILON) / const.EXPLORE

        image_t_1, reward_t_1, score_t_1, is_over_t_1 = game_state.get_state(
            action_t, image_sample=(random.random() < const.INITIAL_EPSILON))
        frame_image_list.pop(0)
        frame_image_list.append(image_t_1)
        stack_frame_t_1 = np.stack(tuple(frame_image_list), axis=2)

        score_stack_t_1 = [score_t_0, score_t_1]

        training_data = {
            "game_index": game_index,
            "time_index": t,
            "frame_t_0": stack_frame_t_0,
            "frame_t_1": stack_frame_t_1,
            "reward_t_0": reward_t_0,
            "reward_t_1": reward_t_1,
            "score_t_0": score_stack_t_0,
            "score_t_1": score_stack_t_1,
            "action_index": action_index,
            "is_over": is_over_t_1
        }

        if len(in_memory_running) == game_index_d:
            in_memory_running.append([])
        in_memory_running[game_index_d].append(training_data)
        # Remove old game if stack in list more than REPLAY_MEMORY
        if len(in_memory_running) > const.REPLAY_MEMORY:
            in_memory_running.pop(0)
            game_index_d -= 1
        # Update game index and delay counting
        if is_over_t_1:
            delay -= 1
            game_index += 1
            game_index_d += 1
            # Recreate frame stack if game over
            image_t_0, reward_t_0, score_t_0, _ = game_state.get_state(
                do_nothing)
            frame_image_list = [image_t_0] * const.INPUT_SIZE[2]
            stack_frame_t_1 = np.stack(tuple(frame_image_list), axis=2)
            score_stack_t_0 = [score_t_0, score_t_0]

        if t > const.OBSERVATION and is_over_t_1 and delay <= 0:
            # Reset delay for training model
            delay = const.DALAY
            # Pause game for training model
            game_state.pause_game()
            time.sleep(5)
            sample_data = create_sample_data(in_memory_running)
            model, q_sa, loss = train_batch(model, sample_data)
            game_state.resume_game()

        if t % const.TRAINING_LOG_INTERVAL == 0:
            clear_output()
        if action_source == "Explore":
            print('\033[91m', "GAME", game_index, "/TIMESTEP", t, "/EPSILON", epsilon, "/ACTION TYPE", action_source,
                  "/ACTION", action_index, "/REWARD", reward_t_1, "/IS OVER", is_over_t_1, "/LOSS", loss, "/Q", q_sa, '\033[0m')
        else:
            print('\033[92m', "GAME", game_index, "/TIMESTEP", t, "/EPSILON", epsilon, "/ACTION TYPE", action_source,
                  "/ACTION", action_index, "/REWARD", reward_t_1, "/IS OVER", is_over_t_1, "/LOSS", loss, "/Q", q_sa, '\033[0m')

        if not is_over_t_1:
            reward_t_0 = reward_t_1
            score_t_0 = score_t_1
            score_stack_t_0 = score_stack_t_1

        stack_frame_t_0 = stack_frame_t_1
        t += 1
