import numpy as np
import tensorflow as tf
import random
import time
import os
import argparse
import json
import matplotlib.pyplot as plt

from datetime import datetime
from shutil import rmtree

from fog_env import Offload
from RL_brain import DeepQNetwork
from utils import plot_graphs

np.set_printoptions(threshold=np.inf)


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def reward_fun(delay, max_delay, unfinish_indi):
    # still use reward, but use the negative value
    if unfinish_indi:
        reward = - max_delay * 2
    else:
        reward = - delay

    return reward


def train(env, iot_RL_list, num_episodes, show=False, training_dir=None):
    start_time = time.time()

    RL_step = 0

    episode_rewards = list()
    episode_dropped = list()
    episode_delay = list()

    fig, axs = plt.subplots(3, figsize=(10, 12), sharex=True)

    for episode in range(num_episodes):
        # BITRATE ARRIVAL
        bitarrive = np.random.uniform(env.min_bit_arrive, env.max_bit_arrive,
                                      size=[env.n_time, env.n_iot])
        task_prob = env.task_arrive_prob
        bitarrive = bitarrive * (
            np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < task_prob)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_iot])

        # rewards_dict = {d: [] for d in range(env.n_iot)}
        rewards_list = list()
        dropped_list = list()
        delay_list = list()

        # ============================================================================= #
        # ========================================= DRL =============================== #
        # ============================================================================= #

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for iot_index in range(env.n_iot):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_iot])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive)

        # TRAIN DRL
        while True:

            # PERFORM ACTION
            action_all = np.zeros([env.n_iot])
            for iot_index in range(env.n_iot):

                observation = np.squeeze(observation_all[iot_index, :])

                if np.sum(observation) == 0:
                    # if there is no task, action = 0 (also need to be stored)
                    action_all[iot_index] = 0
                else:
                    action_all[iot_index] = \
                        iot_RL_list[iot_index].choose_action(observation)

                if observation[0] != 0:
                    iot_RL_list[iot_index].do_store_action(episode, env.time_count,
                                                           action_all[iot_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observation_all_, lstm_state_all_, done = env.step(action_all)

            # should store this information in EACH time slot
            for iot_index in range(env.n_iot):
                iot_RL_list[iot_index].update_lstm(lstm_state_all_[iot_index, :])

            process_delay = env.process_delay
            unfinish_indi = env.process_delay_unfinish_ind

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            for iot_index in range(env.n_iot):

                history[env.time_count - 1][iot_index]['observation'] = \
                    observation_all[iot_index, :]
                history[env.time_count - 1][iot_index]['lstm'] = \
                    np.squeeze(lstm_state_all[iot_index, :])
                history[env.time_count - 1][iot_index]['action'] = action_all[iot_index]
                history[env.time_count - 1][iot_index]['observation_'] = \
                    observation_all_[iot_index]
                history[env.time_count - 1][iot_index]['lstm_'] = \
                    np.squeeze(lstm_state_all_[iot_index, :])

                update_index = np.where((1 - reward_indicator[:, iot_index]) *
                                        process_delay[:, iot_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]

                        reward = reward_fun(
                            process_delay[time_index, iot_index], env.max_delay,
                            unfinish_indi[time_index, iot_index])

                        dropped_list.append(unfinish_indi[time_index, iot_index])
                        if not unfinish_indi[time_index, iot_index]:
                            delay_list.append(process_delay[time_index, iot_index])

                        iot_RL_list[iot_index].store_transition(
                            history[time_index][iot_index]['observation'],
                            history[time_index][iot_index]['lstm'],
                            history[time_index][iot_index]['action'],
                            reward,
                            history[time_index][iot_index]['observation_'],
                            history[time_index][iot_index]['lstm_'])

                        iot_RL_list[iot_index].do_store_reward(
                            episode, time_index, reward)

                        iot_RL_list[iot_index].do_store_delay(
                            episode, time_index, process_delay[time_index, iot_index])

                        reward_indicator[time_index, iot_index] = 1

                        # rewards_dict[iot_index].append(-reward)
                        rewards_list.append(-reward)

            # ADD STEP (one step does not mean one store)
            RL_step += 1

            # UPDATE OBSERVATION
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # CONTROL LEARNING START TIME AND FREQUENCY
            if (RL_step > 200) and (RL_step % 10 == 0):
                for iot in range(env.n_iot):
                    iot_RL_list[iot].learn()

            # GAME ENDS
            if done:
                break

        avg_reward = np.mean(rewards_list)/env.n_iot
        episode_rewards.append(avg_reward)

        dropped_ratio = np.mean(dropped_list)/env.n_iot
        episode_dropped.append(dropped_ratio)

        avg_delay = np.mean(delay_list)/env.n_iot
        episode_delay.append(avg_delay)

        print(f"Episode: {episode} - Reward: {avg_reward} - Dropped: {dropped_ratio} - "
              + f"Delay: {avg_delay}")

        if episode % 10 == 0:
            plot_graphs(axs, episode_rewards, episode_dropped, episode_delay, show=show,
                        save=True, path=training_dir)

        #  ============================================================================ #
        #  ======================================== DRL END============================ #
        #  ============================================================================ #

    plot_graphs(axs, episode_rewards, episode_dropped, episode_delay, show=show,
                save=True, path=training_dir)

    end_time = time.time()
    print("\nTraining Time: %.2f(s)" % (end_time - start_time))
    input("Completed training.\nPress Enter to Finish")


def main(args):
    # Set random generator seed
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create a timestamp directory to save model, parameter and log files
    training_dir = \
        ('training/' + str(datetime.now().date()) + '_' +
         str(datetime.now().hour).zfill(2) + '-' + str(datetime.now().minute).zfill(2) +
         '/')

    # Delete if a directory with the same name already exists
    if os.path.exists(training_dir):
        rmtree(training_dir)

    # Create empty directories for saving model, parameter and log files
    os.makedirs(training_dir)
    os.makedirs(training_dir + 'plots')
    os.makedirs(training_dir + 'models')
    os.makedirs(training_dir + 'params')

    # Dump params to file
    with open(training_dir + 'params/params.dat', 'w') as jf:
        json.dump(vars(args), jf, indent=4)

    # GENERATE ENVIRONMENT
    env = Offload(args.num_iot, args.num_fog, NUM_TIME, MAX_DELAY, args.task_arrive_prob)

    # GENERATE MULTIPLE CLASSES FOR RL
    iot_RL_list = list()
    for iot in range(args.num_iot):
        iot_RL_list.append(DeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state,
                                        env.n_time,
                                        learning_rate=args.lr,
                                        reward_decay=0.9,
                                        e_greedy=0.99,
                                        replace_target_iter=200,  # update target net
                                        memory_size=500,  # maximum of memory
                                        batch_size=args.batch_size,
                                        optimizer=args.optimizer,
                                        seed=args.seed,
                                        ))

    # TRAIN THE SYSTEM
    train(env, iot_RL_list, args.num_episodes, args.plot, training_dir)
    print('Training Finished')


if __name__ == "__main__":

    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    parser = argparse.ArgumentParser(description='DQL for Mobile Edge Computing')
    parser.add_argument('--num_iot', type=int, default=50,
                        help='number of IOT devices (default: 50)')
    parser.add_argument('--num_fog', type=int, default=5,
                        help='number of FOG stations (default: 5)')
    parser.add_argument('--task_arrive_prob', type=float, default=0.3,
                        help='Task Arrive Probability (default: 0.3)')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='number of training episodes (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for optimizer (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='rms_prop',
                        help='optimizer for updating the NN (default: rms_prop)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the data file (default: None)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--plot',  default=False, action='store_true',
                        help='plot learning curve (default: False)')
    args = parser.parse_args()

    main(args)
