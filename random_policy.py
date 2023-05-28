import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

from fog_env import Offload
from utils import plot_graphs


def reward_fun(delay, max_delay, unfinish_indi):
    # still use reward, but use the negative value
    if unfinish_indi:
        reward = - max_delay * 2
    else:
        reward = - delay

    return reward


def random_policy(env, num_episodes, show=False):
    episode_rewards = list()
    episode_dropped = list()
    episode_delay = list()

    fig, axs = plt.subplots(3, figsize=(10, 12), sharex=True)

    for episode in range(num_episodes):
        rewards_list = list()
        dropped_list = list()
        delay_list = list()

        reward_indicator = np.zeros([env.n_time, env.n_iot])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset()

        # TRAIN DRL
        while True:

            # PERFORM ACTION
            action_all = np.zeros([env.n_iot])
            for iot_index in range(env.n_iot):
                observation = np.squeeze(observation_all[iot_index, :])
                if np.sum(observation) == 0:
                    # if there is no task, action = 0 (also need to be stored)
                    action_all[iot_index] = 0
                else:  # Follow a random action
                    action_all[iot_index] = np.random.randint(env.n_actions)

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observation_all_, lstm_state_all_, done = env.step(action_all)

            process_delay = env.process_delay
            unfinish_indi = env.process_delay_unfinish_ind

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            for iot_index in range(env.n_iot):
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

                        reward_indicator[time_index, iot_index] = 1

                        rewards_list.append(-reward)

                # UPDATE OBSERVATION
                observation_all = observation_all_

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
                        save=False)

    plot_graphs(axs, episode_rewards, episode_dropped, episode_delay, show=show,
                save=False)

    input("Completed.\nPress Enter to Finish")


def main(args):
    # Set random generator seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # GENERATE ENVIRONMENT
    env = Offload(args.num_iot, args.num_fog, NUM_TIME, MAX_DELAY, args.task_arrival_prob)

    # TRAIN THE SYSTEM
    random_policy(env, args.num_episodes, args.plot)


if __name__ == "__main__":

    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    parser = argparse.ArgumentParser(description='DQL for Mobile Edge Computing')
    parser.add_argument('--num_iot', type=int, default=50,
                        help='number of IOT devices (default: 50)')
    parser.add_argument('--num_fog', type=int, default=5,
                        help='number of FOG stations (default: 5)')
    parser.add_argument('--task_arrival_prob', type=float, default=0.3,
                        help='Task Arrival Probability (default: 0.3)')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='number of training episodes (default: 1000)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--plot',  default=False, action='store_true',
                        help='plot learning curve (default: False)')
    args = parser.parse_args()

    main(args)
