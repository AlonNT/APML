import argparse
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def main(directory, gamma, score_scope):
    game_name = os.path.split(os.path.normpath(directory))[-1]
    losses = [f for f in os.listdir(directory) if f.endswith('losses')]

    agents = dict()

    for file in losses:
        losses_file = os.path.join(directory, file)
        rewards_file = os.path.join(directory, file.replace('losses', 'rewards'))

        components = file.split('_')
        agent_name = components[0]

        losses = np.fromfile(losses_file, dtype=np.float32)
        rewards = np.fromfile(rewards_file, dtype=np.float32)
        cumsum = np.cumsum(rewards)

        arange = np.arange(1, len(rewards) + 1, dtype=np.float32)
        averaged_rewards = (1 / arange) * cumsum

        discounted_rewards = np.cumsum((gamma ** arange) * rewards)

        last_score_scope_mean_rewards = (1 / float(score_scope)) * (cumsum[score_scope:] - cumsum[:-score_scope])

        agents[agent_name] = {'losses': losses,
                              'rewards': rewards,
                              'averaged_rewards': averaged_rewards,
                              'discounted_rewards': discounted_rewards,
                              'last_score_scope_mean_rewards': last_score_scope_mean_rewards}

    colors = ['blue', 'red', 'green', 'yellow', 'purple']

    plt.figure()
    plt.suptitle('{}\nLosses'.format(game_name))
    color = iter(colors)
    for agent_name, agent_dict in agents.items():
        plt.plot(agent_dict['losses'], c=next(color), label=agent_name)
    plt.legend()
    plt.savefig(os.path.join(directory, '{}_losses.png'.format(game_name)))

    plt.figure()
    plt.suptitle('{}\nAveraged Rewards'.format(game_name))
    color = iter(colors)
    for agent_name, agent_dict in agents.items():
        plt.plot(agent_dict['averaged_rewards'], c=next(color), label=agent_name)
    plt.legend()
    plt.savefig(os.path.join(directory, '{}_averaged_rewards.png'.format(game_name)))

    plt.figure()
    plt.suptitle('{}\nLast {}-Score-Scope Mean Rewards'.format(game_name, score_scope))
    color = iter(colors)
    for agent_name, agent_dict in agents.items():
        plt.plot(agent_dict['last_score_scope_mean_rewards'], c=next(color), label=agent_name)
    plt.legend()
    plt.savefig(os.path.join(directory, '{}_last_{}_score_scope_mean_rewards.png'.format(game_name, score_scope)))
    plt.show()


def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument('--directory', '-d', type=str, default='.')
    args.add_argument('--gamma', '-g', type=float, default=0.95)
    args.add_argument('--score_scope', '-ss', type=int, default=5000)

    return args.parse_args()


if __name__ == '__main__':
    mpl.style.use('seaborn')
    arguments = parse_args()
    main(directory=arguments.directory, gamma=arguments.gamma, score_scope=arguments.score_scope)
