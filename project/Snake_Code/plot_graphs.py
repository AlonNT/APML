import argparse
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def main(directory, score_scope):
    game_name = os.path.split(os.path.normpath(directory))[-1]
    losses = [f for f in os.listdir(directory) if f.endswith('losses')]

    agents = dict()

    for file in losses:
        losses_file = os.path.join(directory, file)
        rewards_file = os.path.join(directory, file.replace('losses', 'rewards'))
        kills_file = os.path.join(directory, file.replace('losses', 'kills'))

        components = file.split('_')
        agent_name = components[0]

        losses = np.fromfile(losses_file, dtype=np.float32)
        rewards = np.fromfile(rewards_file, dtype=np.float32)
        kills = np.fromfile(kills_file, dtype=np.float32) if os.path.isfile(kills_file) else None

        arange = np.arange(1, len(rewards) + 1, dtype=np.float32)

        rewards_cumsum = np.cumsum(rewards)
        averaged_rewards = (1 / arange) * rewards_cumsum

        coefficient = 1 / float(score_scope)
        last_score_scope_mean_rewards = coefficient * (rewards_cumsum[score_scope:] - rewards_cumsum[:-score_scope])

        if kills is None:
            averaged_kills = None
            last_score_scope_mean_kills = None
        else:
            kills_cumsum = np.cumsum(kills)
            averaged_kills = (1 / arange) * kills_cumsum
            last_score_scope_mean_kills = coefficient * (kills_cumsum[score_scope:] - kills_cumsum[:-score_scope])

        agents[agent_name] = {'losses': losses,
                              'rewards': rewards,
                              'averaged_rewards': averaged_rewards,
                              'last_score_scope_mean_rewards': last_score_scope_mean_rewards,
                              'averaged_kills': averaged_kills,
                              'last_score_scope_mean_kills': last_score_scope_mean_kills}

    colors = ['blue', 'red', 'green', 'yellow', 'purple']

    # plt.figure()
    # plt.suptitle('{}\nLosses'.format(game_name))
    # color = iter(colors)
    # for agent_name, agent_dict in agents.items():
    #     plt.plot(agent_dict['losses'], c=next(color), label=agent_name)
    # plt.legend()
    # plt.savefig(os.path.join(directory, '{}_losses.png'.format(game_name)))

    plt.figure()
    plt.suptitle('{}\nAveraged Rewards'.format(game_name))
    color = iter(colors)
    for agent_name, agent_dict in agents.items():
        plt.plot(agent_dict['averaged_rewards'], c=next(color), label=agent_name)
    plt.legend()
    plt.savefig(os.path.join(directory, '{}_averaged_rewards.svg'.format(game_name)))
    plt.savefig(os.path.join(directory, '{}_averaged_rewards.png'.format(game_name)))

    plt.figure()
    plt.suptitle('{}\nLast {}-Score-Scope Mean Rewards'.format(game_name, score_scope))
    color = iter(colors)
    for agent_name, agent_dict in agents.items():
        plt.plot(agent_dict['last_score_scope_mean_rewards'], c=next(color), label=agent_name)
    plt.legend()
    plt.savefig(os.path.join(directory, '{}_last_{}_score_scope_mean_rewards.svg'.format(game_name, score_scope)))
    plt.savefig(os.path.join(directory, '{}_last_{}_score_scope_mean_rewards.png'.format(game_name, score_scope)))

    if all(agent_dict['averaged_kills'] is not None for agent_dict in agents.values()):
        plt.figure()
        plt.suptitle('{}\nAveraged Kills'.format(game_name))
        color = iter(colors)
        for agent_name, agent_dict in agents.items():
            plt.plot(agent_dict['averaged_kills'], c=next(color), label=agent_name)
        plt.legend()
        plt.savefig(os.path.join(directory, '{}_averaged_kills.svg'.format(game_name)))
        plt.savefig(os.path.join(directory, '{}_averaged_kills.png'.format(game_name)))

    if all(agent_dict['last_score_scope_mean_kills'] is not None for agent_dict in agents.values()):
        plt.figure()
        plt.suptitle('{}\nLast {}-Score-Scope Mean Kills'.format(game_name, score_scope))
        color = iter(colors)
        for agent_name, agent_dict in agents.items():
            plt.plot(agent_dict['last_score_scope_mean_kills'], c=next(color), label=agent_name)
        plt.legend()
        plt.savefig(os.path.join(directory, '{}_last_{}_score_scope_mean_kills.svg'.format(game_name, score_scope)))
        plt.savefig(os.path.join(directory, '{}_last_{}_score_scope_mean_kills.png'.format(game_name, score_scope)))

    plt.show()


def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument('--directory', '-d', type=str, default='.')
    args.add_argument('--score_scope', '-ss', type=int, default=5000)

    return args.parse_args()


if __name__ == '__main__':
    mpl.style.use('seaborn')
    arguments = parse_args()
    main(directory=arguments.directory, score_scope=arguments.score_scope)
