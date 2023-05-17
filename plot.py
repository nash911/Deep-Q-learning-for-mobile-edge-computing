import numpy as np
from glob import glob
import argparse
import json
import matplotlib.pyplot as plt


def plot_graphs(costs, colors, labels, title='Title', show=True, save=True, path=None):
    fig, axs = plt.subplots(1, figsize=(10, 6))
    x = np.arange(len(costs[0])).tolist()

    for cost, color, label in zip(costs, colors, labels):
        axs.plot(x, cost, color=color, label=label)
    axs.set(title=title)
    axs.set(ylabel='Avg. Cost')
    axs.set(xlabel='Episode')
    axs.legend(loc='upper right')

    if save:
        plt.savefig(path + "avg_cost_plot.png")

    if show:
        plt.show(block=False)
        input()


def main(args):
    dirs = glob(f"{args.path}/*/")

    costs = list()
    colors = list()
    labels = list()

    for dir in dirs:
        avg_costs_np = np.load(dir + "/plots/avg_cost.npy")
        avg_costs_np = np.convolve(avg_costs_np, np.ones((args.window,))/args.window,
                                   mode='valid')
        costs.append(avg_costs_np)
        with open(dir + "/plots/plot_props.dat") as fp:
            data = json.load(fp)
        colors.append(data['color'])
        labels.append(data['label'])

    plot_graphs(costs, colors, labels, args.title, path=args.path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot Results for Mobile Edge Computing')
    parser.add_argument('--cost',  default=False, action='store_true',
                        help='plot average cost curves (default: False)')
    parser.add_argument('--path', type=str, default=None,
                        help='path to results directory (default: None)')
    parser.add_argument('--window', type=int, default=50,
                        help='moving average window size (default: 50)')
    parser.add_argument('--title', type=str, default='Title',
                        help='plot title (default: Title)')
    args = parser.parse_args()

    main(args)
