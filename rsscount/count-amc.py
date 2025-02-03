import argparse

import pyapproxmc as pamc
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def main():
    """Use ApproxMC to compute an approximation of the exact model count C*
    with statistical guarantees, computing C s.t.

    Pr[ (1 - epsilon)C* < C < (1 + epsilon) C* ] >= delta

    This number represents a lower bound to the number of RSs in the task.

    """

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument("path", type=str,
                        help="path to CNF file")
    parser.add_argument("-e", "--epsilon", type=float, default=0.8,
                        help="pyapproxmc tolerance")
    parser.add_argument("-d", "--delta", type=float, default=0.2,
                        help="pyapprox confidence")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed number")
    parser.add_argument("--iter", type=int, default=1,
                        help="number of iterations")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Plot results")
    args = parser.parse_args()

    print(f"reading formula to {args.path}")
    with open(args.path, "rt") as fp:
        lines = list(map(str.strip, fp.readlines()))

    print(f"counting @ {args.epsilon}, {args.delta}")
    counts = []
    for i in range(args.iter):
        counter = pamc.Counter(epsilon=args.epsilon, delta=args.delta, seed=args.seed+i)
        for line in lines[1:]:
            counter.add_clause([lit for lit in map(int, line.split()) if lit != 0])
        count = counter.count()
        total = count[0] * 2**count[1]
        iterstr = "" if args.iter == 1 else f"[{i+1}/{args.iter}] "
        print(f"{iterstr} # of models: {count[0]} * 2**{count[1]}, aka {total}")
        counts.append(total)

    if args.iter > 1:

        mean, variance = norm.fit(counts)

        if args.plot:
            PLOT_WIDTH = 2 # times variance
            PLOT_SMOOTHNESS = 3

            x = np.linspace(mean - PLOT_WIDTH * variance,
                            mean + PLOT_WIDTH * variance,
                            len(counts) * 10 ** PLOT_SMOOTHNESS)

            plt.plot(x, norm.pdf(x, mean, variance), 'r-', label='norm pdf')
            plt.plot(counts, np.zeros(len(counts)), 'b', linestyle='', marker='x')
            plt.show()

        print(f"Result: {mean} +- {np.sqrt(variance)}")


if __name__ == "__main__":
    main()
