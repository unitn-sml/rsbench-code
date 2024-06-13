import argparse
import pyapproxmc as pamc


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
    args = parser.parse_args()

    print(f"reading formula to {args.path}")
    with open(args.path, "rt") as fp:
        lines = list(map(str.strip, fp.readlines()))

    print(f"counting @ {args.epsilon}, {args.delta}")
    counter = pamc.Counter(epsilon=args.epsilon, delta=args.delta, seed=args.seed)
    for line in lines[1:]:
        counter.add_clause([lit for lit in map(int, line.split()) if lit != 0])
    count = counter.count()
    total = count[0] * 2**count[1]

    print(f"# of models: {count[0]} * 2**{count[1]}, aka {total}")


if __name__ == "__main__":
    main()
