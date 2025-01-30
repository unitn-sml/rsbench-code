import argparse
import itertools as it
import operator as op
import numpy as np
import pickle

from abc import abstractmethod
from functools import reduce
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state

from pyeda.inter import exprvars, expr2dimacscnf
from pyeda.inter import And, Or, Xor, Implies, Equal, Not
from pyeda.inter import OneHot as PyEDAOneHot


def MyOneHot(*variables):
    onehotcnf = [Or(*variables)]
    for i in range(len(variables)-1):
        for j in range(i+1, len(variables)):
            onehotcnf.append(
                Or(Not(variables[i]),
                   Not(variables[j])))

    return And(*onehotcnf)


def _B_to_ttable(B):
    """Turn the joint-reasoning matrix into a truth table."""
    ttable = {}
    for entry, is_y_class in np.ndenumerate(B):
        x_vals, y = entry[:-1], entry[-1]

        if is_y_class:
            assert(x_vals not in ttable)
            ttable[x_vals] = y

    return ttable


def _pp_solution(sol, dataset, print_B=True):
    """Pretty-print a pyeda model."""
    Asol = np.zeros(shape=(dataset.n_objects * dataset.n_bits,
                           dataset.n_objects * dataset.n_bits),
                    dtype=np.uint8)
    AOsol = np.zeros(shape=(dataset.n_bits, dataset.n_bits),
                     dtype=np.uint8)
    Osol = np.zeros(shape=(dataset.n_objects, dataset.n_objects),
                    dtype=np.uint8)
    Vsol = np.zeros(shape=(dataset.n_variables, dataset.n_variables),
                    dtype=np.uint8)
    Bsol = np.zeros(shape= dataset.n_objects * dataset.domain_sizes + [dataset.n_classes],
                    dtype=np.uint8)

    for k in sol:
        if k.name == 'A' and sol[k]:
            Asol[k.indices] = 1
        elif k.name == 'AO' and sol[k]:
            AOsol[k.indices] = 1
        elif k.name == 'B' and sol[k]:
            Bsol[k.indices] = 1
        elif k.name == 'V' and sol[k]:
            Vsol[k.indices] = 1
        elif k.name == 'O' and sol[k]:
            Osol[k.indices] = 1


    print("domain sizes:", dataset.domain_sizes)
    print("n_classes:", dataset.n_classes)

    if print_B:
        print("\ntruth table:")
        for x, y in _B_to_ttable(Bsol).items():
            print(x, y)
        
    print()    
    print("A:")
    print(Asol)
    print("AO:")
    print(AOsol)
    print("V:")
    print(Vsol)    
    print("O:")
    print(Osol)


def _prop_or_count(n, p):
    return int(p if p > 1 else np.trunc(n * p))


def _booldot(avec, bvec):
    """Boolean dot product."""
    return reduce(op.or_, [a & b for a, b in zip(avec, bvec)])


def _all_equal(avec, bvec):
    """Equality between vectors of symbolic vars."""
    return And(*[Equal(a, b) for a, b in zip(avec, bvec)])


def _bind(variables, clauses):
    """Binds clauses-of-ints to pyeda variables."""
    return And(*[
        Or(*[
            variables[int(i) - 1] if i > 0 else ~variables[-int(i) - 1]
            for i in clause
        ])
        for clause in clauses
    ])


def _read_cnf(path):
    """Reads a CNF in DIMACS format."""
    with open(path, "rt") as fp:
        lines = fp.readlines()

    try:
        header = lines[0].strip()
        _, fmt, n_variables, n_clauses = header.split()
        n_variables = int(n_variables)
        n_clauses = int(n_clauses)
        assert fmt == "cnf"
    except:
        raise RuntimeError("not a valid CNF file")

    clauses = []
    for line in lines[1:]:
        line = line.strip().split()
        assert line[-1] == "0"
        clause = list(map(int, line))[:-1]
        clauses.append(clause)
    assert len(clauses) == n_clauses

    return n_variables, clauses


class Dataset:
    """Abstract Dataset(task) class."""

    def __init__(self, n_objects, domain_sizes, n_classes, cnf_path):
        self.domain_sizes = domain_sizes
        self.cnf_path = cnf_path
        self.gvecs, self.ys = None, None
        self.n_objects = n_objects # how many (homogeneous) objects
        self.n_variables = len(domain_sizes) # per object
        self.n_bits = sum(domain_sizes) # per object
        self.n_classes = n_classes

    @abstractmethod
    def make_data(self):
        """Fills the .gvecs and .ys fields with synthetic data."""
        pass

    @abstractmethod
    def load_data(self, path):
        """Fills the .gvecs and .ys fields with actual annotations."""
        pass

    @abstractmethod
    def k(self, cvec, y):
        """Knowledge for a given example."""
        pass

    def _make_all_gs(self):
        return list(it.product(*[list(range(size)) for size in self.n_objects * self.domain_sizes]))

    def _make_all_data(self, infer):
        """Generates all possible ground-truh concept vectors and labels."""
        gs = self._make_all_gs()
        ys = [infer(g) for g in gs]
        gs, ys = np.array(gs), np.array(ys)

        gs = OneHotEncoder(
            categories=[list(range(size)) for size in self.n_objects * self.domain_sizes],
            sparse_output=False
        ).fit_transform(gs)

        valid = np.where(ys >= 0)[0]
        ys = ys[valid]
        gs = gs[valid]

        return gs.astype(np.uint8), ys.astype(int)

    def subsample(self, p, rng=None):
        """Subsample a portion p (in [0,1]) of the exhaustive dataset."""
        assert self.gvecs is not None
        assert self.ys is not None
        assert len(self.gvecs) == len(self.ys)

        if p != 1:
            rng = check_random_state(rng)
            n_examples = len(self.gvecs)
            n_keep = _prop_or_count(n_examples, p)
            pi = rng.permutation(n_examples)
            self.gvecs = np.array([self.gvecs[i] for i in pi[:n_keep]])
            self.ys = np.array([self.ys[i] for i in pi[:n_keep]])


class CNFDataset(Dataset):
    """Abstract class implementing a logical task over propositions."""

    def __init__(self, args):

        if args.from_cnf is not None:
            basename = ".".join(args.from_cnf.split(".")[:-1])
        else:
            basename = f'rng({args.n_variables},{args.n_clauses},{args.clause_length})'

        super().__init__(
            1, # 1 object
            [2 for _ in range(self.n_variables)], # Boolean vars
            2, # Boolean  output
            f"cnf_{basename}"
        )

    def make_data(self):
        variables = exprvars("v", self.n_variables)
        formula = _bind(variables, self.clauses)

        def _infer(gvec):
            phi = And(*[
                ~variables[i] if g == 0 else variables[i]
                for i, g in enumerate(gvec)
            ])
            phi = formula & phi # already in CNF
            y = 1 if phi.satisfy_one() else 0

            return y

        self.gvecs, self.ys = self._make_all_data(_infer)

    def k(self, cvec, y):
        constraint = _bind([cvec[i] for i in range(1, len(cvec), 2)], self.clauses)
        return constraint if y else ~constraint


class RandomCNFDataset(CNFDataset):
    """Class implementing a random CNF."""

    def __init__(self, args):
        self.n_variables = args.n_variables
        self.n_clauses = args.n_clauses
        self.clause_length = args.clause_length
        self.clauses = self._sample_random_cnf(
            self.n_variables,
            self.n_clauses,
            self.clause_length,
            args.seed
        )

        '''
        print("Generated CNF:")
        for cl in self.clauses:
            print(" ".join(map(str, cl)))
        '''

        super().__init__(args)

    @staticmethod
    def _sample_random_cnf(n, m, k, rng):

        temp_vars = exprvars("v", n)

        def _nontrivial(curr, new):
            f1 = _bind(temp_vars, curr + [new])
            f2 = ~ _bind(temp_vars, [new])
            return (f1.satisfy_one() is not None) and \
                (f2.satisfy_one() is not None)

        rng = check_random_state(rng)
        clauses = []
        while len(clauses) < m:
            # NOTE: indices read from cnf files start from 1
            # we do the same
            indices = rng.choice(n, size=k) + 1
            signs = rng.choice([1, -1], size=len(indices))

            new_clause = list(indices * signs)
            if _nontrivial(clauses, new_clause):
                clauses.append(new_clause)

        return clauses #list(map(list, clauses))


class FileCNFDataset(CNFDataset):
    """Class implementing a custom CNF read from a DIMACS file."""

    def __init__(self, args):
        self.n_variables, self.clauses = _read_cnf(args.from_cnf)
        super().__init__(args)


class XorDataset(Dataset):
    """Class implementing a XOR task (for testing purposes)."""

    def __init__(self, args):
        super().__init__(
            1, # one object
            [2 for _ in range(args.n_variables)], # Boolean vars
            2, # Boolean output
            f"xor{args.n_variables}"
        )

    def make_data(self):
        xor = lambda x: list(it.accumulate(x, op.xor, initial=False))[-1]
        self.gvecs, self.ys = self._make_all_data(xor)

    def load_data(self):
        raise NotImplementedError()

    def k(self, cvec, y):
        constraint = Xor(*[cvec[i] for i in range(1, len(cvec), 2)])
        return constraint if y else ~constraint


class AddDataset(Dataset):
    """Class implementing (MNIST) addition."""

    def __init__(self, args):
        super().__init__(
            2, # two digits
            [10], # 10 values per digit
            19, # sum in [0,18]
            f"mnistadd"
        )

    def make_data(self):
        add = lambda x: x[0] + x[1]
        self.gvecs, self.ys = self._make_all_data(add)

    def load_data(self):
        raise NotImplementedError()

    def k(self, cvec, y):
        # NOTE cvec is one-hot of two 10-wise categoricals
        # NOTE y is categorical
        # XXX assumes that cvec is one-hot encoded

        avec, bvec = cvec[10:], cvec[:10]
        constraint = Or(*[
            And(avec[a], bvec[y - a])
            for a in range(10)
            if 0 <= y - a <= 9
        ])
        return constraint.simplify()


class SumParityDataset(Dataset):
    """Class implementing (MNIST) sum-parity."""

    def __init__(self, args):
        super().__init__(
            2, # two digits
            [10], # 10 values per digit
            2, # parity in {0, 1}
            f"sumparity",
        )

    def make_data(self):
        sumparity = lambda x: (x[0] + x[1]) % 2
        self.gvecs, self.ys = self._make_all_data(sumparity)

    def load_data(self):
        raise NotImplementedError()

    def k(self, cvec, y):
        # NOTE cvec is one-hot of two 10-wise categoricals
        # NOTE y is categorical
        # XXX assumes that cvec is one-hot encoded

        avec, bvec = cvec[10:], cvec[:10]
        constraint = Or(*[
            And(avec[a], bvec[b])
            for a in range(10)
            for b in range(10)
            if (a + b) % 2 == y
        ])
        return constraint.simplify()


class ClevrDataset(Dataset):
    """Class implementing Clevr."""

    # Colors
    GRAY = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    BROWN = 4
    PURPLE = 5
    CYAN = 6
    YELLOW = 7

    # Shapes
    CUBE = 0
    SPHERE = 1
    CYLINDER = 2

    # Materials
    RUBBER = 0
    METAL = 1

    # Sizes
    LARGE = 0
    SMALL = 1

    def __init__(self, args):
        super().__init__(
            2, # two objects
            [8, 3, 2, 2], # 4 feats per object
            3, # three classes
            f"clevr",
        )

    def make_data(self):

        def clevr(x):
            col1, sha1, mat1, siz1 = x[:4]
            col2, sha2, mat2, siz2 = x[4:]

            class1 = (
                siz1 == self.LARGE and sha1 == self.CUBE and
                siz2 == self.LARGE and sha2 == self.CYLINDER
            )
            class2 = (
                siz1 == self.SMALL and mat1 == self.METAL and sha1 == self.CUBE and
                siz2 == self.SMALL and sha2 == self.SPHERE
            )
            class3 = (
                siz1 == self.LARGE and col1 == self.BLUE and sha1 == self.SPHERE and
                siz2 == self.SMALL and col2 == self.YELLOW and sha2 == self.SPHERE
            )

            if class1 + class2 + class3 != 1:
                return -1 # invalid, will be discarded in _make_all_data()
            elif class1:
                return 0
            elif class2:
                return 1
            else:
                return 2

        self.gvecs, self.ys = self._make_all_data(clevr)

    def load_data(self):
        raise NotImplementedError()

    def k(self, cvec, y):
        # NOTE cvec is one-hot of two objects with four properties each
        # NOTE y is categorical

        col1, sha1, mat1, siz1 = cvec[0:8], cvec[8:11], cvec[11:13], cvec[13:15]
        col2, sha2, mat2, siz2 = cvec[15:23], cvec[23:26], cvec[26:28], cvec[28:30]

        rule1 = And(
            siz1[self.LARGE],
            sha1[self.CUBE],
            siz2[self.LARGE],
            sha2[self.CYLINDER],
        ).simplify()
        rule2 = And(
            siz1[self.SMALL],
            mat1[self.METAL],
            sha1[self.CUBE],
            siz2[self.SMALL],
            sha2[self.SPHERE],
        ).simplify()
        rule3 = And(
            siz1[self.LARGE],
            col1[self.BLUE],
            sha1[self.SPHERE],
            siz2[self.SMALL],
            col2[self.YELLOW],
            sha2[self.SPHERE],
        ).simplify()

        if y == 0:
            constraint = And(rule1, ~rule2, ~rule3)
        elif y == 1:
            constraint = And(~rule1, rule2, ~rule3)
        else:
            constraint = And(~rule1, ~rule2, rule3)

        return constraint.simplify()



class TinyClevrDataset(Dataset):
    """Class implementing a tiny version of Clevr."""

    # Colors
    RED = 0
    BLUE = 1
    GREEN = 2

    # Shapes
    CUBE = 0
    SPHERE = 1

    def __init__(self, args):
        super().__init__(
            2, # two objects
            [3, 2], # two features per object
            2, # two classes
            f"tinyclevr",
        )

    def make_data(self):

        def clevr(x):
            col1, sha1, col2, sha2, = x

            class1 = (
                col1 == self.RED and sha1 == self.CUBE and
                col2 == self.BLUE
            )
            class2 = (
                not class1
            )

            if class1 + class2 != 1:
                return -1 # invalid, will be discarded in _make_all_data()
            elif class1:
                return 0
            elif class2:
                return 1
            else:
                raise ValueError("what?")

        self.gvecs, self.ys = self._make_all_data(clevr)

    def load_data(self):
        raise NotImplementedError()

    def k(self, cvec, y):
        # NOTE cvec is one-hot of two objects with two properties each
        # NOTE y is categorical

        col1, sha1 = cvec[0:3], cvec[3:5]
        col2, sha2 = cvec[5:8], cvec[8:10]

        rule1 = And(
            col1[self.RED],
            sha1[self.CUBE],
            col2[self.BLUE]
        ).simplify()
        rule2 =  Not(rule1).simplify()

        if y == 0:
            constraint = And(rule1, ~rule2)
        elif y == 1:
            constraint = And(~rule1, rule2)

        return constraint.simplify()

class DebugDataset(Dataset):
    """Class implementing a debug version of TinyClevr."""


    def __init__(self, args):
        super().__init__(
            2, # two objects
            [args.n_variables], # one feature per object (n colors)
            2, # two classes
            f"debug{args.n_variables}",
        )

    def make_data(self):

        def clevr(x):
            col1, col2 = x

            class1 = (
                col1 == 0 and col2 == 0 # same color
            )
            class2 = (
                not class1
            )

            if class1 + class2 != 1:
                #return -1 # invalid, will be discarded in _make_all_data()
                raise ValueError("why?")
            elif class1:
                return 0
            elif class2:
                return 1
            else:
                raise ValueError("what?")

        self.gvecs, self.ys = self._make_all_data(clevr)

    def load_data(self):
        raise NotImplementedError()

    def k(self, cvec, y):
        # NOTE cvec is one-hot of two objects with two properties each
        # NOTE y is categorical
        ncols = self.domain_sizes[0]
        col1, col2 = cvec[0:ncols], cvec[ncols:ncols*2]

        rule1 = And(col1[0], col2[0]).simplify()
        rule2 =  Not(rule1).simplify()

        if y == 0:
            constraint = And(rule1, ~rule2)
        elif y == 1:
            constraint = And(~rule1, rule2)

        return constraint.simplify()


DATASETS = {
    "cnf": FileCNFDataset,
    "random": RandomCNFDataset,
    "xor": XorDataset,
    "add": AddDataset,
    "sumparity": SumParityDataset,
    "clevr": ClevrDataset,
    "tinyclevr": TinyClevrDataset,
    "debug": DebugDataset,
}


def _get_args_string(args):
    fields = [
        ("J", args.joint),
        ("s", args.subsample),
        ("c", args.concept_sup),
        ("T", not args.avoid_tseitin),
        (None, args.seed),
    ]
    basename = '__'.join([
        name + '=' + str(value) if name else str(value)
        for name, value in fields
    ])
    return basename


def _cat_to_ohe(values, dataset):
    """One-hot encodes a vector of categorical values."""
    n_objs = dataset.n_objects
    n_vars = dataset.n_variables
    assert(len(values) == n_objs * n_vars)
    result = np.zeros(n_objs * sum(dataset.domain_sizes))
    n_bits_filled = 0
    for o in range(n_objs):
        for v in range(n_vars):
            value = values[o * n_vars + v]
            result[n_bits_filled + value] = 1            
            n_bits_filled += dataset.domain_sizes[v]

    return result

def _encode_jrs_k(dataset, A, B, cvec, gty, count_equivalent_b=False):
    """Encodes the JRS counting problem.  gty is the (categorical) g-t label."""
    formula = True

    # Iterate over all possible combinations of concepts and labels; this
    # covers the entirety of B.
    
    #for cval, yval in it.product(dataset._make_all_gs(), range(dataset.n_classes)):
    for cval in dataset._make_all_gs():

        if count_equivalent_b:
            formula &= MyOneHot(*[B[cval + (yval,)] for yval in range(dataset.n_classes)])
        else:
            assigned_class = MyOneHot(*[B[cval + (yval,)] for yval in range(dataset.n_classes)])
            unassigned_class = And(*[Not(B[cval + (yval,)]) for yval in range(dataset.n_classes)])
            
            active_tt_entry = True
            for o in range(dataset.n_objects):
                for i, size_i in enumerate(dataset.domain_sizes):
                    xi = cval[o * len(dataset.domain_sizes) + i]
                    row_xi = (o * sum(dataset.domain_sizes)
                              + sum(dataset.domain_sizes[:i])
                              + xi)
                    active_tt_entry &= Or(*A[row_xi, :])
            
            formula &= Implies(active_tt_entry, assigned_class)
            formula &= Implies(Not(active_tt_entry), unassigned_class)
            
        
        for yval in range(dataset.n_classes):
            # Create a one-hot copy of the concept values
            ohe_cval = _cat_to_ohe(cval, dataset)
        
            # Lookup the entry in B that corresponds to cval and the g-t label
            indices = cval + (yval,)
            entry = B[indices]

            # Do the symbolic concepts activate this entry?
            active = _all_equal(cvec, ohe_cval)

            # Does the entry predict the g-t label?
            matches_gt = yval == gty

            # If the entry of B is active (i.e., it is selected by the symbolic
            # concepts), then it must predict the g-t label otherwise it cannot
            # be the g-t label.
            formula &= Implies(active, entry if matches_gt else ~entry)

    return formula.simplify()


def main():
    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument(
        "dataset", choices=sorted(DATASETS.keys()),
        help="dataset to count RSs for"
    )
    parser.add_argument(
        "-J", "--joint", action="store_true",
        help="count joint reasoning shortcuts",
    )
    parser.add_argument(
        "-s", "--subsample", type=float, default=1.0,
        help="fraction or number of observed gvecs to use"
    )
    parser.add_argument(
        "-c", "--concept-sup", type=float, default=0,
        help="fraction or number of gvecs with concept supervision"
    )
    parser.add_argument(
        "-D", "--print-data", action="store_true",
        help="print dataset prior to generating the CNF",
    )
    parser.add_argument(
        "-C", "--count", action="store_true",
        help="count the number of solutions",
    )
    parser.add_argument(
        "-E", "--enumerate", action="store_true",
        help="enumerate all solutions (SLOW!)",
    )
    parser.add_argument(
        "-f", "--from-cnf", type=str, default=None,
        help="cnf dataset: read CNF from this"
    )
    parser.add_argument(
        "-n", "--n-variables", type=int, default=None,
        help="random, xor: number of variables"
    )
    parser.add_argument(
        "-m", "--n-clauses", type=int, default=None,
        help="random: number of clauses"
    )
    parser.add_argument(
        "-k", "--clause-length", type=int, default=None,
        help="random: clause length"
    )
    parser.add_argument(
        "--seed", type=int, default=1,
        help="RNG seed"
    )
    parser.add_argument(
        "--avoid-tseitin", default=False,
        action="store_true",
        help="Avoid using Tseitin CNFization"
    )
    args = parser.parse_args()

    # generating the dataset/task
    print("Creating dataset")
    dataset = DATASETS[args.dataset](args)
    dataset.make_data()

    cnf_path = f"{dataset.cnf_path}__{_get_args_string(args)}.cnf"

    # possibly subsample the labelled data
    dataset.subsample(args.subsample, args.seed)

    n_examples = len(dataset.gvecs)
    n_csup = _prop_or_count(n_examples, args.concept_sup)
    pi = check_random_state(args.seed).permutation(n_examples)
    csup_mask = np.zeros(n_examples)
    csup_mask[pi[:n_csup]] = 1

    if args.print_data:
        print(dataset.gvecs)
        print(dataset.ys)

    print(f"Building formula: {len(dataset.gvecs)} gvecs, {dataset.n_objects * dataset.n_bits} bits")

    # A encodes a function from GT to learned concepts: C* -> C using a bit-wise OHE representation
    # Additional constraints to A are added through helper matrices AO, O and V
    # All matrices define are non-injective and non-surjective functions.
    # We assume objects and variables to be disentangled.
    # The values of A fully characterize a possible (non-joint) RS.
    A = exprvars("A", dataset.n_objects * dataset.n_bits, dataset.n_objects * dataset.n_bits)
    AO = exprvars("AO", dataset.n_bits, dataset.n_bits)
    V = exprvars("V", dataset.n_variables, dataset.n_variables)
    O = exprvars("O", dataset.n_objects, dataset.n_objects)
    if args.joint:
        B = exprvars("B", *(dataset.n_objects * dataset.domain_sizes + [dataset.n_classes]))

    OneHot = MyOneHot if args.avoid_tseitin else PyEDAOneHot 

    # O is a function among different objects in input (e.g. CLEVR items or MNIST digits)
    formula = And(*[OneHot(*O[:, o])
                    for o in range(dataset.n_objects)])
    # O is also injective
    formula &= And(*[OneHot(*O[o, :])
                    for o in range(dataset.n_objects)])

    # V is a function among variables (e.g. "Shape")
    formula &= And(*[OneHot(*V[:, v])
                    for v in range(dataset.n_variables)])

    # AO is a function
    formula &= And(*[OneHot(*AO[:, b])
                    for b in range(dataset.n_bits)])

    # A is a function (now implied by the above and the following constraints)
    formula &= And(*[OneHot(*A[:, i])
                    for i in range(dataset.n_bits)])

    # Mapped values in AO are consistent with the variable mapping in V (we assume a disentangled concept extractor)
    # I.e. AO is a block matrix having non-zero cv,gv-blocks IFF V[cv,gv] = 1
    if args.avoid_tseitin:
        formula &= And(*[Or(Not(V[cv, gv]), *[AO[i, j]
                                              for i in range(sum(dataset.domain_sizes[:cv]), sum(dataset.domain_sizes[:cv+1]))
                                              for j in range(sum(dataset.domain_sizes[:gv]), sum(dataset.domain_sizes[:gv+1]))])
                         for cv in range(dataset.n_variables)
                         for gv in range(dataset.n_variables)])

        formula &= And(*[Or(V[cv, gv], Not(AO[i, j]))
                         for cv in range(dataset.n_variables)
                         for gv in range(dataset.n_variables)
                         for i in range(sum(dataset.domain_sizes[:cv]), sum(dataset.domain_sizes[:cv+1]))
                         for j in range(sum(dataset.domain_sizes[:gv]), sum(dataset.domain_sizes[:gv+1]))
                         ])
    else:
        AO_nonzero_block = lambda cv, gv : Or(*[AO[i, j]
                                                for i in range(sum(dataset.domain_sizes[:cv]), sum(dataset.domain_sizes[:cv+1]))
                                                for j in range(sum(dataset.domain_sizes[:gv]), sum(dataset.domain_sizes[:gv+1]))])

        formula &= And(*[Equal(V[cv, gv], AO_nonzero_block(cv, gv))
                         for cv in range(dataset.n_variables)
                         for gv in range(dataset.n_variables)])


    # Mapped values in A are consistent with both AO (i.e. the concept extractor for a single object)
    # and the object mapping in O (we assume disentangled objects too)
    # I.e. A is a block matrix having AO as co,go-blocks IFF O[co,go] = 1
    if args.avoid_tseitin:
        formula &= And(*[Or(O[co, go], Not(A[dataset.n_bits * co + i, dataset.n_bits * go + j]))
                         for co in range(dataset.n_objects)
                         for go in range(dataset.n_objects)
                         for i in range(dataset.n_bits)
                         for j in range(dataset.n_bits)])

        formula &= And(*[And(Or(Not(O[co, go]),
                                Not(A[dataset.n_bits * co + i, dataset.n_bits * go + j]),
                                AO[i, j]),
                             Or(Not(O[co, go]),
                                A[dataset.n_bits * co + i, dataset.n_bits * go + j],
                                Not(AO[i, j])))
                         for co in range(dataset.n_objects)
                         for go in range(dataset.n_objects)
                         for i in range(dataset.n_bits)
                         for j in range(dataset.n_bits)])
    else:
        A_zero_block = lambda co, go : And(*[Not(A[dataset.n_bits * co + i, dataset.n_bits * go + j])
                                             for i in range(dataset.n_bits)
                                             for j in range(dataset.n_bits)])

        formula &= And(*[Implies(Not(O[co, go]), A_zero_block(co, go))
                         for co in range(dataset.n_objects)
                         for go in range(dataset.n_objects)])

        A_AO_block = lambda co, go : And(*[Equal(A[dataset.n_bits * co + i, dataset.n_bits * go + j],
                                             AO[i, j])
                                           for i in range(dataset.n_bits)
                                           for j in range(dataset.n_bits)])
    
        formula &= And(*[Implies(O[co, go], A_AO_block(co, go))
                         for co in range(dataset.n_objects)
                         for go in range(dataset.n_objects)])



    # force RSs to achieve perfect performance on data
    for gvec, y, has_csup in zip(dataset.gvecs, dataset.ys, csup_mask):

        cvec = [_booldot(A[i, :], gvec).simplify()
                for i in range(len(gvec))]

        offset = 0
        for vsize in dataset.domain_sizes:
            ohevar = OneHot(*cvec[offset:offset+vsize])
            formula &= (ohevar.to_cnf() if args.avoid_tseitin else ohevar)
            offset += vsize

        if not args.joint:
            correct_prediction = dataset.k(cvec, y)

        else:
            correct_prediction = _encode_jrs_k(dataset, A, B, cvec, y)

        formula &= (correct_prediction.to_cnf() if args.avoid_tseitin else correct_prediction)

        if has_csup:
            for i in range(dataset.n_bits):
                formula &= cvec[i] if gvec[i] else ~cvec[i]

    # export the formula in DIMACS format
    print("converting formula to CNF...")
    if args.avoid_tseitin:
        _, cnf = expr2dimacscnf(formula.simplify())
    else:
        _, cnf = expr2dimacscnf(formula.tseitin().to_cnf())

    print("Formula support:", len(formula.support))

    print(f"writing formula to {cnf_path}")
    with open(cnf_path, "wt") as fp:
        fp.write(str(cnf))

    # WARNING: use the enumerate flag for small problems only!!
    if args.count or args.enumerate:
        n_sol = 0
        for sol in formula.satisfy_all():
            if args.enumerate:
                _pp_solution(sol, dataset, args.joint)
                print("=" * 78)
            n_sol += 1

        print(f"{n_sol} solutions")

if __name__ == "__main__":
    main()
