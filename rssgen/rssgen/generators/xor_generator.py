from rssgen.generators.dataset_generator import GenericSyntheticDatasetGenerator
from rssgen.generators.mnist_utils import MNISTUtils
from sympy.logic.inference import satisfiable
import sympy as sp
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class SyntheticXORGenerator(GenericSyntheticDatasetGenerator):
    """Mnist logic generator"""

    def __init__(
        self,
        output_path,
        val_prop,
        test_prop,
        n_digits,
        logic,
        symbols,
        ood_prop,
        use_mnist=True,
        mnist_path="data/MNIST/raw",
        **kwargs
    ):
        super().__init__(output_path, val_prop, test_prop, ood_prop)
        self.num_digits = n_digits
        self.logic = logic
        self.symbols = symbols
        self.use_mnist = use_mnist
        self.mnist = None

        if self.use_mnist:
            self.mnist = MNISTUtils(mnist_path)

    def _get_mnist_digit(self, digit_value):
        """Return MNIST digit"""
        image, _ = self.mnist.get_random_specific_mnist_digit(
            "train",
            self.mnist.train_mnist_images,
            self.mnist.train_mnist_labels,
            digit_value,
        )
        image = np.array(image).reshape(28, 28)
        return image

    def generate_synthetic_data(self, *args, world_to_generate=None):
        """Generate the synthetic data"""
        synthetic_image = {"image": None, "color": None}

        # container of the labels
        concepts = []
        image = np.zeros((self.num_digits), dtype=int)

        if self.use_mnist:
            image = np.zeros((28, 28 * self.num_digits), dtype=int)

        for j in range(self.num_digits):
            digit = (
                np.random.randint(0, 2)
                if world_to_generate is None
                else world_to_generate[j]
            )

            concepts.append(digit)
            if self.use_mnist:
                start = j * 28
                image[:, start : start + 28] = self._get_mnist_digit(digit)
            else:
                image[j] = digit

        # label is obtained by applying the logic
        label = self.evaluate_logic_expression(concepts, self.logic, self.symbols)

        if not self.use_mnist:
            image = image.reshape(1, self.num_digits)

        synthetic_image["image"] = image
        synthetic_image["cmap"] = "gray"

        label = bool(label)
        return synthetic_image, label, {"concepts": concepts}

    def _generate_binary_combinations(self, n):
        """Generate all the binary combinations"""
        digits = [False, True]
        combinations = list(product(digits, repeat=n))
        return set(combinations)

    def _sort_element(self, assignment, key_order):
        return [assignment[sp.symbols(key)] for key in key_order]

    def positive_combinations(self, combinations_in_distribution=None):
        """Get positive combinations"""
        p_combinations = []
        assignments = satisfiable(self.logic, all_models=True)
        for assignment in assignments:
            p_combinations.append(self._sort_element(assignment, self.symbols))
        set_combinations = {tuple(inner_list) for inner_list in p_combinations}
        return set_combinations

    def handle_given_combinations(self, combinations):
        """Handle combinations, given by the user"""

        def is_binary_string(binary):
            return all(bit in {"0", "1"} and len(bit) == 1 for bit in binary)

        if not all(is_binary_string(binary) for binary in combinations):
            raise ValueError("Provided combinations are not in the desired format")

        boolean_tuples = {
            tuple(int(bit) == 1 for bit in binary) for binary in combinations
        }
        return boolean_tuples

    def negative_combinations(self, combinations_in_distribution=None):
        """Deal with negative combinations"""
        all_combinations = self._generate_binary_combinations(self.num_digits)
        positive_combinations = self.positive_combinations()
        return all_combinations.difference(positive_combinations)


if __name__ == "__main__":
    # set backend
    matplotlib.use("qtagg")

    generator = SyntheticXORGenerator(
        output_path="synthetic_xor", val_prop=0.2, test_prop=0.3, num_digits=3
    )
    synthetic_image, label, meta = generator.generate_synthetic_data()
    print("Label", label, "Meta", meta)
    plt.imshow(synthetic_image["image"], cmap=synthetic_image["color"])
    plt.show()
