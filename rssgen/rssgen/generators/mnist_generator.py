from rssgen.generators.dataset_generator import GenericSyntheticDatasetGenerator
from rssgen.generators.utils import get_exp
from rssgen.utils import log

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from mnist import MNIST
import urllib.request
import re

from itertools import product, islice


class SyntheticMNISTGenerator(GenericSyntheticDatasetGenerator):
    """Mnist Math generator"""

    def __init__(
        self,
        output_path,
        val_prop,
        test_prop,
        num_digits,
        digit_values,
        logic,
        symbols,
        multiple_labels,
        ood_prop,
        mnist_path="data/MNIST/raw",
        **kwargs,
    ):
        super().__init__(output_path, val_prop, test_prop, ood_prop)
        self.num_digits = num_digits
        self.digits = digit_values
        self.logic = logic
        self.mnist_math = False
        self.n_equations = 1
        if isinstance(logic, list) and len(logic) > 1:
            self.mnist_math = True
            self.n_equations = len(logic)
        self.symbols = symbols
        self.multiple_labels = multiple_labels

        # MNIST DATA
        self.mnist_path = mnist_path
        self.download_mnist_if_not_exists()
        self.mnist_data = MNIST(self.mnist_path)
        self.mnist_data.gz = True

        (
            self.train_mnist_images,
            self.train_mnist_labels,
        ) = self.mnist_data.load_training()
        self.train_mnist_images, self.train_mnist_labels = self.filter_mnist_by_digits(
            self.train_mnist_images, self.train_mnist_labels, self.digits
        )

        self.test_mnist_images, self.test_mnist_labels = self.mnist_data.load_testing()
        self.test_mnist_images, self.test_mnist_labels = self.filter_mnist_by_digits(
            self.test_mnist_images, self.test_mnist_labels, self.digits
        )

        self.train_filtered_dictionary = {}
        self.test_filtered_dictionary = {}

    def positive_combinations(self, combinations_in_distribution=None):
        """Take first half of all possible combinations"""
        all_combinations = self._all_combinations(self.num_digits, self.digits)
        half_len = len(all_combinations) // 2
        first_half_combinations = list(islice(all_combinations, half_len))
        return set(first_half_combinations)

    def negative_combinations(self, combinations_in_distribution=None):
        """Take second half of all the possible combinations"""
        # set sampled della metà, ci sta. Da mettere qualcosa per fare sampling
        all_combinations = self._all_combinations(self.num_digits, self.digits)
        half_len = len(all_combinations) // 2
        second_half_combinations = list(islice(all_combinations, half_len, None))
        return set(second_half_combinations)

    def _all_combinations(self, num_digits, digits):
        """Return all the combinations (counting the system of equations)"""
        return list(product(digits, repeat=num_digits * self.n_equations))

    def filter_mnist_by_digits(self, images, labels, digits):
        """Filter MNSIT by digit"""
        # Convert images and labels to numpy arrays
        images_array = np.array(images)
        labels_array = np.array(labels)

        # Filter images and labels based on specified digits
        mask = np.isin(labels_array, digits)
        filtered_images = images_array[mask]
        filtered_labels = labels_array[mask]

        # Shuffle the data
        indices = np.arange(len(filtered_labels))
        np.random.shuffle(indices)

        filtered_images = filtered_images[indices]
        filtered_labels = filtered_labels[indices]

        return filtered_images, filtered_labels

    def get_filtered_data(self, name, images, labels, digit):
        """Filter data"""
        if name == "train":
            if digit in self.train_filtered_dictionary:
                return self.train_filtered_dictionary[digit]

        if name == "test":
            if digit in self.test_filtered_dictionary:
                return self.test_filtered_dictionary[digit]

        filtered_images, filtered_labels = self.filter_mnist_by_digits(
            images, labels, digit
        )

        if name == "train":
            self.train_filtered_dictionary[digit] = [filtered_images, filtered_labels]

        if name == "test":
            self.test_filtered_dictionary[digit] = [filtered_images, filtered_labels]

        return filtered_images, filtered_labels

    def get_random_specific_mnist_digit(self, name, images, labels, digit):
        """Get a specific MNIST digit"""
        filtered_images, filtered_labels = self.get_filtered_data(
            name, images, labels, digit
        )
        idx = np.random.randint(0, len(filtered_images))
        return filtered_images[idx], filtered_labels[idx]

    def generate_synthetic_data(self, *args, train=True, world_to_generate=None):
        """Generate synthetic MNISTAdd and MNISTMath data"""
        synthetic_image = {"image": None, "color": None}
        background = None

        # get images and labels
        mnist_images, mnist_labels = (
            (self.train_mnist_images, self.train_mnist_labels)
            if train
            else (self.test_mnist_images, self.test_mnist_labels)
        )
        train_name = "train" if train else "test"

        # container of the concept annotation
        concepts = []

        # loop over the equations it has to generate
        for t in range(self.n_equations):
            c_list = []
            # loop over the number of digits required
            for j in range(self.num_digits):
                # select random digit
                idx = np.random.randint(0, len(mnist_images))
                digit = mnist_images[idx]
                a_concept = mnist_labels[idx]

                # if the world is passed, then override the random digit
                if world_to_generate is not None:
                    digit, a_concept = self.get_random_specific_mnist_digit(
                        train_name,
                        mnist_images,
                        mnist_labels,
                        world_to_generate[t * self.n_equations + j],
                    )

                # save digit and concept
                digit = np.array(digit).reshape(28, 28)
                c_list.append(a_concept)

                if background is None:
                    background = digit
                else:
                    background = np.concatenate((background, digit), axis=1)

            # to distinguish between mnist math and mnist add
            if self.mnist_math:
                concepts.append(c_list)
            else:
                concepts = c_list

        # label is obtained by applying symbolic evaluation of the logic
        labels = None

        # compute the label
        if self.multiple_labels:
            labels = []
            for k, rule in enumerate(self.logic):
                # evaluate the logic (different rules, on the different set of compets)
                labels.append(
                    self.evaluate_logic_expression(concepts[k], rule, self.symbols)
                )
        else:
            labels = self.evaluate_logic_expression(concepts, self.logic, self.symbols)

        synthetic_image["image"] = np.clip(background, 0, 255).astype(np.uint8)
        synthetic_image["cmap"] = "gray"

        return synthetic_image, labels, {"concepts": concepts}

    def download_file(self, url, destination):
        """Download MNIST zip file"""
        log("info", f"Downloading {url}...")
        urllib.request.urlretrieve(url, destination)
        log("info", f"Download completed.")

    def download_mnist_if_not_exists(self):
        """Download all four MNIST files directly from the LeCun website"""
        if not os.path.exists(self.mnist_path):
            os.makedirs(self.mnist_path, exist_ok=True)

            base_url = "http://yann.lecun.com/exdb/mnist/"
            files = [
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            ]

            for file in files:
                file_url = base_url + file
                file_path = os.path.join(self.mnist_path, file)
                self.download_file(file_url, file_path)

            log("info", "Downloading MNIST dataset...")

            # Continue with the rest of the MNIST dataset download
            mndata = MNIST(self.mnist_path)
            mndata.gz = True  # No need to use Gzip since files are decompressed
            mndata.load_training()
            mndata.load_testing()

            log("info", "MNIST dataset downloaded and saved.")
        else:
            log("info", "MNIST dataset already exists.")

    def handle_given_combinations(self, combinations):
        """Handle passed combinations, basically read them and threat as integers"""
        result = []
        for s in combinations:
            digits = [int(digit) for digit in re.findall(r"\d", s)]
            result.append(tuple(digits))
        return result


if __name__ == "__main__":
    symbols = ["a", "b", "c"]
    logic_expression = "a + b + c"
    exp = get_exp(symbols, logic_expression)
    print("Logical expression: ", exp)

    # set backend
    matplotlib.use("qtagg")

    generator = SyntheticMNISTGenerator(
        output_path="../../data/synthetic_mnist",
        val_prop=0.2,
        test_prop=0.3,
        num_digits=3,
        digits=[0, 1, 2],
        logic=exp,
        symbols=symbols,
        mnist_path="../../data",
        digit_values=[0, 4],
        multiple_labels=True,
        ood_prop=0.0,
    )
    synthetic_image, label, meta = generator.generate_synthetic_data()
    print("Label", label, "Meta", meta)
    plt.imshow(synthetic_image["image"], cmap=synthetic_image["color"])
    plt.savefig("minst.png")
    plt.close()
