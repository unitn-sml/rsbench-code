from rssgen.generators.mnist_generator import SyntheticMNISTGenerator


class MNISTUtils(SyntheticMNISTGenerator):
    """Mnist Utils"""

    def __init__(
        self,
        mnist_path="data",
    ):
        super(MNISTUtils, self).__init__(
            output_path="",
            val_prop=0,
            test_prop=0,
            num_digits=1,
            digit_values=[0, 1],
            logic=None,
            symbols=None,
            multiple_labels=False,
            ood_prop=0,
            mnist_path=mnist_path,
        )
