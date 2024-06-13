from rssgen.generators.dataset_generator import GenericSyntheticDatasetGenerator
from rssgen.generators.utils import get_exp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
from PIL import Image, ImageDraw
import random
import re

from itertools import product
from rssgen.utils import log
import sympy as sp


WIDTH = 120
MINSIZE = 20
MAXSIZE = 40

ALL_SHAPES = ["square", "circle", "triangle"]
ALL_COLORS = ["red", "yellow", "blue"]


class SyntheticKandinksyGenerator(GenericSyntheticDatasetGenerator):
    def __init__(
        self,
        output_path,
        val_prop,
        test_prop,
        ood_prop,
        shapes,
        colors,
        logic,
        n_shapes,
        n_figures,
        symbols,
        aggregator_logic,
        sample_size,
        aggregator_symbols,
        **kwargs,
    ):
        """Kandinsky generator"""
        super().__init__(output_path, val_prop, test_prop, ood_prop)
        self.kandinsky_shapes = [self.square, self.circle, self.triangle]
        self.kandinsky_named_shapes = list(
            set(ALL_SHAPES) & set(shapes)
        )  # intersect shapes
        self.kandinsky_colors = list(set(ALL_COLORS) & set(colors))  # intersect colors
        self.kandinsky_colors_map_int = {"red": 1, "yellow": 2, "blue": 3}
        self.kandinsky_shapes_map_int = {"square": 4, "circle": 5, "triangle": 6}
        self.logic = logic
        self.symbols = symbols
        self.n_shapes = n_shapes
        self.n_figures = n_figures
        self.aggregator_logic = aggregator_logic
        self.aggregator_symbols = aggregator_symbols
        self.sample_size = sample_size
        self.pos_set = set()
        self.neg_set = set()

    """Geometric sizes generators"""

    def square(self, d, cx, cy, s, f):
        s = 0.7 * s
        d.rectangle(((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)), fill=f)

    def circle(self, d, cx, cy, s, f):
        s = 0.7 * s * 4 / math.pi
        d.ellipse(((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)), fill=f)

    def triangle(self, d, cx, cy, s, f):
        r = math.radians(30)
        s = 0.7 * s * 3 * math.sqrt(3) / 4
        dx = s * math.cos(r) / 2
        dy = s * math.sin(r) / 2
        d.polygon([(cx, cy - s / 2), (cx + dx, cy + dy), (cx - dx, cy + dy)], fill=f)

    def kandinskyFigure(self, shapes, subsampling=1):
        """Single Kandinsky figure"""
        image = Image.new(
            "RGBA", (subsampling * WIDTH, subsampling * WIDTH), (255, 255, 255, 255)
        )
        d = ImageDraw.Draw(image)
        for s in shapes:
            s["shape_fun"](
                d,
                subsampling * s["cx"],
                subsampling * s["cy"],
                subsampling * s["size"],
                s["color"],
            )
        if subsampling > 1:
            image = image.resize((WIDTH, WIDTH), Image.BICUBIC)
        return image

    def overlaps(self, shapes):
        image = Image.new("L", (WIDTH, WIDTH), 0)
        sumarray = np.array(image)
        d = ImageDraw.Draw(image)

        for s in shapes:
            image = Image.new("L", (WIDTH, WIDTH), 0)
            d = ImageDraw.Draw(image)
            s["shape_fun"](d, s["cx"], s["cy"], s["size"], 10)
            sumarray = sumarray + np.array(image)

        sumimage = Image.fromarray(sumarray)
        return sumimage.getextrema()[1] > 10

    def combineFigures(self, n, f, world_to_generate=None):
        """Combine generated figures"""

        def generate_figure(idx, world_to_generate, f):
            if world_to_generate is None:
                shapes = f()
            else:
                shapes = self.generate_world(idx, world_to_generate)
            return shapes

        log("debug", "world to generate in combine figures", world_to_generate)

        images = []
        concepts = []
        for i in range(n):
            shapes = generate_figure(i, world_to_generate, f)

            while self.overlaps(shapes):
                shapes = generate_figure(i, world_to_generate, f)

            log(
                "debug",
                "shapes generated",
                [shapes[j]["shape"] for j in range(len(shapes))],
            )
            log(
                "debug",
                "shapes color generated",
                [shapes[j]["color"] for j in range(len(shapes))],
            )

            image = self.kandinskyFigure(shapes, 4)
            images.append(image)
            concepts.append(shapes)

        allimages = Image.new("RGBA", (WIDTH * n, WIDTH), (255, 255, 255, 255))
        for i in range(n):
            allimages.paste(images[i], (WIDTH * i, 0))
        return allimages, concepts

    def randomShapes(self):
        """Get random shapes"""
        nshapes = self.n_shapes
        shapes = []
        for _ in range(nshapes):
            cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
            cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
            size = random.randint(MINSIZE, MAXSIZE)
            col = random.randint(0, len(self.kandinsky_colors) - 1)
            sha = random.randint(0, len(self.kandinsky_shapes) - 1)
            shape = {
                "shape_fun": self.kandinsky_shapes[sha],
                "shape": self.kandinsky_named_shapes[sha],
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": self.kandinsky_colors[col],
            }
            shapes.append(shape)
        return shapes

    def _get_shape(self, shape_name):
        """Get shape"""
        if shape_name == "square":
            return self.square
        if shape_name == "circle":
            return self.circle
        if shape_name == "triangle":
            return self.triangle

        log("error", f"shape not recognized: {shape_name}!")
        exit(1)

    def generate_world(self, idx, world_to_generate):
        """Generate possible world"""
        nshapes = len(world_to_generate[idx])
        shapes = []
        for j in range(0, nshapes, 2):
            cx = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
            cy = random.randint(MAXSIZE / 2, WIDTH - MAXSIZE / 2)
            size = random.randint(MINSIZE, MAXSIZE)
            col = world_to_generate[idx][j]
            sha = world_to_generate[idx][j + 1]
            shape = {
                "shape_fun": self._get_shape(sha),
                "shape": sha,
                "cx": cx,
                "cy": cy,
                "size": size,
                "color": col,
            }
            shapes.append(shape)
        return shapes

    def filter_concepts(self, concepts):
        """Filter concepts"""
        filtered_concepts = []

        for fig_concepts in concepts:
            concepts_per_figure = []
            for object_concepts in fig_concepts:
                name = self.kandinsky_shapes_map_int.get(object_concepts["shape"])
                concepts_per_figure.append(name)
                color = self.kandinsky_colors_map_int.get(object_concepts["color"])
                concepts_per_figure.append(color)
            filtered_concepts.append(concepts_per_figure)
        return filtered_concepts

    def generate_synthetic_data(self, *args, world_to_generate=None):
        """Generate data"""
        synthetic_image = {"image": None, "color": None}

        log("debug", "world to generate", world_to_generate)

        # generate figure with concepts
        if world_to_generate is None:
            image, concepts = self.combineFigures(self.n_figures, self.randomShapes)
        else:
            image, concepts = self.combineFigures(
                self.n_figures, None, world_to_generate=world_to_generate
            )

        # filter concepts
        filtered_concepts = self.filter_concepts(concepts)

        log("debug", "filtered concept value", filtered_concepts)

        patterns_predictions = []
        for fig_concepts in filtered_concepts:
            patterns_predictions.append(
                self.evaluate_logic_expression(fig_concepts, self.logic, self.symbols)
            )

        label = self.evaluate_logic_expression(
            patterns_predictions, self.aggregator_logic, self.aggregator_symbols
        )

        log(
            "debug",
            "Patterns prediction...",
            patterns_predictions,
            "Final prediction...",
            label,
        )

        synthetic_image["image"] = image
        synthetic_image["cmap"] = None

        label = bool(label)

        return synthetic_image, label, {"concepts": filtered_concepts}

    def _filter_combinations(self, combinations_in_distribution=None):
        """Filter combinations"""
        single_figure_combinations = self._all_combinations(
            self.kandinsky_colors, self.kandinsky_named_shapes, self.n_shapes
        )
        log("info", f"You asked to sample {self.sample_size} figure combinations...")
        log(
            "info",
            f"Sampling {self.sample_size * self.n_figures} single combinations...",
        )

        to_remove = 0
        if combinations_in_distribution is not None:
            to_remove = len(combinations_in_distribution)
            combinations_in_distribution = self._handle_combinations(
                combinations_in_distribution, return_tuple=False
            )

        random_samples = random.choices(
            single_figure_combinations,
            k=(self.sample_size * self.n_figures) - to_remove,
        )
        combinations = [
            random_samples[i : i + self.n_shapes]
            for i in range(0, len(random_samples), self.n_shapes)
        ]

        # filtering before combinations_in_distribution
        if combinations_in_distribution is not None:
            combinations.extend(combinations_in_distribution)

        log("debug", f"Figure combinations: {len(combinations)}")
        log("debug", f"Figure combinations: {combinations}")

        for combo in combinations:
            log("debug", f"Combo: {combo}")

            fig_logic_out_list = [
                self.evaluate_logic_expression(
                    self.map_vector(fig_combo), self.logic, self.symbols
                )
                for fig_combo in combo
            ]

            if any(
                not isinstance(
                    fig_logic_out,
                    (sp.logic.boolalg.BooleanFalse, sp.logic.boolalg.BooleanTrue),
                )
                for fig_logic_out in fig_logic_out_list
            ):
                log(
                    "error",
                    f"Some logic outputs are not boolean values: {fig_logic_out_list}",
                )
                exit(1)

            label = self.evaluate_logic_expression(
                fig_logic_out_list, self.aggregator_logic, self.aggregator_symbols
            )

            if not isinstance(
                label, (sp.logic.boolalg.BooleanFalse, sp.logic.boolalg.BooleanTrue)
            ):
                log(
                    "error",
                    f"Aggregator logic output is not a boolean value: {type(label)}, Value: {label}",
                )
                exit(1)

            if label:
                self.pos_set.add(tuple(combo))
            else:
                self.neg_set.add(tuple(combo))

        if not self.pos_set or not self.neg_set:
            log(
                "error",
                "Logic is either a contradiction or a tautology or the sampling rate is too low",
            )
            exit(1)

        log(
            "info",
            f"True assignments: {len(self.pos_set)}, False assignments: {len(self.neg_set)}",
        )

    def map_color_to_integer(self, color):
        """Map colors to integers"""
        log("debug", "current color...", color)
        color_mapping = {"red": 1, "blue": 2, "yellow": 3}
        return color_mapping.get(color, 0)

    def map_shape_to_integer(self, shape):
        """Map shape to integers"""
        log("debug", "current shape...", shape)
        shapes_mapping = {"circle": 10, "square": 20, "triangle": 30}
        return shapes_mapping.get(shape, 0)

    def map_vector(self, vector):
        """Map vector to categorical"""
        vector = list(vector)
        log("debug", "changed vector in list", vector)
        for i in range(0, len(vector), 2):
            vector[i] = self.map_color_to_integer(vector[i])
            vector[i + 1] = self.map_shape_to_integer(vector[i + 1])
        log("debug", "changed vector in integer", vector)
        return tuple(vector)

    def positive_combinations(self, combinations_in_distribution=None):
        """Get positive combinations"""
        if (not self.pos_set) or (not self.neg_set):
            self._filter_combinations(combinations_in_distribution)
        log("debug", "positive set", self.pos_set)
        return self.pos_set

    def negative_combinations(self, combinations_in_distribution=None):
        """Get negative combinations"""
        if (not self.pos_set) or (not self.neg_set):
            self._filter_combinations(combinations_in_distribution)
        log("debug", "negative set", self.neg_set)
        return self.neg_set

    def _all_combinations(self, colors, shapes, n_objects):
        """Get all combinations"""
        all_comb_in_a_figure = list(product(colors, shapes, repeat=n_objects))
        return all_comb_in_a_figure

    def handle_given_combinations(self, combinations):
        """Handle given combinations"""
        return self._handle_combinations(
            combinations,
        )

    def _parse_combination_string(self, combo_figure):
        """Parse combianation strings"""
        result_per_figure = []

        for s in combo_figure:
            words = re.findall(r"\b\w+\b", s)

            log("info", "obtained world is", words)

            if (
                len(words) == 2
                and words[0] in self.kandinsky_colors
                and words[1] in self.kandinsky_named_shapes
            ):
                result_per_figure.extend(words)
            else:
                log(
                    "error",
                    f"Error: Invalid words in string '{s}'. Please use valid colors and shapes.",
                )
                exit(1)

        if len(result_per_figure) // 2 != self.n_shapes:
            log(
                "error",
                f"Error: Invalid number of objects in a figure:",
                len(result_per_figure),
            )
            exit(1)

        return tuple(result_per_figure)

    def _handle_combinations(self, combinations, return_tuple=True):
        """Handle given combinations"""

        if len(combinations[0]) != self.n_figures:
            log(
                "error",
                f"Error: Invalid number of combinations. Expected {self.n_figures}, got {len(combinations)}.",
            )
            exit(1)

        result_total = []

        for combo in combinations:
            result_combo = [
                self._parse_combination_string(combo_figure) for combo_figure in combo
            ]
            result_total.append(tuple(result_combo) if return_tuple else result_combo)

        log("info", "total parsed combinations", result_total)

        return result_total


if __name__ == "__main__":
    # set backend
    matplotlib.use("qtagg")

    logic_expression = """
        (Eq(color_1, color_2) & Eq(shape_1, shape_2) & Ne(shape_1, shape_3)) |
        (Eq(color_1, color_3) & Eq(shape_1, shape_3) & Ne(shape_1, shape_2)) |
        (Eq(color_2, color_3) & Eq(shape_2, shape_3) & Ne(shape_1, shape_3))
    """
    aggregator_logic = """
        pattern_1 & pattern_2 & pattern_3
    """

    symbols = ["shape_1", "color_1", "shape_2", "color_2", "shape_3", "color_3"]
    aggregator_symbols = ["pattern_1", "pattern_2", "pattern_3"]

    exp = get_exp(symbols, logic_expression)
    aggregator_exp = get_exp(aggregator_symbols, aggregator_logic)

    print("Logical expression: ", exp)
    print("Aggregating logical expression: ", aggregator_exp)

    generator = SyntheticKandinksyGenerator(
        output_path="../../data/synthetic_kandinksy",
        val_prop=0.2,
        test_prop=0.3,
        n_shapes=3,
        n_figures=3,
        logic=exp,
        symbols=symbols,
        aggregator_logic=aggregator_exp,
        aggregator_symbols=aggregator_symbols,
    )

    synthetic_image, label, meta = generator.generate_synthetic_data()
    print("Label", label)

    plt.imshow(synthetic_image["image"], cmap=synthetic_image["color"])
    plt.show()
