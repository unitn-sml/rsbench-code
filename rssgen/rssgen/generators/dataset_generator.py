import os
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from rssgen.utils import log

from PIL import Image


class GenericSyntheticDatasetGenerator:
    """Base class for the synthetic dataset generation"""

    def __init__(self, output_path, val_prop, test_prop, ood_prop):
        self.output_path = output_path

        self.val_prop = val_prop
        self.test_prop = test_prop
        self.ood_prop = ood_prop

        # Create output directory if it doesn't exist
        self.train_path = os.path.join(self.output_path, "train")
        self.val_path = os.path.join(self.output_path, "val")
        self.test_path = os.path.join(self.output_path, "test")
        self.ood_path = os.path.join(self.output_path, "ood")

        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.ood_path, exist_ok=True)

    def generate_synthetic_data(self):
        raise NotImplementedError(
            "Subclasses must implement the generate_synthetic_data method."
        )

    def evaluate_logic_expression(self, values, logic_expression, symbols_names):
        substitutions_dict = {
            symbol_name: value for symbol_name, value in zip(symbols_names, values)
        }
        log("debug", "substituting...", substitutions_dict, "to", logic_expression)
        # Substitute values into the expression
        result = logic_expression.subs(substitutions_dict)
        log("debug", "result of the substitution", result)
        return result

    def positive_combinations(self, combinations_in_distribution=None):
        raise NotImplementedError(
            "Subclasses must implement the positive_combinations method."
        )

    def negative_combinations(self, combinations_in_distribution=None):
        raise NotImplementedError(
            "Subclasses must implement the negative_combinations method."
        )

    def handle_given_combinations(self, combinations):
        return combinations

    def split_set(self, set_variable, percentage):
        list_set = list(set_variable)
        split_index = int(len(list_set) * percentage)
        first_part = list_set[:split_index]
        second_part = list_set[split_index:]
        return first_part, second_part

    def filering_given_combinations(self, starting_set, given_combinations):
        given_combinations_set = set(given_combinations)
        combinations_in_starting = starting_set.intersection(given_combinations_set)
        combinations_not_in_starting = starting_set.difference(combinations_in_starting)
        return list(combinations_in_starting), list(combinations_not_in_starting)

    def _is_pil_image(self, img):
        try:
            return isinstance(img, Image.Image)
        except Exception:
            return False

    def _save_img(self, img, color, img_path):
        if self._is_pil_image(img):
            img.save(img_path)
        else:
            plt.imsave(img_path, img, cmap=color)

    def generate_dataset(
        self,
        *args,
        num_samples=1000,
        compression_type=None,
        keep_only_compressed=False,
        prop_in_distribution=1,
        combinations_in_distribution=None,
        **kwargs,
    ):
        synthetic_image, label, meta = self.generate_synthetic_data(args)

        train_size = num_samples
        val_size = int(num_samples * self.val_prop)
        test_size = int(num_samples * self.test_prop)

        # get proportion of in_out_distribution
        positive_combinations = self.positive_combinations(combinations_in_distribution)
        negative_combinations = self.negative_combinations(combinations_in_distribution)

        log("info", "Positive combinations", positive_combinations)

        log("info", "Negative combinations", negative_combinations)

        if combinations_in_distribution is not None:
            log(
                "info",
                "Splitting the dataset according to the given combinations, in_distribution proportion will be ignored",
            )

            log("info", "Given combinations", combinations_in_distribution)

            try:
                combinations_in_distribution = self.handle_given_combinations(
                    combinations_in_distribution
                )
            except ValueError as e:
                log("error", e)
                exit(1)

            log("info", "Handled combinations", combinations_in_distribution)

            # splitting according to a combination
            positive_id, positive_ood = self.filering_given_combinations(
                positive_combinations, combinations_in_distribution
            )

            log("debug", "Positive combinations in distribution", positive_id)

            log("debug", "Positive combinations ood distribution", positive_ood)

            negative_id, negative_ood = self.filering_given_combinations(
                negative_combinations, combinations_in_distribution
            )

            log("debug", "Negative combinations in distribution", negative_id)

            log("debug", "Negative combinations ood distribution", negative_ood)

            log("info", "OOD combinations", (len(positive_ood) + len(positive_ood)))
            log("info", "ID combinations", (len(positive_id) + len(negative_id)))

            if len(positive_id) + len(negative_id) == 0:
                log("error", "There must be at least one combination in distribution!")
                exit(1)

            if len(positive_ood) + len(negative_ood) == 0:
                log(
                    "error",
                    "There must be at least one combination out of distribution if you specify `combinations_in_distribution`!",
                )
                exit(1)

        else:
            log(
                "info",
                "Splitting the dataset according to the in_distribution proportion",
            )
            # splitting according to a proportion
            positive_id, positive_ood = self.split_set(
                positive_combinations, prop_in_distribution
            )
            negative_id, negative_ood = self.split_set(
                negative_combinations, prop_in_distribution
            )

        log("info", "positive samples in distribution", len(positive_id))
        log("info", "negative samples in distribution", len(negative_id))
        log("info", "positive samples out of distribution", len(positive_ood))
        log("info", "negative samples out of distribution", len(negative_ood))

        ood_size = 0 if (len(positive_ood) + len(negative_ood)) == 0 else test_size

        for idx, (name, dataset_size, folder) in enumerate(
            zip(
                ["train", "val", "test", "ood"],
                [train_size, val_size, test_size, ood_size],
                [self.train_path, self.val_path, self.test_path, self.ood_path],
            )
        ):
            log("info", "Doing", name, "with", dataset_size, "examples...")
            train = True if (name != "test" and name != "ood") else False

            # positive and negative worlds to sample
            positive_to_sample = positive_id
            negative_to_sample = negative_id
            if name == "ood":
                positive_to_sample = positive_ood
                negative_to_sample = negative_ood

            # count of samples to generate
            total_positive_samples = train_size // 2

            log(
                "info",
                "Currently for",
                name,
                "I have",
                total_positive_samples,
                "positive samples of",
                len(positive_to_sample),
                "positive combinations to sample",
                "and",
                train_size - total_positive_samples,
                "negative samples of",
                len(negative_to_sample),
                "negative combinations to sample",
            )

            # Already Generated
            already_generated = set()

            for i in range(dataset_size):
                # GET PROPROTIONATE WORLDS
                if i < total_positive_samples and len(positive_to_sample) > 0:
                    # generate positive samples
                    idx_sampling = i % len(positive_to_sample)
                    world_to_generate = positive_to_sample[idx_sampling]
                elif len(negative_to_sample) > 0:
                    # generate negative samples
                    idx_sampling = i % len(negative_to_sample)
                    world_to_generate = negative_to_sample[idx_sampling]

                if not world_to_generate in already_generated:
                    log(
                        "info",
                        "For",
                        name,
                        "generating world:",
                        world_to_generate,
                    )

                # Add the current world
                already_generated.add(world_to_generate)

                synthetic_image, label, meta = self.generate_synthetic_data(
                    train, args, world_to_generate=world_to_generate
                )

                log("debug", "data generated", synthetic_image)
                log("debug", "label generated", label)
                log("debug", "meta generated", meta)
                log("debug", "example done")

                # image
                image = synthetic_image["image"]
                color = synthetic_image["cmap"]

                # Save image
                image_path = os.path.join(folder, f"{i}.png")
                self._save_img(image, color, image_path)

                # Save metadata as joblib file
                metadata = {"label": label, "meta": meta}
                metadata_path = os.path.join(folder, f"{i}.joblib")
                joblib.dump(metadata, metadata_path)

        if compression_type is not None:
            self.compress_dataset(compression_type, keep_only_compressed)

    def compress_dataset(self, compression_type, keep_only_compressed=False):
        import gzip
        import zipfile
        import tarfile

        for folder in [self.train_path, self.val_path, self.test_path]:
            log("info", f"Compressing files in {folder} using {compression_type}...")

            files_to_compress = [
                f for f in os.listdir(folder) if not f.endswith(f".{compression_type}")
            ]

            if not files_to_compress:
                log("info", "No files to compress.")
                return

            compressed_file_path = os.path.join(
                self.output_path, f"{folder.split('/')[-1]}.{compression_type}"
            )

            if compression_type == "zip":
                with zipfile.ZipFile(
                    compressed_file_path, "w", zipfile.ZIP_DEFLATED
                ) as zipf:
                    for file in tqdm(files_to_compress, desc="Compressing"):
                        file_path = os.path.join(folder, file)
                        zipf.write(file_path, os.path.basename(file_path))

            elif compression_type == "gzip":
                with open(compressed_file_path, "wb") as gzipped_file:
                    with gzip.GzipFile(fileobj=gzipped_file, mode="wb") as zipf:
                        for file in tqdm(files_to_compress, desc="Compressing"):
                            file_path = os.path.join(folder, file)
                            with open(file_path, "rb") as f:
                                zipf.write(f.read())

            elif compression_type == "tar.gz":
                with tarfile.open(compressed_file_path, "w:gz") as tarf:
                    for file in tqdm(files_to_compress, desc="Compressing"):
                        file_path = os.path.join(folder, file)
                        tarf.add(file_path, arcname=os.path.basename(file_path))

            elif compression_type == "bz2":
                with tarfile.open(compressed_file_path, "w:bz2") as tarf:
                    for file in tqdm(files_to_compress, desc="Compressing"):
                        file_path = os.path.join(folder, file)
                        tarf.add(file_path, arcname=os.path.basename(file_path))

            else:
                log("info", f"Unsupported compression type: {compression_type}")
                return

            log("info", f"Compression for {folder} complete.")

            # Delete the original files if the user wants
            if keep_only_compressed:
                log("info", f"Deleting uncompressed {folder} samples..")
                for file in tqdm(files_to_compress, desc="Deleting"):
                    file_path = os.path.join(folder, file)
                    os.remove(file_path)
                log("info", f"Done!")
