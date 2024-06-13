# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import argparse, json, os
import tqdm
import gzip
import zipfile
import tarfile

"""
Compress the CLEVR images according to the specified compression
"""

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="../out")
parser.add_argument("--output_dir", default="../compressed")
parser.add_argument("--version", default="1.0")
parser.add_argument(
    "--output-compression",
    choices=[None, "bz2", "gzip", "zip", "gzip", "tar.gz"],
    required=False,
    default=None,
    help="Output compression format.",
)
parser.add_argument(
    "--keep-only-compressed",
    action="store_true",
    default=False,
    required=False,
    help="Keep only the compressed folders",
)


def compress_dataset(
    output_dir, input_folder, compression_type, keep_only_compressed=False
):
    """Compress the dataset"""
    for folder in input_folder:
        print(f"Compressing files in {folder} using {compression_type}...")

        files_to_compress = [
            f for f in os.listdir(folder) if not f.endswith(f".{compression_type}")
        ]

        if not files_to_compress:
            print("No files to compress.")
            return

        compressed_file_path = os.path.join(
            output_dir, f"{folder.split('/')[-1]}.{compression_type}"
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
            print(f"Unsupported compression type: {compression_type}")
            return

        print(f"Compression for {folder} complete.")

        # Delete the original files if the user wants
        if keep_only_compressed:
            print(f"Deleting uncompressed {folder} samples..")
            for file in tqdm(files_to_compress, desc="Deleting"):
                file_path = os.path.join(folder, file)
                os.remove(file_path)
            print(f"Done!")


def main(args):
    compress_dataset(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        version=args.version,
        compression_type=args.output_compression,
        keep_only_compressed=args.keep_only_compressed,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
