import argparse
from polyleven import levenshtein
from sklearn.decomposition import IncrementalPCA
import numpy as np
import random
import ray


def distance(a, b):
    return levenshtein(a, b)


@ray.remote(num_cpus=4)
def generate_sub_distance_matrix(passwords, bucket):
    matrix = np.zeros((len(bucket), len(passwords)))

    for i, b in enumerate(bucket):
        for j, p in enumerate(passwords):
            matrix[i, j] = distance(p, b)

    return matrix.T


def generate_distance_matrix(passwords, bucket):
    """Generate a distance matrix for a list of passwords against the bucket."""

    ray.init()

    matrix = np.zeros((len(bucket), len(passwords)))

    n_processes = 4

    remotes = []

    for i in range(0, n_processes):
        start = int((i / n_processes) * len(passwords))
        end = int(((i + 1) / n_processes) * len(passwords))

        remotes.append(
            generate_sub_distance_matrix.remote(bucket, passwords[start:end])
        )

    matrix = np.hstack(ray.get(remotes))

    ray.shutdown()

    return matrix


def split_into_buckets(passwords, bucket_size):
    """Split a list into sublists of a specific size."""
    return [
        passwords[i : i + bucket_size] for i in range(0, len(passwords), bucket_size)
    ]


def write_to_csv(stacked, passwords, output_file):
    """Write the stacked array to a CSV file with passwords as the first column."""

    # Convert passwords to a NumPy array and reshape to a column vector
    passwords = np.array(passwords).reshape((-1, 1))

    # Prepend passwords to the stacked array
    output_array = np.hstack((passwords, stacked))

    # Write to CSV
    np.savetxt(output_file, output_array, delimiter=",", fmt="%s")


def main(
    passwords_file_path,
    bucket_size,
    shuffle,
    ipca_components,
    limit=None,
    output_file=None,
):
    with open(passwords_file_path, "r", encoding="utf-8") as file:
        passwords = file.read().splitlines()

        if shuffle:
            random.shuffle(passwords)

        if limit:
            passwords = passwords[:limit]

    buckets = split_into_buckets(passwords, bucket_size)

    transformer = IncrementalPCA(n_components=ipca_components)

    stacked = []

    for i, bucket in enumerate(buckets):
        print(f"Processing bucket {i + 1}/{len(buckets)}")
        matrix = generate_distance_matrix(passwords, bucket)

        transformed_matrix = transformer.fit_transform(matrix)

        stacked.append(transformed_matrix)

    stacked = np.vstack(stacked)

    if output_file:
        write_to_csv(stacked, passwords, output_file)

    print(stacked.shape)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide passwords into buckets.")
    parser.add_argument(
        "passwords_file_path", type=str, help="Path to the password file."
    )
    parser.add_argument(
        "-b",
        "--bucket_size",
        type=int,
        default=320,
        help="Size of the password buckets.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Number of passwords to use from list.",
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the passwords",
    )
    parser.add_argument(
        "-i",
        "--ipca_components",
        type=int,
        default=30,
        help="Number of components to decompose to with IPCA",
    )
    args = parser.parse_args()

    main(
        args.passwords_file_path,
        args.bucket_size,
        args.shuffle,
        args.ipca_components,
        args.limit,
        args.output_file,
    )
