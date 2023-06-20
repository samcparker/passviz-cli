import argparse
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


def dimensionality_reduction_method(data, n_dimensions=2, n_jobs=1):
    """
    Reduce the dimensionality of the data.
    In this case, we are using t-SNE, but you can use any other method you prefer.
    """
    tsne = TSNE(n_components=n_dimensions, n_jobs=n_jobs)
    return tsne.fit_transform(data)


def normalize_coordinates(data, lower_bound=-1, upper_bound=1):
    """
    Normalize the coordinates to be within the given bounds.
    """
    scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
    return scaler.fit_transform(data)


def main(input_file, output_file, n_dimensions, n_jobs):
    # Load data from CSV
    data = np.genfromtxt(
        input_file, delimiter=",", dtype=str, invalid_raise=False, encoding="utf-8"
    )

    # Separate passwords and data
    passwords = data[:, 0]
    data = data[:, 1:].astype(float)

    # Perform dimensionality reduction
    reduced_data = dimensionality_reduction_method(data, n_dimensions, n_jobs)

    # Normalize the coordinates
    reduced_data = normalize_coordinates(reduced_data)

    # Append passwords to reduced_data
    output_data = np.column_stack((passwords, reduced_data))

    # Write to CSV
    np.savetxt(output_file, output_data, delimiter=",", fmt="%s", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform dimensionality reduction.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="reduced_data.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "-n",
        "--n_dimensions",
        type=int,
        default=2,
        help="Number of dimensions to reduce to.",
    )
    parser.add_argument(
        "-j",
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs.",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.n_dimensions, args.n_jobs)
