from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA


def read_passwords(passwords_file_path):
    """Read passwords from file."""
    with open(passwords_file_path, "r", encoding="utf-8") as file:
        passwords = file.read().splitlines()
    return passwords


def normalize_coordinates(data, lower_bound=-1, upper_bound=1):
    """
    Normalize the coordinates to be within the given bounds.
    """
    scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
    return scaler.fit_transform(data)


def train_tfidf(data, ngram_range=(3, 5)):
    """Train a TF-IDF model."""
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
    )
    model = vectorizer.fit_transform(data)
    return model


def dimensionality_reduction_method(data, n_dimensions=2, n_jobs=8):
    """
    Reduce the dimensionality of the data.
    In this case, we are using t-SNE, but you can use any other method you prefer.
    """
    tsne = TSNE(
        n_components=n_dimensions,
        n_jobs=n_jobs,
        perplexity=30,
        early_exaggeration=12,
        learning_rate=200,
        n_iter=1000,
        random_state=42,
    )
    return tsne.fit_transform(data)


def write_to_csv(password_vectors, output_file):
    """Write the stacked array to a CSV file with passwords as the first column."""

    print("Writing to CSV...")

    # Convert passwords to a NumPy array and reshape to a column vector
    output_array = np.array(password_vectors)

    # Write to CSV
    np.savetxt(output_file, output_array, delimiter=",", fmt="%s", encoding="utf-8")


def pca(data, n_dimensions=2):
    """Reduce the dimensionality of the data using PCA."""
    pca = PCA(n_components=n_dimensions)
    return pca.fit_transform(data)


def main(passwords_file_path):
    passwords = read_passwords(passwords_file_path)[:10000]

    print("training tfidf...")
    model = train_tfidf(passwords)

    print("pca...")
    print(model.toarray().shape)
    pcad = pca(model.toarray(), 30)
    print("embedding")
    print(pcad.shape)
    embedded = dimensionality_reduction_method(pcad, 3)
    # embedded = normalize_coordinates(embedded)

    output_data = np.column_stack((passwords, embedded))
    write_to_csv(output_data, "output.csv")


if __name__ == "__main__":
    passwords_file_path = "passwords.txt"  # replace with your file path
    main(passwords_file_path)
