from gensim.models import Word2Vec
import numpy as np


def write_to_csv(password_vectors, output_file):
    """Write the stacked array to a CSV file with passwords as the first column."""

    print("Writing to CSV...")

    # Convert passwords to a NumPy array and reshape to a column vector
    output_array = np.array(password_vectors)

    print(output_array.shape)
    print(output_array[11])

    # Write to CSV
    np.savetxt(output_file, output_array, delimiter=",", fmt="%s", encoding="utf-8")


def read_passwords(passwords_file_path):
    """Read passwords from file."""
    with open(passwords_file_path, "r", encoding="utf-8") as file:
        passwords = file.read().splitlines()
    return passwords


def get_password_vector(model, password):
    """Get the vector representation of a password by averaging the vectors of its characters."""
    vectors = [
        model.wv[char] for char in password if char in model.wv.key_to_index
    ]  # ensure the char is in the trained model's vocabulary
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def main():
    model = Word2Vec.load("word2vec.model")  # load the trained model
    passwords = read_passwords("./passwords.txt")

    password_vectors = []

    for i, password in enumerate(passwords):
        if i % 1000 == 0:
            print(f"Processing password {i + 1}/{len(passwords)}")
        password_vectors.append([password, *get_password_vector(model, password)])

    write_to_csv(password_vectors, "password_vectors.csv")


if __name__ == "__main__":
    main()
