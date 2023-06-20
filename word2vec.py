from gensim.models import Word2Vec


def read_passwords(passwords_file_path):
    """Read passwords from file."""
    with open(passwords_file_path, "r", encoding="utf-8") as file:
        passwords = file.read().splitlines()
    return passwords


def prepare_data(passwords):
    """Prepare data for Word2Vec (treat each character as a word)."""
    data = [[char for char in password] for password in passwords]
    return data


def train_word2vec(data, vector_size=100, window=5, min_count=1, workers=4):
    """Train a Word2Vec model."""
    model = Word2Vec(
        sentences=data,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
    )
    return model


def main(passwords_file_path):
    passwords = read_passwords(passwords_file_path)
    data = prepare_data(passwords)
    model = train_word2vec(data)

    # save model for later use
    model.save("word2vec.model")


if __name__ == "__main__":
    passwords_file_path = "passwords.txt"  # replace with your file path
    main(passwords_file_path)
