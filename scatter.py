import csv
import matplotlib.pyplot as plt


def plot_passwords(filename):
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)

    passwords = [row[0] for row in data]
    x_coords = [float(row[1]) for row in data]
    y_coords = [float(row[2]) for row in data]

    fig, ax = plt.subplots()
    ax.scatter(x_coords, y_coords)

    # for i, txt in enumerate(passwords):
    # ax.annotate(txt, (x_coords[i], y_coords[i]))

    plt.show()


if __name__ == "__main__":
    plot_passwords("output.csv")
