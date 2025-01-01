import matplotlib.pyplot as plt


def plot_best_loss(file_path):
    # Lists to store tournament and best_loss values
    tournaments = []
    best_losses = []

    # Read the file and extract relevant data
    with open(file_path, 'r') as file:
        idx = 0
        for line in file:
            if "tournament=" in line and "best_loss=" in line:
                parts = line.split(',')
                tournament = int(parts[0].split('=')[1].strip())
                best_loss = float(parts[1].split('=')[1].strip())
                idx += 1
                tournaments.append(idx)
                best_losses.append(best_loss)

    # Plot best_loss over tournament
    plt.figure(figsize=(10, 6))
    plt.plot(tournaments, best_losses, linewidth=0.7, label="Best MSE (px^2)")
    plt.title("Genetic Algorithm (Self-Play)")
    plt.xlabel("Tournament")
    plt.ylabel("Best MSE (px^2)")
    plt.grid(True)
    plt.legend()
    plt.show()


# Example usage
file_path = "data.txt"  # Replace with the path to your document
plot_best_loss(file_path)