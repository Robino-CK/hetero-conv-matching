import numpy as np
import matplotlib.pyplot as plt

def plot_line(original_acc, coarsend_acc, original_loss, coar_loss, epochs):
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    
    plt.plot(epochs, original_loss, label="Original Graph")
    plt.plot(epochs, coar_loss, label="Coarsened Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    epochs.append(epochs[-1] + 1)
    plt.plot(epochs, original_acc, 'o-', label="Original Graph")
    plt.plot(epochs, coarsend_acc, 's-', label="Coarsened Graph")
    #plt.plot(x, results["inverted_acc"], '^-', label="Inverted Coarsened")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Final comparison
    print("Final Performance Comparison:")
    print(f"Original Graph Accuracy: {original_acc[-1]:.4f}")
    print(f"Coarsened Graph Accuracy: {coarsend_acc[-1]:.4f}")



def plot_bar(original_accuracies, coarsened_accuracies):

    def summarize(name, values):
        values = np.array(values)
        print(f"\n{name}:")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Std:  {values.std():.4f}")
        return values

    orig_vals = summarize("Original Graph Accuracy", original_accuracies)
    coar_vals = summarize("Coarsened Graph Accuracy", coarsened_accuracies)

    labels = [f"Run {i + 1}" for i in range(len(original_accuracies))]
    x = np.arange(len(original_accuracies))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, orig_vals, width, label='Original')
    plt.bar(x, coar_vals, width, label='Coarsened')
    plt.xlabel('Run')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Across Multiple Runs')
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.show()
