import matplotlib.pyplot as plt
import numpy as np
import torch

# set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_ucc_model_stats(
        experiment, epochs,
        ucc_training_losses, ae_training_losses, combined_training_losses,
        ucc_training_accuracy,
        ucc_validation_losses, ae_validation_losses, combined_validation_losses,
        ucc_validation_accuracy
    ):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # Plot training losses
    axes[0, 0].plot(epochs, ucc_training_losses, marker="o", color="red", label="UCC Training Loss")
    axes[0, 0].plot(epochs, ae_training_losses, marker="o", color="blue", label="AE Training Loss")
    axes[0, 0].plot(epochs, combined_training_losses, marker="o", color="green", label="Combined Training Loss")
    axes[0, 0].set_title(f'{experiment}: Training Loss vs Epochs')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].legend()  # Display the legend

    # Plot training accuracy
    axes[0, 1].plot(epochs, ucc_training_accuracy, marker="o", color="red", label="UCC Training Accuracy")
    axes[0, 1].set_title(f'{experiment}: Training Accuracy vs Epochs')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Training Accuracy')
    axes[0, 1].legend()  # Display the legend

    # Plot validation losses
    axes[1, 0].plot(epochs, ucc_validation_losses, marker="o", color="red", label="UCC Validation Loss")
    axes[1, 0].plot(epochs, ae_validation_losses, marker="o", color="blue", label="AE Validation Loss")
    axes[1, 0].plot(epochs, combined_validation_losses, marker="o", color="green", label="Combined Validation Loss")
    axes[1, 0].set_title(f'{experiment}: Validation Loss vs Epochs')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].legend()  # Display the legend

    # Plot validation accuracy 1,1
    axes[1, 1].plot(epochs, ucc_validation_accuracy, marker="o", color="red", label="UCC Validation Accuracy")
    axes[1, 1].set_title(f'{experiment}: Validation Accuracy vs Epochs')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].legend()  # Display the legend

    # Add space between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    # close it properly
    plt.clf()
    plt.cla()
    plt.close()


def plot_ucc_rcc_model_stats(
        experiment, epochs,
        ucc_training_losses, ae_training_losses, rcc_training_losses, combined_training_losses,
        ucc_training_accuracy, rcc_training_accuracy,
        ucc_validation_losses, ae_validation_losses, rcc_validation_losses, combined_validation_losses,
        ucc_validation_accuracy, rcc_validation_accuracy
    ):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # Plot training losses
    axes[0, 0].plot(epochs, ucc_training_losses, marker="o", color="red", label="UCC Training Loss")
    axes[0, 0].plot(epochs, ae_training_losses, marker="o", color="blue", label="AE Training Loss")
    axes[0, 0].plot(epochs, rcc_training_losses, marker="o", color="yellow", label="RCC Training Loss")
    axes[0, 0].plot(epochs, combined_training_losses, marker="o", color="green", label="Combined Training Loss")
    axes[0, 0].set_title(f'{experiment}: Training Loss vs Epochs')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].legend()  # Display the legend

    # Plot training accuracy
    axes[0, 1].plot(epochs, ucc_training_accuracy, marker="o", color="red", label="UCC Training Accuracy")
    axes[0, 1].plot(epochs, rcc_training_accuracy, marker="o", color="green", label="RCC Training Accuracy")
    axes[0, 1].set_title(f'{experiment}: Training Accuracy vs Epochs')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Training Accuracy')
    axes[0, 1].legend()  # Display the legend

    # Plot validation losses
    axes[1, 0].plot(epochs, ucc_validation_losses, marker="o", color="red", label="UCC Validation Loss")
    axes[1, 0].plot(epochs, ae_validation_losses, marker="o", color="blue", label="AE Validation Loss")
    axes[1, 0].plot(epochs, rcc_validation_losses, marker="o", color="yellow", label="RCC Validation Loss")
    axes[1, 0].plot(epochs, combined_validation_losses, marker="o", color="green", label="Combined Validation Loss")
    axes[1, 0].set_title(f'{experiment}: Validation Loss vs Epochs')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].legend()  # Display the legend

    # Plot validation accuracy 1,1
    axes[1, 1].plot(epochs, ucc_validation_accuracy, marker="o", color="red", label="UCC Validation Accuracy")
    axes[1, 1].plot(epochs, rcc_validation_accuracy, marker="o", color="green", label="RCC Validation Accuracy")
    axes[1, 1].set_title(f'{experiment}: Validation Accuracy vs Epochs')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].legend()  # Display the legend

    # Add space between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    # close it properly
    plt.clf()
    plt.cla()
    plt.close()
