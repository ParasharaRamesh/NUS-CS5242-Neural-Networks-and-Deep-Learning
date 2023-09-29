#some common functions related to plotting
import matplotlib.pyplot as plt
import random
import torch

def plot_model_training_stats(experiment, epochs, training_losses, validation_losses, training_accuracy, validation_accuracy):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot data on each subplot and add labels
    axes[0,0].plot(epochs, training_losses)
    axes[0,0].set_title(f'{experiment}: Training Loss vs Epochs')
    axes[0,0].set_xlabel('Epochs')
    axes[0,0].set_ylabel('Training Loss')

    axes[0,1].plot(epochs, training_accuracy)
    axes[0,1].set_title(f'{experiment}: Training Accuracy vs Epochs')
    axes[0,1].set_xlabel('Epochs')
    axes[0,1].set_ylabel('Training Accuracy')

    axes[1,0].plot(epochs, validation_losses)
    axes[1,0].set_title(f'{experiment}: Validation Loss vs Epochs')
    axes[1,0].set_xlabel('Epochs')
    axes[1,0].set_ylabel('Validation Loss')

    axes[1,1].plot(epochs, validation_accuracy)
    axes[1,1].set_title(f'{experiment}: Validation Accuracy vs Epochs')
    axes[1,1].set_xlabel('Epochs')
    axes[1,1].set_ylabel('Validation Accuracy')

    # Add space between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    #close it properly
    plt.clf()
    plt.cla()
    plt.close()

def show_predictions(model, test_data_loader, num_samples=5, class_names=None):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Get random samples
    sample_indices = random.sample(range(len(test_data_loader.dataset)), num_samples)

    # Create a subplot grid
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 8))

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # Get a random sample from the data loader
            sample_data, true_label = test_data_loader.dataset[idx]
            sample_data = sample_data.unsqueeze(0).to(device)

            # Forward pass through the model
            predicted_logits = model(sample_data)
            predicted_label = torch.argmax(predicted_logits, dim=1).item()

            # Determine title color based on correctness of prediction
            if predicted_label == true_label:
                title_color = 'green'  # Correct prediction
            else:
                title_color = 'red'  # Incorrect prediction

            # Plot image with title showing Ground Truth and Predicted labels
            title = f'Ground Truth: {class_names[true_label]}\nPredicted: {class_names[predicted_label]}'
            axes[i].imshow(sample_data.squeeze().cpu().permute(1, 2, 0))
            axes[i].set_title(title, color=title_color)
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()