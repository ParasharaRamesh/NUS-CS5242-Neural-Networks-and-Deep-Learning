import torch


def perform_validation(criterion, device, model, val_loader):
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0
    total_val_samples = 0
    with torch.no_grad():
        for val_batch_idx, val_data in enumerate(val_loader):
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_images)

            # Validation loss update
            val_loss += criterion(val_outputs, val_labels).item()

            # Compute validation accuracy for this batch
            val_predicted = torch.argmax(val_outputs.data, 1)
            total_val_samples += val_labels.size(0)
            val_correct_predictions += (val_predicted == val_labels).sum().item()

    # Calculate average validation loss for the epoch
    avg_val_loss_for_epoch = val_loss / len(val_loader)
    # Calculate validation accuracy for the epoch
    avg_val_accuracy = val_correct_predictions / total_val_samples
    return avg_val_accuracy, avg_val_loss_for_epoch


def perform_test(criterion, device, model, test_loader):
    model.eval()
    test_loss = 0.0
    test_correct_predictions = 0
    total_val_samples = 0
    with torch.no_grad():
        for val_batch_idx, val_data in enumerate(test_loader):
            test_images, test_labels = val_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            val_outputs = model(test_images)

            # Validation loss update
            test_loss += criterion(val_outputs, test_labels).item()

            # Compute validation accuracy for this batch
            test_predicted = torch.argmax(val_outputs.data, 1)
            total_val_samples += test_labels.size(0)
            test_correct_predictions += (test_predicted == test_labels).sum().item()

    # Calculate average validation loss for the epoch
    avg_test_loss = test_loss / len(test_loader)
    # Calculate validation accuracy for the epoch
    avg_test_accuracy = test_correct_predictions / total_val_samples
    return avg_test_accuracy, avg_test_loss
