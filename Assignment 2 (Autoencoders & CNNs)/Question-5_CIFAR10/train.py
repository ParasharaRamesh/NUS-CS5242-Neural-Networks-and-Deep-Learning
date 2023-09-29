import torch
from torch import optim, nn
import os
from tqdm.auto import tqdm


def load_checkpointed_model_params(model, optimizer, resume_checkpoint):
    checkpoint = torch.load(resume_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    # Things we are keeping track of
    epoch_numbers = checkpoint['epoch_numbers']
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    training_accuracy = checkpoint['training_accuracy']
    validation_accuracy = checkpoint['validation_accuracy']
    print(f"Model checkpoint {resume_checkpoint} loaded! Will resume the epochs from number #{start_epoch}")
    return start_epoch, epoch_numbers, training_losses, training_accuracy, validation_losses, validation_accuracy


def save_model_checkpoint(experiment, model, optimizer, params, epoch, epoch_numbers, training_losses, validation_losses, training_accuracy, validation_accuracy):
    # Checkpoint the model at the end of each epoch
    checkpoint_path = os.path.join(params["save_dir"], experiment, f'model_epoch_{epoch + 1}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
        'epoch_numbers': epoch_numbers,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'training_accuracy': training_accuracy,
        'validation_accuracy': validation_accuracy,
    }, checkpoint_path)
    print(f"Save checkpointed the model at the path {checkpoint_path}")


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


'''
TODO: things to introduce.
* early stopping
* the moment I interrupt the program it should immeaditely checkpoint the state I want
* In the tqdm bar I want to see the validation loss, final training loss, training accuracy and validation accuracy and also print it just to be safe
* make it faster.
'''


def train_model(model, train_loader, val_loader, num_epochs, params, experiment, epoch_saver_count=5, resume_checkpoint=None):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Things we are keeping track of
    start_epoch = 0
    epoch_numbers = []
    training_losses = []
    validation_losses = []
    training_accuracy = []
    validation_accuracy = []

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # loss
    criterion = nn.CrossEntropyLoss()

    # load checkpoint
    if resume_checkpoint:
        start_epoch, epoch_numbers, training_losses, training_accuracy, validation_losses, validation_accuracy = load_checkpointed_model_params(
            model,
            optimizer,
            resume_checkpoint
        )

    # Custom progress bar for total epochs with color and displaying average epoch loss
    total_progress_bar = tqdm(total=num_epochs, desc=f"Total Epochs", position=0,
                              bar_format="{desc}: {percentage}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                              dynamic_ncols=True, ncols=100, colour='red')

    # training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_training_loss = 0.0
        train_correct_predictions = 0
        total_samples = 0

        # Custom progress bar for each epoch with color
        epoch_progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}",
                                  position=1, leave=False, dynamic_ncols=True, ncols=100, colour='green')

        for batch_idx, data in enumerate(train_loader):
            # boilerplate
            optimizer.zero_grad()

            # get the data and outputs
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # training loss update
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            epoch_training_loss += loss.item()

            # Compute training accuracy for this batch
            predicted = torch.argmax(outputs.data, 1)

            # batch stats
            batch_correct_predictions = (predicted == labels).sum().item()
            batch_size = labels.size(0)

            total_samples += batch_size  # batch size basically
            train_correct_predictions += batch_correct_predictions

            # Update the epoch progress bar (overwrite in place)
            epoch_progress_bar.set_postfix({
                "loss": loss.item(),
                "batch_acc": batch_correct_predictions / batch_size
            })
            epoch_progress_bar.update(1)

        # Close the epoch progress bar
        # epoch_progress_bar.close()

        # Calculate average loss for the epoch
        avg_training_loss_for_epoch = epoch_training_loss / len(train_loader)

        # Calculate training accuracy for the epoch
        avg_training_accuracy = train_correct_predictions / total_samples

        # Validation loop
        avg_val_accuracy, avg_val_loss_for_epoch = perform_validation(criterion, device, model, val_loader)

        # Store values
        training_accuracy.append(avg_training_accuracy)
        training_losses.append(avg_training_loss_for_epoch)
        validation_accuracy.append(avg_val_accuracy)
        validation_losses.append(avg_val_loss_for_epoch)
        epoch_numbers.append(epoch + 1)

        # Update the total progress bar
        total_progress_bar.set_postfix(
            {
                "loss": avg_training_loss_for_epoch,
                "train_acc": avg_training_accuracy,
                "val_loss": avg_val_loss_for_epoch,
                "val_acc": avg_val_accuracy,
            }

        )
        total_progress_bar.update(1)

        # Print state
        print(
            f'Epoch {epoch + 1}: train_loss: {avg_training_loss_for_epoch} | train_accuracy: {avg_training_accuracy} | val_loss: {avg_val_loss_for_epoch} | val_accuracy: {avg_val_accuracy} '
        )

        # Save model checkpoint periodically
        need_to_save_model_checkpoint = (epoch + 1) % epoch_saver_count == 0
        if need_to_save_model_checkpoint:
            save_model_checkpoint(
                experiment,
                model,
                optimizer,
                params,
                epoch,
                epoch_numbers,
                training_losses,
                validation_losses,
                training_accuracy,
                validation_accuracy
            )

    # Close the total progress bar
    total_progress_bar.close()

    # Return things needed for plotting
    return epoch_numbers, training_losses, training_accuracy, validation_losses, validation_accuracy

# %%
