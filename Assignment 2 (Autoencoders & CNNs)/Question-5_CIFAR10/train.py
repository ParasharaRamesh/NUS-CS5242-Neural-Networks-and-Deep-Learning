import torch
from torch import optim, nn
import os
from tqdm.auto import tqdm
from model import *
from data import *
from torch.cuda.amp import autocast, GradScaler
from validate_and_test import *


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


def save_model_checkpoint(experiment, model, optimizer, params, epoch, epoch_numbers, training_losses,
                          validation_losses, training_accuracy, validation_accuracy):
    # create the directory if it doesnt exist
    model_save_directory = os.path.join(params["save_dir"], experiment)
    os.makedirs(model_save_directory, exist_ok=True)

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


def train_model(model, train_loader, val_loader, num_epochs, params, experiment, epoch_saver_count=5,
                resume_checkpoint=None):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()

    # Things we are keeping track of
    start_epoch = 0
    epoch_numbers = []
    training_losses = []
    validation_losses = []
    training_accuracy = []
    validation_accuracy = []

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, params['learning_rate'],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )

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

            # dummy init
            outputs = None
            loss = None

            # Wrap the forward pass and loss computation with autocast
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss = criterion(outputs, labels)

            # training loss update (Multi precision training for faster training)
            scaler.scale(loss).backward()

            # Gradient clipping
            nn.utils.clip_grad_value_(model.parameters(), params['grad_clip'])

            #update optimizer but using GradientScaler mixed precision training
            scaler.step(optimizer)
            scaler.update()

            #scheduler update
            sched.step()

            epoch_training_loss += loss.item()

            # batch stats
            # Compute training accuracy for this batch
            predicted = torch.argmax(outputs.data, 1)
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
        epoch_progress_bar.close()

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

        #Close the tqdm bat
        total_progress_bar.update(1)

        # Print state
        print(
            f'Epoch {epoch + 1}: train_loss: {avg_training_loss_for_epoch} | train_accuracy: {avg_training_accuracy} | val_loss: {avg_val_loss_for_epoch} | val_accuracy: {avg_val_accuracy} '
        )

        # Save model checkpoint periodically
        need_to_save_model_checkpoint = (epoch + 1) % epoch_saver_count == 0
        if need_to_save_model_checkpoint:
            print(f"Going to save model @ Epoch:{epoch + 1}")
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
if __name__ == '__main__':
    params = {
        'batch_size': 32,
        'learning_rate': 0.0045,
        'save_dir': 'model_ckpts'
    }
    train_data_loader = create_train_data_loader(32)
    test_data_loader, validation_data_loader = create_test_and_validation_data_loader(32)

    full_experiment = "Full Data"
    # Check if GPU is available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_cifar_model = CIFARClassifier()
    full_cifar_model.to(device)

    train_model(
        full_cifar_model,
        train_data_loader,
        validation_data_loader,
        2,
        params,
        full_experiment,
        epoch_saver_count=1,
        resume_checkpoint=None
    )
