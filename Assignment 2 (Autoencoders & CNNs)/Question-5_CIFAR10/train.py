import torch
from torch import optim
from tqdm.auto import tqdm

def train_model(params, model, train_loader, num_epochs, save_dir, resume_checkpoint=None, old_training_loss=None, old_epoch_numbers=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = params['loss_criterion']  # L1 loss
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Model checkpoint loaded, will resume the epochs from number #{start_epoch}")
    else:
        start_epoch = 0

    # Custom progress bar for total epochs with color and displaying average epoch loss
    total_progress_bar = tqdm(total=num_epochs, desc=f"Total Epochs", position=0, bar_format="{desc}: {percentage}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]", dynamic_ncols=True, ncols=100, colour='red')

    # Lists to store training loss and epoch number
    training_loss = old_training_loss if old_training_loss else []
    epoch_numbers = old_epoch_numbers if old_epoch_numbers else []

    checkpointed_paths = []
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0.0

        # Custom progress bar for each epoch with color
        epoch_progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}", position=1, leave=False, dynamic_ncols=True, ncols=100, colour='green')

        for batch_idx, data in enumerate(train_loader):
            images, _ = data
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update the epoch progress bar (overwrite in place)
            epoch_progress_bar.set_postfix(loss=loss.item())
            epoch_progress_bar.update(1)

        # Close the epoch progress bar
        epoch_progress_bar.close()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(train_loader)

        # Append training loss and epoch number to lists
        training_loss.append(average_loss)
        epoch_numbers.append(epoch + 1)

        # Print and save checkpoint
        print(f'Epoch {epoch + 1} Loss: {average_loss}')

        # Update the total progress bar
        total_progress_bar.set_postfix(loss=average_loss)
        total_progress_bar.update(1)

        # Save only once in 5 epochs
        if (epoch + 1) % 5 == 0:
            # Checkpoint the model at the end of each epoch
            checkpoint_path = os.path.join(save_dir, str(Lambda), f'model_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
            }, checkpoint_path)
            checkpointed_paths.append(f"Checkpointed the model at the path {checkpoint_path}")

    # Close the total progress bar
    total_progress_bar.close()

    # Print what was checkpointed
    for checkpoint_path in checkpointed_paths:
        print(checkpoint_path)

    # Return training loss and epoch numbers for plotting
    return training_loss, epoch_numbers
