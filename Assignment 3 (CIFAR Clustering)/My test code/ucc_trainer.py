from torch import nn, optim
from tqdm.auto import tqdm
import os
from params import *


class UCCTrainer:
    def __init__(self,
                 name, autoencoder_model, ucc_predictor_model,
                 train_loader, test_loader, val_loader,
                 save_dir, device=config.device):
        self.name = name
        self.save_dir = save_dir
        self.device = device
        self.train_loader, self.test_loader, self.val_loader = train_loader, test_loader, val_loader

        # create the directory if it doesn't exist!
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, self.name), exist_ok=True)

        self.autoencoder_model = autoencoder_model
        self.ucc_predictor_model = ucc_predictor_model

        # Adam optimizer(s)
        self.ae_optimizer = optim.Adam(self.autoencoder_model.parameters(), lr=config.learning_rate,
                                       weight_decay=config.weight_decay)
        self.ucc_optimizer = optim.Adam(self.ucc_predictor_model.parameters(), lr=config.learning_rate,
                                        weight_decay=config.weight_decay)

        # Loss criterion(s)
        self.ae_loss_criterion = nn.MSELoss()
        self.ucc_loss_criterion = nn.CrossEntropyLoss()

        # Values which can change based on loaded checkpoint
        self.start_epoch = 0
        self.epoch_numbers = []
        self.training_ae_losses = []
        self.training_ucc_losses = []
        self.training_losses = []
        self.training_ucc_accuracies = []

        self.val_ae_losses = []
        self.val_ucc_losses = []
        self.val_losses = []
        self.val_ucc_accuracies = []

        self.train_correct_predictions = 0
        self.train_total_batches = 0

    # main train code
    def train(self,
              num_epochs,
              resume_epoch_num=None,
              load_from_checkpoint=False,
              epoch_saver_count=2):
        torch.cuda.empty_cache()

        # TODO.x 1 need to change this
        # initialize the params from the saved checkpoint
        self.init_params_from_checkpoint_hook(load_from_checkpoint, resume_epoch_num)

        # set up scheduler
        self.init_scheduler_hook(num_epochs)

        # Custom progress bar for total epochs with color and displaying average epoch batch_loss
        total_progress_bar = tqdm(
            total=num_epochs, desc=f"Total Epochs", position=0,
            bar_format="{desc}: {percentage}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True, ncols=100, colour='red'
        )

        # Train loop
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            # Custom progress bar for each epoch with color
            epoch_progress_bar = tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{self.start_epoch + num_epochs}",
                position=1,
                leave=False,
                dynamic_ncols=True,
                ncols=100,
                colour='green'
            )

            # set all models to train mode
            self.autoencoder_model.train()
            self.ucc_predictor_model.train()

            # set the epoch training batch_loss
            epoch_training_loss = 0.0
            epoch_ae_loss = 0.0
            epoch_ucc_loss = 0.0

            # iterate over each batch
            for batch_idx, data in enumerate(self.train_loader):
                images, one_hot_ucc_labels = data

                # calculate losses from both models for a batch of bags
                ae_loss, encoded, decoded = self.forward_propagate_autoencoder(images)
                ucc_loss, batch_ucc_accuracy = self.forward_propogate_ucc(decoded, one_hot_ucc_labels)

                # calculate combined loss
                batch_loss = ae_loss + ucc_loss

                # Gradient clipping
                nn.utils.clip_grad_value_(self.autoencoder_model.parameters(), config.grad_clip)
                nn.utils.clip_grad_value_(self.ucc_predictor_model.parameters(), config.grad_clip)

                # do optimizer step and zerograd for autoencoder model
                self.ae_optimizer.step()
                self.ae_optimizer.zero_grad()

                # do optimizer step and zerograd for ucc model
                self.ucc_optimizer.step()
                self.ucc_optimizer.zero_grad()

                # scheduler update
                self.ae_scheduler.step()
                self.ucc_scheduler.step()

                # add to epoch batch_loss
                epoch_training_loss += batch_loss.item()
                epoch_ae_loss += ae_loss.item()
                epoch_ucc_loss += ucc_loss.item()

                # Update the epoch progress bar (overwrite in place)
                batch_stats = {
                    "batch_loss": batch_loss.item(),
                    "ae_loss": ae_loss.item(),
                    "ucc_loss": ucc_loss.item(),
                    "batch_ucc_acc": batch_ucc_accuracy
                }

                epoch_progress_bar.set_postfix(batch_stats)
                epoch_progress_bar.update(1)

            # close the epoch progress bar
            epoch_progress_bar.close()

            # calculate average epoch train statistics
            avg_train_stats = self.calculate_avg_train_stats_hook(epoch_training_loss, epoch_ae_loss, epoch_ucc_loss)

            # TODO.x13 have to calculate things like training accuracy, training batch_loss for all models
            # calculate validation statistics
            avg_val_stats = self.validation_hook()

            # Store running history
            self.store_running_history_hook(epoch, avg_train_stats, avg_val_stats)

            # Show epoch stats
            print(f"# Epoch {epoch + 1}")
            epoch_postfix = self.calculate_and_print_epoch_stats_hook(avg_train_stats, avg_val_stats)

            # Update the total progress bar
            total_progress_bar.set_postfix(epoch_postfix)

            # Close tqdm bar
            total_progress_bar.update(1)

            # Save model checkpoint periodically
            need_to_save_model_checkpoint = (epoch + 1) % epoch_saver_count == 0
            if need_to_save_model_checkpoint:
                print(f"Going to save model {self.name} @ Epoch:{epoch + 1}")
                self.save_model_checkpoint_hook(epoch, avg_train_stats, avg_val_stats)

            print("-" * 60)

        # Close the total progress bar
        total_progress_bar.close()

        # Return the current state
        return self.get_current_running_history_state_hook()

    # hooks
    def init_params_from_checkpoint_hook(self, load_from_checkpoint, resume_checkpoint):
        raise NotImplementedError("Need to implement hook for initializing params from checkpoint")

    def init_scheduler_hook(self, num_epochs):
        # steps per epoch here is multiplied with bag size as we are doing it at an image level
        self.ae_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.ae_optimizer,
            config.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader) * config.bag_size
        )

        # here we are doing it at a bag level
        self.ucc_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.ucc_optimizer,
            config.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader)
        )

    def forward_propagate_autoencoder(self, images):
        # data is of shape (batchsize=2,bag=10,channels=3,height=32,width=32)
        # generally batch size of 16 is good for cifar10 so predicting 20 wont be so bad
        batch_size, bag_size, num_channels, height, width = images.size()
        images = images.view(batch_size * bag_size, num_channels, height, width)
        encoded, decoded = self.autoencoder_model(images)  # we are feeding in Batch*bag images of shape (3,32,32)
        ae_loss = self.ae_loss_criterion(decoded, images)  # compares (Batch * Bag, 3,32,32)
        return ae_loss, encoded, decoded

    def forward_propogate_ucc(self, decoded, one_hot_ucc_labels):
        # decoded is of shape [Batch * Bag, 48*16] ->  make it into shape [Batch, Bag, 48*16]
        batch_size, bag_size = config.batch_size, config.bag_size
        decoded = decoded.view(batch_size, bag_size, -1)
        ucc_logits = self.ucc_predictor_model(decoded)

        # compute the ucc_loss
        ucc_loss = self.ucc_loss_criterion(ucc_logits, one_hot_ucc_labels)

        # compute the batch stats right here and save it
        ucc_probs = nn.Softmax(dim=1)(ucc_logits)
        predicted = torch.argmax(ucc_probs, 1)
        labels = torch.argmax(one_hot_ucc_labels, 1)
        batch_correct_predictions = (predicted == labels).sum().item()
        batch_size = labels.size(0)

        # calculate batchwise accuracy/ucc_loss
        batch_ucc_accuracy = batch_correct_predictions / batch_size
        self.train_correct_predictions += batch_correct_predictions
        self.train_total_batches += batch_size

        return ucc_loss, batch_ucc_accuracy

    def calculate_avg_train_stats_hook(self, epoch_training_loss, epoch_ae_loss, epoch_ucc_loss):
        avg_training_loss_for_epoch = epoch_training_loss / len(self.train_loader)
        avg_ae_loss_for_epoch = epoch_ae_loss / len(self.train_loader)
        avg_ucc_loss_for_epoch = epoch_ucc_loss / len(self.train_loader)
        avg_ucc_training_accuracy = self.train_correct_predictions / self.train_total_batches

        epoch_train_stats = {
            "avg_training_loss": avg_training_loss_for_epoch,
            "avg_ae_loss": avg_ae_loss_for_epoch,
            "avg_ucc_loss": avg_ucc_loss_for_epoch,
            "avg_ucc_training_accuracy": avg_ucc_training_accuracy
        }

        # reset
        self.train_correct_predictions = 0
        self.train_total_batches = 0

        return epoch_train_stats

    def validation_hook(self):
        raise NotImplementedError("Need to implement this hook to calculate the validation stats")

    def calculate_and_print_epoch_stats_hook(self):
        raise NotImplementedError(
            "Need to implement this hook to calculate and print the epoch statistics and return the postfix dictinoary")

    def store_running_history_hook(self, epoch, avg_train_stats, avg_val_stats):
        self.epoch_numbers.append(epoch + 1)

        self.training_ae_losses.append(avg_train_stats["avg_ae_loss"])
        self.training_ucc_losses.append(avg_train_stats["avg_ucc_loss"])
        self.training_losses.append(avg_train_stats["avg_training_loss"])
        self.training_ucc_accuracies.append(avg_train_stats["avg_ucc_training_accuracy"])

        self.val_ae_losses.append(avg_val_stats["avg_val_ae_loss"])
        self.val_ucc_losses.append(avg_val_stats["avg_val_ucc_loss"])
        self.val_losses.append(avg_val_stats["avg_val_loss"])
        self.val_ucc_accuracies.append(avg_val_stats["avg_val_ucc_training_accuracy"])

    def save_model_checkpoint_hook(self, epoch, avg_train_stats, avg_val_stats):
        raise NotImplementedError("Need to implement this hook to save the model checkpoints")

    def get_current_running_history_state_hook(self):
        return self.epoch_numbers, \
            self.training_ae_losses, self.training_ucc_losses, self.training_losses, self.training_ucc_accuracies, \
            self.val_ae_losses, self.val_ucc_losses, self.val_losses, self.val_ucc_accuracies

    # TODO.x check all the functions below this!
    # find the most recent file and return the path
    def get_model_checkpoint_path(self, epoch_num=None):
        directory = os.path.join(self.save_dir, self.name)
        if epoch_num == None:
            # Get a list of all files in the directory
            files = os.listdir(directory)

            # Filter out only the files (exclude directories)
            files = [f for f in files if os.path.isfile(os.path.join(directory, f))]

            # Sort the files by their modification time in descending order (most recent first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

            # Get the name of the most recently added file
            model_file = files[0] if files else None
        else:
            model_file = f"model_epoch_{epoch_num}.pt"
        return os.path.join(directory, model_file)

    def test_model(self):
        raise NotImplementedError("Need to implement this hook to return the history after training the model")
