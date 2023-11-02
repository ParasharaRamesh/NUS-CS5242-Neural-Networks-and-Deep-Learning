from torch import optim
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm
import os
from params import *
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from loss import *
from device_data_loader import *


class AutoencoderTrainer:
    def __init__(self,
                 name, autoencoder_model,
                 dataset, save_dir, device=config.device,
                 batch_size=config.bag_size * config.batch_size):
        self.name = name
        self.save_dir = save_dir
        self.device = device

        # data
        self.dataset = dataset
        self.train_dataset = ConcatDataset(dataset.train_sub_datasets)
        self.val_dataset = ConcatDataset(dataset.val_sub_datasets)

        self.train_loader = DeviceDataLoader(self.train_dataset, batch_size)
        self.val_loader = DeviceDataLoader(self.val_dataset, batch_size)

        # create the directory if it doesn't exist!
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, self.name), exist_ok=True)

        self.autoencoder_model = autoencoder_model

        # Adam optimizer(s)
        self.ae_optimizer = optim.Adam(self.autoencoder_model.parameters(), lr=config.learning_rate,
                                       weight_decay=config.weight_decay)

        # Loss criterion(s)
        self.ae_loss_criterion = nn.MSELoss()
        # self.ae_loss_criterion = SSIMLoss()

        # Transforms
        self.tensor_to_img_transform = transforms.ToPILImage()

        # Values which can change based on loaded checkpoint
        self.start_epoch = 0
        self.epoch_numbers = []
        self.training_ae_losses = []
        self.val_ae_losses = []

    # main train code
    def train(self,
              num_epochs,
              resume_epoch_num=None,
              load_from_checkpoint=False,
              epoch_saver_count=2):
        torch.cuda.empty_cache()

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

            # set the epoch training batch_loss
            epoch_ae_loss = 0.0

            # iterate over each batch
            for batch_idx, data in enumerate(self.train_loader):
                images, _ = data

                # forward propogate through the combined model
                encoded, decoded = self.autoencoder_model(images)

                # calculate losses from both models for a batch of bags
                ae_loss = self.calculate_autoencoder_loss(images, decoded)

                # do loss backward for all losses
                ae_loss.backward()

                # Gradient clipping (commenting this out as it is causing colab to crash!)
                nn.utils.clip_grad_value_(self.autoencoder_model.parameters(), config.grad_clip)

                # do optimizer step and zerograd for autoencoder model
                self.ae_optimizer.step()
                self.ae_optimizer.zero_grad()

                # scheduler update (remove if it doesnt work!)
                self.ae_scheduler.step()

                # add to epoch batch_loss
                epoch_ae_loss += ae_loss.item()

                # Update the epoch progress bar (overwrite in place)
                batch_stats = {
                    "ae_loss": ae_loss.item()
                }

                epoch_progress_bar.set_postfix(batch_stats)
                epoch_progress_bar.update(1)

            # close the epoch progress bar
            epoch_progress_bar.close()

            # calculate average epoch train statistics
            avg_train_stats = self.calculate_avg_train_stats_hook(epoch_ae_loss)

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
                self.save_model_checkpoint_hook(epoch)

            print("-" * 60)

        # Close the total progress bar
        total_progress_bar.close()

        # Return the current state
        return self.get_current_running_history_state_hook()

    # hooks
    def init_params_from_checkpoint_hook(self, load_from_checkpoint, resume_epoch_num):
        if load_from_checkpoint:
            # NOTE: resume_epoch_num can be None here if we want to load from the most recently saved checkpoint!
            checkpoint_path = self.get_model_checkpoint_path(resume_epoch_num)
            checkpoint = torch.load(checkpoint_path)

            # load previous state of models
            self.autoencoder_model.load_state_dict(checkpoint['ae_model_state_dict'])

            # load previous state of optimizers
            self.ae_optimizer.load_state_dict(checkpoint['ae_optimizer_state_dict'])

            # Things we are keeping track of
            self.start_epoch = checkpoint['epoch']
            self.epoch_numbers = checkpoint['epoch_numbers']

            self.training_ae_losses = checkpoint['training_ae_losses']
            self.val_ae_losses = checkpoint['val_ae_losses']

            print(f"Model checkpoint for {self.name} is loaded from {checkpoint_path}!")

    def init_scheduler_hook(self, num_epochs):
        # here we are doing it at a bag level
        self.ae_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.ae_optimizer,
            config.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader)
        )

    def calculate_autoencoder_loss(self, images, decoded):
        # data is of shape (batchsize=2,bag=10,channels=3,height=32,width=32)
        # generally batch size of 16 is good for cifar10 so predicting 20 won't be so bad
        ae_loss = self.ae_loss_criterion(decoded, images)  # compares (Batch * Bag, 3,32,32)
        return ae_loss

    def calculate_avg_train_stats_hook(self, epoch_ae_loss):
        no_of_bags = len(self.train_loader) * config.batch_size
        avg_ae_loss_for_epoch = epoch_ae_loss / no_of_bags

        epoch_train_stats = {
            "avg_ae_loss": avg_ae_loss_for_epoch
        }

        return epoch_train_stats

    def validation_hook(self):
        val_ae_loss = 0.0

        with torch.no_grad():
            # set all models to eval mode
            self.autoencoder_model.eval()

            for val_batch_idx, val_data in enumerate(self.val_loader):
                val_images, _ = val_data

                # forward propogate through the model
                val_encoded, val_decoded = self.autoencoder_model(val_images)

                # calculate losses from both models for a batch of bags
                val_batch_ae_loss = self.calculate_autoencoder_loss(val_images, val_decoded)

                # cummulate the losses
                val_ae_loss += val_batch_ae_loss.item()

        # Calculate average validation loss for the epoch
        no_of_bags = len(self.val_loader) * config.batch_size
        avg_val_ae_loss = val_ae_loss / no_of_bags

        print("Now showing a sample reconstruction")

        # show some sample predictions
        self.show_sample_reconstructions(self.val_loader)

        return {
            "avg_val_ae_loss": avg_val_ae_loss
        }

    def calculate_and_print_epoch_stats_hook(self, avg_train_stats, avg_val_stats):
        epoch_ae_loss = avg_train_stats["avg_ae_loss"]
        epoch_val_ae_loss = avg_val_stats["avg_val_ae_loss"]

        print(f"[TRAIN]: Epoch AE Loss: {epoch_ae_loss}")
        print(f"[VAL]: Val AE Loss: {epoch_val_ae_loss}")

        return {
            "epoch_ae_loss": epoch_ae_loss,
            "epoch_val_ae_loss": epoch_val_ae_loss
        }

    def store_running_history_hook(self, epoch, avg_train_stats, avg_val_stats):
        self.epoch_numbers.append(epoch + 1)

        self.training_ae_losses.append(avg_train_stats["avg_ae_loss"])

        self.val_ae_losses.append(avg_val_stats["avg_val_ae_loss"])

    def get_current_running_history_state_hook(self):
        return self.epoch_numbers, self.training_ae_losses, self.val_ae_losses

    def save_model_checkpoint_hook(self, epoch):
        # set it to train mode to save the weights (but doesn't matter apparently!)
        self.autoencoder_model.train()

        # create the directory if it doesn't exist
        model_save_directory = os.path.join(self.save_dir, self.name)
        os.makedirs(model_save_directory, exist_ok=True)

        # Checkpoint the model at the end of each epoch
        checkpoint_path = os.path.join(model_save_directory, f'model_epoch_{epoch + 1}.pt')
        torch.save(
            {
                'ae_model_state_dict': self.autoencoder_model.state_dict(),
                'ae_optimizer_state_dict': self.ae_optimizer.state_dict(),
                'epoch': epoch + 1,
                'epoch_numbers': self.epoch_numbers,
                'training_ae_losses': self.training_ae_losses,
                'val_ae_losses': self.val_ae_losses
            },
            checkpoint_path
        )
        print(f"Saved the model checkpoint for experiment {self.name} for epoch {epoch + 1}")

    def show_sample_reconstructions(self, dataloader):
        # Create a subplot grid
        fig, axes = plt.subplots(1, 2, figsize=(3, 3))

        with torch.no_grad():
            # set all models to eval mode
            self.autoencoder_model.eval()

            for val_data in dataloader:
                val_images, _ = val_data

                # forward propagate through the model
                _, val_reconstructed_images = self.autoencoder_model(val_images)
                print("Got a sample reconstruction, now trying to reshape in order to show an example")

                # take only one image from the bag
                sample_image = val_images[0]
                predicted_image = val_reconstructed_images[0]

                # get it to cpu
                sample_image = sample_image.to("cpu")
                predicted_image = predicted_image.to("cpu")

                # convert to PIL Image
                sample_image = self.tensor_to_img_transform(sample_image)
                predicted_image = self.tensor_to_img_transform(predicted_image)

                axes[0].imshow(sample_image)
                axes[0].set_title(f"Orig", color='green')
                axes[0].axis('off')

                axes[1].imshow(predicted_image)
                axes[1].set_title(f"Recon", color='red')
                axes[1].axis('off')

                # show only one image
                break

        plt.tight_layout()
        plt.show()

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
