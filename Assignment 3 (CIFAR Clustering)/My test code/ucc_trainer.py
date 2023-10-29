from torch import nn, optim
from tqdm.auto import tqdm
import os
from params import *
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

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

        #Transforms
        self.tensor_to_img_transform = transforms.ToPILImage()

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
                ucc_loss, batch_ucc_accuracy = self.forward_propogate_ucc(decoded, one_hot_ucc_labels, True)

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

                # scheduler update (remove if it doesnt work!)
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
    def init_params_from_checkpoint_hook(self, load_from_checkpoint, resume_epoch_num):
        if load_from_checkpoint:
            # NOTE: resume_epoch_num can be None here if we want to load from the most recently saved checkpoint!
            checkpoint_path = self.get_model_checkpoint_path(resume_epoch_num)
            checkpoint = torch.load(checkpoint_path)

            # load previous state of models
            self.autoencoder_model.load_state_dict(checkpoint['ae_model_state_dict'])
            self.ucc_predictor_model.load_state_dict(checkpoint['ucc_model_state_dict'])

            # load previous state of optimizers
            self.ae_optimizer.load_state_dict(checkpoint['ae_optimizer_state_dict'])
            self.ucc_optimizer.load_state_dict(checkpoint['ucc_optimizer_state_dict'])

            # Things we are keeping track of
            self.start_epoch = checkpoint['epoch']
            self.epoch_numbers = checkpoint['epoch_numbers']

            self.training_losses = checkpoint['training_losses']
            self.training_ae_losses = checkpoint['training_ae_losses']
            self.training_ucc_losses = checkpoint['training_ucc_losses']
            self.training_ucc_accuracies = checkpoint['training_ucc_accuracies']

            self.val_losses = checkpoint['val_losses']
            self.val_ae_losses = checkpoint['val_ae_losses']
            self.val_ucc_losses = checkpoint['val_ucc_losses']
            self.val_ucc_accuracies = checkpoint['val_ucc_accuracies']

            print(f"Model checkpoint for {self.name} is loaded from {checkpoint_path}!")

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
        # generally batch size of 16 is good for cifar10 so predicting 20 won't be so bad
        batch_size, bag_size, num_channels, height, width = images.size()
        bag_images = images.view(batch_size * bag_size, num_channels, height, width)
        encoded, decoded = self.autoencoder_model(bag_images)  # we are feeding in Batch*bag images of shape (3,32,32)
        ae_loss = self.ae_loss_criterion(decoded, bag_images)  # compares (Batch * Bag, 3,32,32)
        return ae_loss, encoded, decoded

    def forward_propogate_ucc(self, decoded, one_hot_ucc_labels, is_train_mode=True):
        # decoded is of shape [Batch * Bag, 48*16] ->  make it into shape [Batch, Bag, 48*16]
        batch_times_bag_size, feature_size = decoded.size()
        bag_size = config.bag_size
        batch_size = batch_times_bag_size // bag_size
        decoded = decoded.view(batch_size, bag_size, feature_size)
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
        if is_train_mode:
            self.train_correct_predictions += batch_correct_predictions
            self.train_total_batches += batch_size
        else:
            self.eval_correct_predictions += batch_correct_predictions
            self.eval_total_batches += batch_size
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
        # class level init
        self.eval_correct_predictions = 0
        self.eval_total_batches = 0

        val_loss = 0.0
        val_ae_loss = 0.0
        val_ucc_loss = 0.0

        # set all models to eval mode
        self.autoencoder_model.eval()
        self.ucc_predictor_model.eval()

        with torch.no_grad():
            for val_batch_idx, val_data in enumerate(self.val_loader):
                val_images, val_one_hot_ucc_labels = val_data

                # calculate losses from both models for a batch of bags
                val_batch_ae_loss, val_encoded, val_decoded = self.forward_propagate_autoencoder(val_images)
                val_batch_ucc_loss, val_batch_ucc_accuracy = self.forward_propogate_ucc(val_decoded,
                                                                                        val_one_hot_ucc_labels, False)

                # calculate combined loss
                val_batch_loss = val_batch_ae_loss + val_batch_ucc_loss

                # cummulate the losses
                val_ae_loss += val_batch_ae_loss
                val_ucc_loss += val_batch_ucc_loss
                val_loss += val_batch_loss

        # Calculate average validation loss for the epoch
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_ucc_loss = val_ucc_loss / len(self.val_loader)
        avg_val_ae_loss = val_ae_loss / len(self.val_loader)
        avg_val_ucc_training_accuracy = self.eval_correct_predictions / self.eval_total_batches

        # show some sample predictions
        self.show_sample_reconstructions(self.val_loader)

        return {
            "avg_val_loss": avg_val_loss,
            "avg_val_ae_loss": avg_val_ae_loss,
            "avg_val_ucc_loss": avg_val_ucc_loss,
            "avg_val_ucc_training_accuracy": avg_val_ucc_training_accuracy
        }

    def calculate_and_print_epoch_stats_hook(self, avg_train_stats, avg_val_stats):
        epoch_loss = avg_train_stats["avg_training_loss"]
        epoch_ae_loss = avg_train_stats["avg_ae_loss"]
        epoch_ucc_loss = avg_train_stats["avg_ucc_loss"]
        epoch_ucc_accuracy = avg_train_stats["avg_ucc_training_accuracy"]

        epoch_val_loss = avg_val_stats["avg_val_loss"]
        epoch_val_ae_loss = avg_val_stats["avg_val_ae_loss"]
        epoch_val_ucc_loss = avg_val_stats["avg_val_ucc_loss"]
        epoch_val_ucc_accuracy = avg_val_stats["avg_val_ucc_training_accuracy"]

        print(
            f"[TRAIN]: Epoch Loss: {epoch_loss} | AE Loss: {epoch_ae_loss} | UCC Loss: {epoch_ucc_loss} | UCC Acc: {epoch_ucc_accuracy}")
        print(
            f"[VAL]: Val Loss: {epoch_val_loss} | Val AE Loss: {epoch_val_ae_loss} | Val UCC Loss: {epoch_val_ucc_loss} | Val UCC Acc: {epoch_val_ucc_accuracy}")

        return {
            "epoch_loss": epoch_loss,
            "epoch_ae_loss": epoch_ae_loss,
            "epoch_ucc_loss": epoch_ucc_loss,
            "epoch_ucc_acc": epoch_ucc_accuracy,
            "epoch_val_loss": epoch_val_loss,
            "epoch_val_ae_loss": epoch_val_ae_loss,
            "epoch_val_ucc_loss": epoch_val_ucc_loss,
            "epoch_val_ucc_acc": epoch_val_ucc_accuracy
        }

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

    def get_current_running_history_state_hook(self):
        return self.epoch_numbers, \
            self.training_ae_losses, self.training_ucc_losses, self.training_losses, self.training_ucc_accuracies, \
            self.val_ae_losses, self.val_ucc_losses, self.val_losses, self.val_ucc_accuracies

    def save_model_checkpoint_hook(self, epoch, avg_train_stats, avg_val_stats):
        # set it to train mode to save the weights (but doesn't matter apparently!)
        self.autoencoder_model.train()
        self.ucc_predictor_model.train()

        # create the directory if it doesn't exist
        model_save_directory = os.path.join(self.save_dir, self.name)
        os.makedirs(model_save_directory, exist_ok=True)

        # Checkpoint the model at the end of each epoch
        checkpoint_path = os.path.join(model_save_directory, f'model_epoch_{epoch + 1}.pt')
        torch.save(
            {
                'ae_model_state_dict': self.autoencoder_model.state_dict(),
                'ucc_model_state_dict': self.ucc_predictor_model.state_dict(),
                'ae_optimizer_state_dict': self.ae_optimizer.state_dict(),
                'ucc_optimizer_state_dict': self.ucc_optimizer.state_dict(),
                'epoch': epoch + 1,
                'epoch_numbers': self.epoch_numbers,
                'training_losses': self.training_losses,
                'training_ae_losses': self.training_ae_losses,
                'training_ucc_losses': self.training_ucc_losses,
                'training_ucc_accuracies': self.training_ucc_accuracies,
                'val_losses': self.val_losses,
                'val_ae_losses': self.val_ae_losses,
                'val_ucc_losses': self.val_ucc_losses,
                'val_ucc_accuracies': self.val_ucc_accuracies,
            },
            checkpoint_path
        )
        print(f"Saved the model checkpoint for experiment {self.name} for epoch {epoch + 1}")

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

    def show_sample_reconstructions(self, dataloader):
        self.autoencoder_model.eval()

        # Create a subplot grid
        fig, axes = plt.subplots(1, 2, figsize=(9, 9))

        with torch.no_grad():
            for val_data in dataloader:
                val_images, _ = val_data

                batch_size, bag_size, num_channels, height, width = val_images.size()
                bag_val_images = val_images.view(batch_size * bag_size, num_channels, height, width)

                # Forward pass through the model
                _, _, val_reconstructed_images = self.forward_propagate_autoencoder(val_images)

                # take only one image from the bag
                sample_image = bag_val_images[0]
                predicted_image = val_reconstructed_images[0]

                #get it to cpu
                sample_image = sample_image.to("cpu")
                predicted_image = predicted_image.to("cpu")

                # convert to PIL Image
                sample_image = self.tensor_to_img_transform(sample_image)
                predicted_image = self.tensor_to_img_transform(predicted_image)

                axes[0].imshow(sample_image)
                axes[0].set_title(f"Sample Original Image", color='green')
                axes[0].axis('off')

                axes[1].imshow(predicted_image)
                axes[1].set_title(f"Sample Reconstructed Image", color='red')
                axes[1].axis('off')

                # show only one image
                break

        plt.tight_layout()
        plt.show()

    def test_model(self):
        # class level init
        self.eval_correct_predictions = 0
        self.eval_total_batches = 0

        test_loss = 0.0
        test_ae_loss = 0.0
        test_ucc_loss = 0.0

        # set all models to eval mode
        self.autoencoder_model.eval()
        self.ucc_predictor_model.eval()

        with torch.no_grad():
            for test_batch_idx, test_data in enumerate(self.test_loader):
                test_images, test_one_hot_ucc_labels = test_data

                # calculate losses from both models for a batch of bags
                test_batch_ae_loss, test_encoded, test_decoded = self.forward_propagate_autoencoder(test_images)
                test_batch_ucc_loss, test_batch_ucc_accuracy = self.forward_propogate_ucc(test_decoded,
                                                                                        test_one_hot_ucc_labels, False)

                # calculate combined loss
                test_batch_loss = test_batch_ae_loss + test_batch_ucc_loss

                # cummulate the losses
                test_ae_loss += test_batch_ae_loss
                test_ucc_loss += test_batch_ucc_loss
                test_loss += test_batch_loss

        # Calculate average validation loss for the epoch
        avg_test_loss = test_loss / len(self.val_loader)
        avg_test_ucc_loss = test_ucc_loss / len(self.val_loader)
        avg_test_ae_loss = test_ae_loss / len(self.val_loader)
        avg_test_ucc_training_accuracy = self.eval_correct_predictions / self.eval_total_batches

        # show some sample predictions
        self.show_sample_reconstructions(self.test_loader)

        return {
            "avg_test_loss": avg_test_loss,
            "avg_test_ae_loss": avg_test_ae_loss,
            "avg_test_ucc_loss": avg_test_ucc_loss,
            "avg_test_ucc_training_accuracy": avg_test_ucc_training_accuracy
        }

    def calculate_min_js_divergence(self):
        pass

    def calculate_clustering_accuracy(self):
        pass
