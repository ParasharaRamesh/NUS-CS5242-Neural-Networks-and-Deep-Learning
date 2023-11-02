import torch
import numpy as np
from torch import nn, optim
from tqdm.auto import tqdm
import os
from params import *
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from loss import *


class RCCTrainer:
    def __init__(self,
                 name, rcc_model,
                 dataset, save_dir, device=config.device,
                 importances={"ae": 0.2, "ucc": 0.4, "rcc": 0.4}
                 ):
        self.name = name
        self.save_dir = save_dir
        self.device = device

        # importances
        self.ae_importance = importances["ae"]
        self.ucc_importance = importances["ucc"]
        self.rcc_importance = importances["rcc"]

        # data
        self.train_loader = dataset.ucc_rcc_train_dataloader
        self.test_loader = dataset.ucc_rcc_test_dataloader
        self.val_loader = dataset.ucc_rcc_val_dataloader
        self.kde_loaders = dataset.kde_test_dataloaders  # each dataloader here will return shape of (batch, bag, 3,32,32) of a pure dataset
        self.autoencoder_loaders = dataset.autoencoder_test_dataloaders

        # create the directory if it doesn't exist!
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, self.name), exist_ok=True)

        self.rcc_model = rcc_model

        # Adam optimizer(s)
        self.rcc_optimizer = optim.Adam(self.rcc_model.parameters(), lr=config.learning_rate,
                                        weight_decay=config.weight_decay)

        # Loss criterion(s)
        self.ae_loss_criterion = SSIMLoss()
        self.ucc_loss_criterion = nn.CrossEntropyLoss()
        self.rcc_loss_criterion = nn.MSELoss()

        # Transforms
        self.tensor_to_img_transform = transforms.ToPILImage()

        # Values which can change based on loaded checkpoint
        self.start_epoch = 0
        self.epoch_numbers = []

        self.training_losses = []
        self.training_ae_losses = []
        self.training_ucc_losses = []
        self.training_rcc_losses = []
        self.training_ucc_accuracies = []
        self.training_rcc_accuracies = []

        self.val_losses = []
        self.val_ae_losses = []
        self.val_ucc_losses = []
        self.val_rcc_losses = []
        self.val_ucc_accuracies = []
        self.val_rcc_accuracies = []

        self.train_ucc_correct_predictions = 0
        self.train_ucc_total_batches = 0

        self.train_rcc_correct_predictions = 0
        self.train_rcc_total_batches = 0

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
            self.rcc_model.train()

            # set the epoch training batch_loss
            epoch_training_loss = 0.0
            epoch_ae_loss = 0.0
            epoch_ucc_loss = 0.0
            epoch_rcc_loss = 0.0

            # iterate over each batch
            for batch_idx, data in enumerate(self.train_loader):
                images, one_hot_ucc_labels, rcc_labels = data

                # forward propogate through the combined model
                rcc_logits, ucc_logits, decoded = self.rcc_model(images)

                # calculate losses from both models for a batch of bags
                ae_loss = self.calculate_autoencoder_loss(images, decoded)
                ucc_loss, batch_ucc_accuracy = self.calculate_ucc_loss_and_acc(ucc_logits, one_hot_ucc_labels, True)
                rcc_loss, batch_rcc_accuracy = self.calculate_rcc_loss_and_acc(rcc_logits, rcc_labels, True)

                # calculate combined loss
                batch_loss = (self.ae_importance * ae_loss) + (self.ucc_importance * ucc_loss) + (self.rcc_importance * rcc_loss)

                # do loss backward for all losses
                batch_loss.backward()

                # Gradient clipping(causing colab to crash!)
                nn.utils.clip_grad_value_(self.rcc_model.parameters(), config.grad_clip)

                # do optimizer step and zerograd
                self.rcc_optimizer.step()
                self.rcc_optimizer.zero_grad()

                # scheduler update (remove if it doesnt work!)
                self.rcc_scheduler.step()

                # add to epoch batch_loss
                epoch_training_loss += batch_loss.item()
                epoch_ae_loss += ae_loss.item()
                epoch_ucc_loss += ucc_loss.item()
                epoch_rcc_loss += rcc_loss.item()

                # Update the epoch progress bar (overwrite in place)
                batch_stats = {
                    "batch_loss": batch_loss.item(),
                    "ae_loss": ae_loss.item(),
                    "ucc_loss": ucc_loss.item(),
                    "rcc_loss": rcc_loss.item(),
                    "batch_ucc_acc": batch_ucc_accuracy,
                    "batch_rcc_acc": batch_rcc_accuracy
                }

                epoch_progress_bar.set_postfix(batch_stats)
                epoch_progress_bar.update(1)

            # close the epoch progress bar
            epoch_progress_bar.close()

            # calculate average epoch train statistics
            avg_train_stats = self.calculate_avg_train_stats_hook(epoch_training_loss, epoch_ae_loss, epoch_ucc_loss,
                                                                  epoch_rcc_loss)

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
    # DONE
    def init_params_from_checkpoint_hook(self, load_from_checkpoint, resume_epoch_num):
        if load_from_checkpoint:
            # NOTE: resume_epoch_num can be None here if we want to load from the most recently saved checkpoint!
            checkpoint_path = self.get_model_checkpoint_path(resume_epoch_num)
            checkpoint = torch.load(checkpoint_path)

            # load previous state of models
            self.rcc_model.load_state_dict(checkpoint['rcc_model_state_dict'])

            # load previous state of optimizers
            self.rcc_optimizer.load_state_dict(checkpoint['rcc_optimizer_state_dict'])

            # Things we are keeping track of
            self.start_epoch = checkpoint['epoch']
            self.epoch_numbers = checkpoint['epoch_numbers']

            self.training_losses = checkpoint['training_losses']
            self.training_ae_losses = checkpoint['training_ae_losses']
            self.training_ucc_losses = checkpoint['training_ucc_losses']
            self.training_rcc_losses = checkpoint['training_rcc_losses']
            self.training_ucc_accuracies = checkpoint['training_ucc_accuracies']
            self.training_rcc_accuracies = checkpoint['training_rcc_accuracies']

            self.val_losses = checkpoint['val_losses']
            self.val_ae_losses = checkpoint['val_ae_losses']
            self.val_ucc_losses = checkpoint['val_ucc_losses']
            self.val_rcc_losses = checkpoint['val_rcc_losses']
            self.val_ucc_accuracies = checkpoint['val_ucc_accuracies']
            self.val_rcc_accuracies = checkpoint['val_rcc_accuracies']

            print(f"Model checkpoint for {self.name} is loaded from {checkpoint_path}!")

    # DONE
    def init_scheduler_hook(self, num_epochs):
        self.rcc_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.rcc_optimizer,
            config.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader)
        )

    # DONE
    def calculate_autoencoder_loss(self, images, decoded):
        # data is of shape (batchsize=2,bag=10,channels=3,height=32,width=32)
        # generally batch size of 16 is good for cifar10 so predicting 20 won't be so bad
        batch_size, bag_size, num_channels, height, width = images.size()
        batches_of_bag_images = images.view(batch_size * bag_size, num_channels, height, width).to(torch.float32)
        ae_loss = self.ae_loss_criterion(decoded, batches_of_bag_images)  # compares (Batch * Bag, 3,32,32)
        return ae_loss

    # DONE
    def calculate_ucc_loss_and_acc(self, ucc_logits, one_hot_ucc_labels, is_train_mode=True):
        # compute the ucc_loss between [batch, 4]
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
            self.train_ucc_correct_predictions += batch_correct_predictions
            self.train_ucc_total_batches += batch_size
        else:
            self.eval_ucc_correct_predictions += batch_correct_predictions
            self.eval_ucc_total_batches += batch_size
        return ucc_loss, batch_ucc_accuracy

    # DONE
    '''
    NOTE: To improve this I can also add a rcc-ucc-enforcement loss where the number of unique classes should match the ucc exactly
    '''

    def calculate_rcc_loss_and_acc(self, rcc_logits, rcc_labels, is_train_mode=True):
        # compute the rcc_loss between [batch, 10] ( as there are 10 classes)
        batch_times_bag_size = config.batch_size * config.bag_size

        # round it to the nearest integer
        predicted = torch.round(rcc_logits).to(torch.float32)

        # compute the rcc_loss
        rcc_loss = self.rcc_loss_criterion(rcc_logits, rcc_labels)

        # NOTE: not sure if it is dim
        batch_correct_predictions = (predicted == rcc_labels).sum().item()

        # calculate batchwise accuracy/ucc_loss
        batch_rcc_accuracy = batch_correct_predictions / batch_times_bag_size
        if is_train_mode:
            self.train_rcc_correct_predictions += batch_correct_predictions
            self.train_rcc_total_batches += batch_times_bag_size
        else:
            self.eval_rcc_correct_predictions += batch_correct_predictions
            self.eval_rcc_total_batches += batch_times_bag_size
        return rcc_loss, batch_rcc_accuracy

    def calculate_avg_train_stats_hook(self, epoch_training_loss, epoch_ae_loss, epoch_ucc_loss, epoch_rcc_loss):
        no_of_bags = len(self.train_loader) * config.batch_size
        avg_training_loss_for_epoch = epoch_training_loss / no_of_bags
        avg_ae_loss_for_epoch = epoch_ae_loss / no_of_bags
        avg_ucc_loss_for_epoch = epoch_ucc_loss / no_of_bags
        avg_rcc_loss_for_epoch = epoch_rcc_loss / no_of_bags
        avg_ucc_training_accuracy = self.train_ucc_correct_predictions / self.train_ucc_total_batches
        avg_rcc_training_accuracy = self.train_rcc_correct_predictions / self.train_rcc_total_batches

        epoch_train_stats = {
            "avg_training_loss": avg_training_loss_for_epoch,
            "avg_ae_loss": avg_ae_loss_for_epoch,
            "avg_ucc_loss": avg_ucc_loss_for_epoch,
            "avg_rcc_loss": avg_rcc_loss_for_epoch,
            "avg_ucc_training_accuracy": avg_ucc_training_accuracy,
            "avg_rcc_training_accuracy": avg_rcc_training_accuracy
        }

        # reset
        self.train_ucc_correct_predictions = 0
        self.train_ucc_total_batches = 0

        self.train_rcc_correct_predictions = 0
        self.train_rcc_total_batches = 0

        return epoch_train_stats

    def validation_hook(self):
        # class level init
        self.eval_ucc_correct_predictions = 0
        self.eval_ucc_total_batches = 0

        self.eval_rcc_correct_predictions = 0
        self.eval_rcc_total_batches = 0

        val_loss = 0.0
        val_ae_loss = 0.0
        val_ucc_loss = 0.0
        val_rcc_loss = 0.0

        with torch.no_grad():
            # set all models to eval mode
            self.rcc_model.eval()

            for val_batch_idx, val_data in enumerate(self.val_loader):
                val_images, val_one_hot_ucc_labels, val_rcc_labels = val_data

                # forward propogate through the model
                val_rcc_logits, val_ucc_logits, val_decoded = self.rcc_model(val_images)

                # calculate losses from both models for a batch of bags
                val_batch_ae_loss = self.calculate_autoencoder_loss(val_images, val_decoded)
                val_batch_ucc_loss, val_batch_ucc_accuracy = self.calculate_ucc_loss_and_acc(val_ucc_logits,
                                                                                             val_one_hot_ucc_labels,
                                                                                             False)
                val_batch_rcc_loss, val_batch_rcc_accuracy = self.calculate_rcc_loss_and_acc(val_rcc_logits,
                                                                                             val_rcc_labels,
                                                                                             False)

                # calculate combined loss
                val_batch_loss = val_batch_ae_loss + val_batch_ucc_loss + val_batch_rcc_loss

                # cummulate the losses
                val_loss += val_batch_loss.item()
                val_ae_loss += val_batch_ae_loss.item()
                val_ucc_loss += val_batch_ucc_loss.item()
                val_rcc_loss += val_batch_rcc_loss.item()

        # Calculate average validation loss for the epoch
        no_of_bags = len(self.val_loader) * config.batch_size
        avg_val_loss = val_loss / no_of_bags
        avg_val_ucc_loss = val_ucc_loss / no_of_bags
        avg_val_ae_loss = val_ae_loss / no_of_bags
        avg_val_rcc_loss = val_rcc_loss / no_of_bags
        avg_val_ucc_training_accuracy = self.eval_ucc_correct_predictions / self.eval_ucc_total_batches
        avg_val_rcc_training_accuracy = self.eval_rcc_correct_predictions / self.eval_rcc_total_batches

        # show some sample predictions
        self.show_sample_reconstructions(self.val_loader)

        return {
            "avg_val_loss": avg_val_loss,
            "avg_val_ae_loss": avg_val_ae_loss,
            "avg_val_ucc_loss": avg_val_ucc_loss,
            "avg_val_rcc_loss": avg_val_rcc_loss,
            "avg_val_ucc_training_accuracy": avg_val_ucc_training_accuracy,
            "avg_val_rcc_training_accuracy": avg_val_rcc_training_accuracy
        }

    def calculate_and_print_epoch_stats_hook(self, avg_train_stats, avg_val_stats):
        epoch_loss = avg_train_stats["avg_training_loss"]
        epoch_ae_loss = avg_train_stats["avg_ae_loss"]
        epoch_ucc_loss = avg_train_stats["avg_ucc_loss"]
        epoch_rcc_loss = avg_train_stats["avg_rcc_loss"]
        epoch_ucc_accuracy = avg_train_stats["avg_ucc_training_accuracy"]
        epoch_rcc_accuracy = avg_train_stats["avg_rcc_training_accuracy"]

        epoch_val_loss = avg_val_stats["avg_val_loss"]
        epoch_val_ae_loss = avg_val_stats["avg_val_ae_loss"]
        epoch_val_ucc_loss = avg_val_stats["avg_val_ucc_loss"]
        epoch_val_rcc_loss = avg_val_stats["avg_val_rcc_loss"]
        epoch_val_ucc_accuracy = avg_val_stats["avg_val_ucc_training_accuracy"]
        epoch_val_rcc_accuracy = avg_val_stats["avg_val_rcc_training_accuracy"]

        print(
            f"[TRAIN]: Epoch Loss: {epoch_loss} | AE Loss: {epoch_ae_loss} | UCC Loss: {epoch_ucc_loss} | UCC Acc: {epoch_ucc_accuracy} | RCC Loss: {epoch_rcc_loss} | RCC Acc: {epoch_rcc_accuracy}")
        print(
            f"[VAL]: Val Loss: {epoch_val_loss} | Val AE Loss: {epoch_val_ae_loss} | Val UCC Loss: {epoch_val_ucc_loss} | Val UCC Acc: {epoch_val_ucc_accuracy} | Val RCC Loss: {epoch_val_rcc_loss} | Val RCC Acc: {epoch_val_rcc_accuracy}")

        return {
            "epoch_loss": epoch_loss,
            "epoch_ae_loss": epoch_ae_loss,
            "epoch_ucc_loss": epoch_ucc_loss,
            "epoch_rcc_loss": epoch_rcc_loss,
            "epoch_ucc_acc": epoch_ucc_accuracy,
            "epoch_rcc_acc": epoch_rcc_accuracy,
            "epoch_val_loss": epoch_val_loss,
            "epoch_val_ae_loss": epoch_val_ae_loss,
            "epoch_val_ucc_loss": epoch_val_ucc_loss,
            "epoch_val_rcc_loss": epoch_val_rcc_loss,
            "epoch_val_ucc_acc": epoch_val_ucc_accuracy,
            "epoch_val_rcc_acc": epoch_val_rcc_accuracy
        }

    def store_running_history_hook(self, epoch, avg_train_stats, avg_val_stats):
        self.epoch_numbers.append(epoch + 1)

        self.training_ae_losses.append(avg_train_stats["avg_ae_loss"])
        self.training_ucc_losses.append(avg_train_stats["avg_ucc_loss"])
        self.training_rcc_losses.append(avg_train_stats["avg_rcc_loss"])
        self.training_losses.append(avg_train_stats["avg_training_loss"])
        self.training_ucc_accuracies.append(avg_train_stats["avg_ucc_training_accuracy"])
        self.training_rcc_accuracies.append(avg_train_stats["avg_rcc_training_accuracy"])

        self.val_ae_losses.append(avg_val_stats["avg_val_ae_loss"])
        self.val_ucc_losses.append(avg_val_stats["avg_val_ucc_loss"])
        self.val_rcc_losses.append(avg_val_stats["avg_val_rcc_loss"])
        self.val_losses.append(avg_val_stats["avg_val_loss"])
        self.val_ucc_accuracies.append(avg_val_stats["avg_val_ucc_training_accuracy"])
        self.val_rcc_accuracies.append(avg_val_stats["avg_val_rcc_training_accuracy"])

    def get_current_running_history_state_hook(self):
        return self.epoch_numbers, \
            self.training_ae_losses, self.training_ucc_losses, self.training_rcc_losses, self.training_losses, self.training_ucc_accuracies, self.training_rcc_accuracies, \
            self.val_ae_losses, self.val_ucc_losses, self.val_rcc_losses, self.val_losses, self.val_ucc_accuracies, self.val_rcc_accuracies

    def save_model_checkpoint_hook(self, epoch):
        # set it to train mode to save the weights (but doesn't matter apparently!)
        self.rcc_model.train()

        # create the directory if it doesn't exist
        model_save_directory = os.path.join(self.save_dir, self.name)
        os.makedirs(model_save_directory, exist_ok=True)

        # Checkpoint the model at the end of each epoch
        checkpoint_path = os.path.join(model_save_directory, f'model_epoch_{epoch + 1}.pt')
        torch.save(
            {
                'rcc_model_state_dict': self.rcc_model.state_dict(),
                'rcc_optimizer_state_dict': self.rcc_optimizer.state_dict(),
                'epoch': epoch + 1,
                'epoch_numbers': self.epoch_numbers,
                'training_losses': self.training_losses,
                'training_ae_losses': self.training_ae_losses,
                'training_ucc_losses': self.training_ucc_losses,
                'training_rcc_losses': self.training_rcc_losses,
                'training_ucc_accuracies': self.training_ucc_accuracies,
                'training_rcc_accuracies': self.training_rcc_accuracies,
                'val_losses': self.val_losses,
                'val_ae_losses': self.val_ae_losses,
                'val_ucc_losses': self.val_ucc_losses,
                'val_rcc_losses': self.val_rcc_losses,
                'val_ucc_accuracies': self.val_ucc_accuracies,
                'val_rcc_accuracies': self.val_rcc_accuracies
            },
            checkpoint_path
        )
        print(f"Saved the model checkpoint for experiment {self.name} for epoch {epoch + 1}")

    def test_model(self):
        # class level init
        self.eval_ucc_correct_predictions = 0
        self.eval_ucc_total_batches = 0

        self.eval_rcc_correct_predictions = 0
        self.eval_rcc_total_batches = 0

        test_loss = 0.0
        test_ae_loss = 0.0
        test_ucc_loss = 0.0
        test_rcc_loss = 0.0

        with torch.no_grad():
            # set all models to eval mode
            self.rcc_model.eval()

            for test_batch_idx, test_data in enumerate(self.test_loader):
                test_images, test_one_hot_ucc_labels, test_rcc_labels = test_data

                # forward propogate through the model
                test_rcc_logits, test_ucc_logits, test_decoded = self.rcc_model(test_images)

                # calculate losses from both models for a batch of bags
                test_batch_ae_loss = self.calculate_autoencoder_loss(test_images, test_decoded)
                test_batch_ucc_loss, test_batch_ucc_accuracy = self.calculate_ucc_loss_and_acc(test_ucc_logits,
                                                                                               test_one_hot_ucc_labels,
                                                                                               False)
                test_batch_rcc_loss, test_batch_rcc_accuracy = self.calculate_rcc_loss_and_acc(test_rcc_logits,
                                                                                               test_rcc_labels,
                                                                                               False)

                # calculate combined loss
                test_batch_loss = test_batch_ae_loss + test_batch_ucc_loss + test_batch_rcc_loss

                # cummulate the losses
                test_loss += test_batch_loss.item()
                test_ae_loss += test_batch_ae_loss.item()
                test_ucc_loss += test_batch_ucc_loss.item()
                test_rcc_loss += test_batch_rcc_loss.item()

        # Calculate average validation loss for the epoch
        no_of_bags = len(self.test_loader) * config.batch_size
        avg_test_loss = test_loss / no_of_bags
        avg_test_ucc_loss = test_ucc_loss / no_of_bags
        avg_test_rcc_loss = test_rcc_loss / no_of_bags
        avg_test_ae_loss = test_ae_loss / no_of_bags
        avg_test_ucc_training_accuracy = self.eval_ucc_correct_predictions / self.eval_ucc_total_batches
        avg_test_rcc_training_accuracy = self.eval_rcc_correct_predictions / self.eval_rcc_total_batches

        # show some sample predictions
        self.show_sample_reconstructions(self.test_loader)

        return {
            "avg_test_loss": avg_test_loss,
            "avg_test_ae_loss": avg_test_ae_loss,
            "avg_test_ucc_loss": avg_test_ucc_loss,
            "avg_test_rcc_loss": avg_test_rcc_loss,
            "avg_test_ucc_training_accuracy": avg_test_ucc_training_accuracy,
            "avg_test_rcc_training_accuracy": avg_test_rcc_training_accuracy
        }

    def show_sample_reconstructions(self, dataloader):
        # Create a subplot grid
        fig, axes = plt.subplots(1, 2, figsize=(3, 3))

        with torch.no_grad():
            # set all models to eval mode
            self.rcc_model.eval()

            for val_data in dataloader:
                val_images, _, _ = val_data

                # reshape to appropriate size
                batch_size, bag_size, num_channels, height, width = val_images.size()
                bag_val_images = val_images.view(batch_size * bag_size, num_channels, height, width)
                print("Reshaped the original image into bag format")

                # forward propagate through the model
                _, _, val_reconstructed_images = self.rcc_model(val_images)
                print("Got a sample reconstruction, now trying to reshape in order to show an example")

                # take only one image from the bag
                sample_image = bag_val_images[0]
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

    def js_divergence(self, p, q):
        """
        Calculate the Jensen-Shannon Divergence between two probability distributions p and q.

        Args:
        p (torch.Tensor): Probability distribution p.
        q (torch.Tensor): Probability distribution q.

        Returns:
        torch.Tensor: Jensen-Shannon Divergence between p and q.
        """
        # Calculate the average distribution 'm'
        m = 0.5 * (p + q)

        # Calculate the KL Divergence of 'p' and 'q' from 'm'
        kl_div_p = F.kl_div(p.log(), m, reduction='batchmean')
        kl_div_q = F.kl_div(q.log(), m, reduction='batchmean')

        # Compute the JS Divergence
        js_divergence = 0.5 * (kl_div_p + kl_div_q)

        return js_divergence

    def calculate_min_js_divergence(self):
        num_classes = len(self.kde_loaders)
        kde_per_class = {class_idx: 0.0 for class_idx in range(num_classes)}

        # find the average kde across all classes
        for class_idx, pure_class_kde_loader in enumerate(self.kde_loaders):
            print(f"Kde Loader {class_idx} started!")
            num_bags_in_class = 0
            for images in tqdm(pure_class_kde_loader):
                # get the first element
                images = images[0]

                # Stage.1 pass through the encoder to get the latent features
                # batch data is of shape ( Batch,bag, 3,32,32)
                batch_size, bag_size, num_channels, height, width = images.size()
                # reshaping to shape ( batch * bag, 3 ,32,32)
                batches_of_bag_images = images.view(batch_size * bag_size, num_channels, height, width).to(
                    torch.float32)
                latent_features = self.rcc_model.autoencoder.encoder(
                    batches_of_bag_images)  # shape (Batch * bag, 48*16)
                latent_features = latent_features.to(torch.float32)  # shape (Batch * Bag, 256, 2, 2)

                # Stage.2 pass through KDE
                # encoded is of shape [Batch * Bag, 256,2,2] ->  make it into shape [Batch, Bag, 256*2*2]
                latent_features = latent_features.view(batch_size, bag_size, latent_features.size(1) * latent_features.size(2) * latent_features.size(3))

                batch_kde_distributions = self.rcc_model.ucc_predictor.kde(latent_features)  # shape [Batch, 1024*11]

                # Stage.3 Take sum
                num_bags_in_class += batch_kde_distributions.size(0)
                kde_distributions = torch.sum(batch_kde_distributions, dim=0)
                kde_per_class[class_idx] += kde_distributions

            # Stage.4 Take average
            kde_per_class[class_idx] /= num_bags_in_class
            print(f"Kde Loader {class_idx} done!")

        print("Computed all the kde's! Now finding min js divergence..")

        # find the js_divergence
        min_divergence = torch.inf
        best_i = None
        best_j = None
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                divergence = self.js_divergence(kde_per_class[i], kde_per_class[j])
                print(f"JS Divergence between {i} & {j} is {divergence}")
                if divergence < min_divergence:
                    min_divergence = divergence
                    best_i = i
                    best_j = j

        print(f"Min JS Divergence is {min_divergence} between classes {best_i} & {best_j}")
        # return the min divergence
        return min_divergence

    def calculate_clustering_accuracy(self):
        all_latent_features = []
        truth_labels_arr = []
        print(f"No of autoencoder loaders are {len(self.autoencoder_loaders)}")
        for autoencoder_idx, pure_autoencoder_loader in enumerate(self.autoencoder_loaders):
            print(f"Started pure autoencoder loader {autoencoder_idx} of length {len(pure_autoencoder_loader)}")
            for data in tqdm(pure_autoencoder_loader):
                # batch data is of shape (1,3,32,32), (1,1)
                image, label = data
                latent_features = self.rcc_model.autoencoder.encoder(image)  # shape (1, 48*16)
                latent_features = latent_features.view(-1)  # flatten

                latent_features = latent_features.squeeze().detach().cpu().numpy()  # ndarray shape (48*16)
                label = label.squeeze().detach().cpu().numpy()  # ndarray shape (1)

                all_latent_features.append(latent_features)
                truth_labels_arr.append(label.item())

        all_latent_features = np.array(all_latent_features)
        truth_labels_arr = np.array(truth_labels_arr)
        print("Got the latent features for all test images, now doing Kmeans")

        # Do kmeans fit
        estimator = KMeans(n_clusters=10, init='k-means++', n_init=10)
        estimator.fit(all_latent_features)
        predicted_clustering_labels = estimator.labels_

        print("Got the kmeans predicted labels, now computing clustering accuracy")

        # Calculate accuracy
        cost_matrix = np.zeros((10, 10))
        num_samples = np.zeros(10)
        for truth_val in range(10):
            temp_sample_indices = np.where(truth_labels_arr == truth_val)[0]
            num_samples[truth_val] = temp_sample_indices.shape[0]

            temp_predicted_labels = predicted_clustering_labels[temp_sample_indices]

            for predicted_val in range(10):
                temp_matching_pairs = np.where(temp_predicted_labels == predicted_val)[0]
                cost_matrix[truth_val, predicted_val] = 1 - (
                        temp_matching_pairs.shape[0] / temp_sample_indices.shape[0])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        cost = cost_matrix[row_ind, col_ind]

        clustering_acc = ((1 - cost) * num_samples).sum() / num_samples.sum()
        return clustering_acc

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
