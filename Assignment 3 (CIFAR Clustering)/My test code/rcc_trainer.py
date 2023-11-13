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
                 dataloader, save_dir, device=config.device,
                 importances={"ae": 0.33, "ucc": 0.33, "rcc": 0.34},
                 use_ssim=False
                 ):
        self.name = name
        self.save_dir = save_dir
        self.device = device

        # importances
        self.ae_importance = importances["ae"]
        self.ucc_importance = importances["ucc"]
        self.rcc_importance = importances["rcc"]

        # dataloaders
        self.train_loader, self.val_loader, self.test_loader = dataloader.get_rcc_dataloaders()
        self.cifar_train_loader, self.cifar_val_loader, self.cifar_test_loader = dataloader.get_cifar_dataloaders()

        # create the directory if it doesn't exist!
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, self.name), exist_ok=True)

        self.rcc_model = rcc_model

        # Adam optimizer(s)
        self.rcc_optimizer = optim.Adam(self.rcc_model.parameters(), lr=config.learning_rate,
                                        weight_decay=config.weight_decay)

        # Loss criterion(s)
        self.ae_loss_criterion = SSIMLoss() if use_ssim else nn.MSELoss()
        self.ucc_loss_criterion = nn.CrossEntropyLoss()
        self.rcc_loss_criterion = nn.MSELoss()

        # Transforms
        self.tensor_to_img_transform = transforms.ToPILImage()

        # Values which can change based on loaded checkpoint
        self.steps = []
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

        # Debug saver lists (i.e. capture these stats for every debug_steps)
        self.debug_ae_losses = []
        self.debug_ucc_losses = []
        self.debug_rcc_losses = []
        self.debug_total_losses = []

    # main train code
    def train(self,
              resume_steps=None,
              load_from_checkpoint=False,
              saver_steps=100):
        torch.cuda.empty_cache()

        # initialize the params from the saved checkpoint
        self.init_params_from_checkpoint_hook(load_from_checkpoint, resume_steps)

        # set up scheduler
        self.init_scheduler_hook()

        # Custom progress bar for each epoch with color
        batch_progress_bar = tqdm(
            total=len(self.train_loader),
            desc=f"Steps",
            position=1,
            leave=False,
            dynamic_ncols=True,
            ncols=100,
            colour='green'
        )

        # set all models to train mode
        self.rcc_model.train()

        # iterate over each batch
        for step, data in enumerate(self.train_loader):
            #zero grad
            self.rcc_optimizer.zero_grad()

            images, ucc_labels, rcc_labels = data

            # forward propogate through the combined model
            rcc_logits, ucc_logits, decoded = self.rcc_model(images)

            # calculate losses from both models for a batch of bags
            ae_loss = self.calculate_autoencoder_loss(images, decoded)
            ucc_loss, batch_ucc_accuracy = self.calculate_ucc_loss_and_acc(ucc_logits, ucc_labels, True)
            rcc_loss, batch_rcc_accuracy = self.calculate_rcc_loss_and_acc(rcc_logits, rcc_labels, True)

            # calculate combined loss
            step_loss = (self.ae_importance * ae_loss) + (self.ucc_importance * ucc_loss) + (self.rcc_importance * rcc_loss)

            # do loss backward for all losses
            step_loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.rcc_model.parameters(), max_norm=config.grad_clip)

            # do optimizer step and zerograd
            self.rcc_optimizer.step()

            # scheduler update (remove if it doesnt work!)
            self.rcc_scheduler.step()

            # add to epoch batch_loss
            self.debug_ae_losses.append(ae_loss.item())
            self.debug_ucc_losses.append(ucc_loss.item())
            self.debug_rcc_losses.append(rcc_loss.item())
            self.debug_total_losses.append(step_loss.item())

            # Update the epoch progress bar (overwrite in place)
            batch_stats = {
                "batch_loss": step_loss.item(),
                "batch_ae_loss": ae_loss.item(),
                "batch_ucc_loss": ucc_loss.item(),
                "batch_rcc_loss": rcc_loss.item(),
                "batch_ucc_acc": batch_ucc_accuracy,
                "batch_rcc_acc": batch_rcc_accuracy
            }

            batch_progress_bar.set_postfix(batch_stats)
            batch_progress_bar.update(1)

            # Compute the average stats for every config.debug_steps steps
            if (step + 1) % config.debug_steps == 0:
                # calculate average epoch train statistics
                avg_train_stats = self.calculate_avg_train_stats_hook()

                # calculate validation statistics
                avg_val_stats = self.validation_hook()

                # Store running history
                self.print_stats_and_store_running_history_hook(step + 1, avg_train_stats, avg_val_stats)

                # Clear the list
                self.debug_ae_losses = []
                self.debug_ucc_losses = []
                self.debug_rcc_losses = []
                self.debug_total_losses = []

            # Save model checkpoint periodically
            if (step + 1) % saver_steps == 0:
                print(f"Going to save model {self.name} @ Step:{step + 1}")
                self.save_model_checkpoint_hook(step + 1)

        # close the epoch progress bar
        batch_progress_bar.close()

        # Return the current state
        return self.get_current_running_history_state_hook()

    # hooks
    def init_params_from_checkpoint_hook(self, load_from_checkpoint, resume_steps):
        if load_from_checkpoint:
            # NOTE: resume_epoch_num can be None here if we want to load from the most recently saved checkpoint!
            checkpoint_path = self.get_model_checkpoint_path(resume_steps)
            checkpoint = torch.load(checkpoint_path)

            # load previous state of models
            self.rcc_model.load_state_dict(checkpoint['rcc_model_state_dict'])

            # load previous state of optimizers
            self.rcc_optimizer.load_state_dict(checkpoint['rcc_optimizer_state_dict'])

            # Things we are keeping track of
            self.steps = checkpoint['steps']

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

    def init_scheduler_hook(self, num_epochs):
        self.rcc_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.rcc_optimizer,
            config.learning_rate,
            total_steps=len(self.train_loader)
        )

    def calculate_autoencoder_loss(self, images, decoded):
        # data is of shape (batchsize=2,bag=10,channels=3,height=32,width=32)
        # generally batch size of 16 is good for cifar10 so predicting 20 won't be so bad
        batch_size, bag_size, num_channels, height, width = images.size()
        batches_of_bag_images = images.view(batch_size * bag_size, num_channels, height, width).to(torch.float32)
        ae_loss = self.ae_loss_criterion(decoded, batches_of_bag_images)  # compares (Batch * Bag, 3,32,32)
        return ae_loss

    def calculate_ucc_loss_and_acc(self, ucc_logits, ucc_labels, is_train_mode=True):
        # compute the batch stats right here and save it
        ucc_probs = nn.Softmax(dim=1)(ucc_logits)
        predicted = torch.argmax(ucc_probs, 1)  # class [8,6]
        labels = ucc_labels  # class [7,6]
        batch_correct_predictions = (predicted == labels).sum().item()  # 0.5
        batch_size = labels.size(0)
        batch_ucc_accuracy = batch_correct_predictions / batch_size

        # compute the ucc_loss between [batch, 4], [Batch,] batch size = 2
        ucc_loss = self.ucc_loss_criterion(ucc_logits, ucc_labels)

        # calculate batchwise accuracy/ucc_loss
        if is_train_mode:
            self.train_correct_predictions += batch_correct_predictions
            self.train_total_batches += batch_size
        else:
            self.eval_correct_predictions += batch_correct_predictions
            self.eval_total_batches += batch_size
        return ucc_loss, batch_ucc_accuracy

    '''
    NOTE: To improve this I can also add a rcc-ucc-enforcement loss where the number of unique classes should match the ucc exactly
    '''
    def calculate_rcc_loss_and_acc(self, rcc_logits, rcc_labels, is_train_mode=True):
        # compute the rcc_loss between [batch, 10] ( as there are 10 classes)
        batch_times_bag_size = config.batch_size * config.bag_size

        # round it to the nearest integer
        predicted = torch.round(rcc_logits).to(torch.float32)

        # NOTE: not sure if it is dim
        batch_correct_predictions = (predicted == rcc_labels).sum().item()

        # compute the rcc_loss
        rcc_loss = self.rcc_loss_criterion(rcc_logits, rcc_labels)

        # calculate batchwise accuracy/ucc_loss
        batch_rcc_accuracy = batch_correct_predictions / batch_times_bag_size
        if is_train_mode:
            self.train_rcc_correct_predictions += batch_correct_predictions
            self.train_rcc_total_batches += batch_times_bag_size
        else:
            self.eval_rcc_correct_predictions += batch_correct_predictions
            self.eval_rcc_total_batches += batch_times_bag_size
        return rcc_loss, batch_rcc_accuracy

    def calculate_avg_train_stats_hook(self):
        avg_training_loss_for_epoch = np.mean(np.array(self.debug_total_losses))
        avg_ae_loss_for_epoch = np.mean(np.array(self.debug_ae_losses))
        avg_ucc_loss_for_epoch = np.mean(np.array(self.debug_ucc_losses))
        avg_rcc_loss_for_epoch = np.mean(np.array(self.debug_rcc_losses))
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

        val_loss = []
        val_ae_loss = []
        val_ucc_loss = []
        val_rcc_loss = []

        with torch.no_grad():
            # set all models to eval mode
            self.rcc_model.eval()

            for val_batch_idx, val_data in enumerate(self.val_loader):
                val_images, val_ucc_labels, val_rcc_labels = val_data

                # forward propogate through the model
                val_rcc_logits, val_ucc_logits, val_decoded = self.rcc_model(val_images)

                # calculate losses from both models for a batch of bags
                val_batch_ae_loss = self.calculate_autoencoder_loss(val_images, val_decoded)
                val_batch_ucc_loss, val_batch_ucc_accuracy = self.calculate_ucc_loss_and_acc(val_ucc_logits,
                                                                                             val_ucc_labels,
                                                                                             False)
                val_batch_rcc_loss, val_batch_rcc_accuracy = self.calculate_rcc_loss_and_acc(val_rcc_logits,
                                                                                             val_rcc_labels,
                                                                                             False)

                # calculate combined loss
                val_batch_loss = (self.ae_importance * val_batch_ae_loss) + (self.ucc_importance * val_batch_ucc_loss) + (self.rcc_importance * val_batch_rcc_loss)

                # cummulate the losses
                val_loss.append(val_batch_loss.item())
                val_ae_loss.append(val_batch_ae_loss.item())
                val_ucc_loss.append(val_batch_ucc_loss.item())
                val_rcc_loss.append(val_batch_rcc_loss.item())

        # Calculate average validation loss for the epoch
        avg_val_loss = np.mean(np.array(val_loss))
        avg_val_ucc_loss = np.mean(np.array(val_ucc_loss))
        avg_val_ae_loss = np.mean(np.array(val_ae_loss))
        avg_val_rcc_loss = np.mean(np.array(val_rcc_loss))
        avg_val_ucc_training_accuracy = self.eval_ucc_correct_predictions / self.eval_ucc_total_batches
        avg_val_rcc_training_accuracy = self.eval_rcc_correct_predictions / self.eval_rcc_total_batches

        stats = {
            "avg_val_loss": avg_val_loss,
            "avg_val_ae_loss": avg_val_ae_loss,
            "avg_val_ucc_loss": avg_val_ucc_loss,
            "avg_val_rcc_loss": avg_val_rcc_loss,
            "avg_val_ucc_training_accuracy": avg_val_ucc_training_accuracy,
            "avg_val_rcc_training_accuracy": avg_val_rcc_training_accuracy
        }

        print(stats)
        print("Finished computing val stats, now showing a sample reconstruction")

        # show some sample predictions
        self.show_sample_reconstructions(self.val_loader)
        return stats

    def print_stats_and_store_running_history_hook(self, curr_step, avg_train_stats, avg_val_stats):
        loss = avg_train_stats["avg_training_loss"]
        ae_loss = avg_train_stats["avg_ae_loss"]
        ucc_loss = avg_train_stats["avg_ucc_loss"]
        rcc_loss = avg_train_stats["avg_rcc_loss"]
        ucc_accuracy = avg_train_stats["avg_ucc_training_accuracy"]
        rcc_accuracy = avg_train_stats["avg_rcc_training_accuracy"]

        val_loss = avg_val_stats["avg_val_loss"]
        val_ae_loss = avg_val_stats["avg_val_ae_loss"]
        val_ucc_loss = avg_val_stats["avg_val_ucc_loss"]
        val_rcc_loss = avg_val_stats["avg_val_rcc_loss"]
        val_ucc_accuracy = avg_val_stats["avg_val_ucc_training_accuracy"]
        val_rcc_accuracy = avg_val_stats["avg_val_rcc_training_accuracy"]

        # store running history
        self.steps.append(curr_step)
        self.training_losses.append(loss)
        self.training_ae_losses.append(ae_loss)
        self.training_ucc_losses.append(ucc_loss)
        self.training_rcc_losses.append(rcc_loss)
        self.training_ucc_accuracies.append(ucc_accuracy)
        self.training_rcc_accuracies.append(rcc_accuracy)

        self.val_losses.append(val_loss)
        self.val_ae_losses.append(val_ae_loss)
        self.val_ucc_losses.append(val_ucc_loss)
        self.val_rcc_losses.append(val_rcc_loss)
        self.val_ucc_accuracies.append(val_ucc_accuracy)
        self.val_rcc_accuracies.append(val_rcc_accuracy)

        # print stats
        print(
            f"[TRAIN]:Step: {curr_step} | Loss: {loss} | AE Loss: {ae_loss} | UCC Loss: {ucc_loss} | UCC Acc: {ucc_accuracy} | RCC Loss: {rcc_loss} | RCC Acc: {rcc_accuracy}"
        )
        print(
            f"[VAL]:Step: {curr_step} | Val Loss: {val_loss} | Val AE Loss: {val_ae_loss} | Val UCC Loss: {val_ucc_loss} | Val UCC Acc: {val_ucc_accuracy} | Val RCC Loss: {val_rcc_loss} | Val RCC Acc: {val_rcc_accuracy}"
        )
        print()
        print("-" * 60)

    def get_current_running_history_state_hook(self):
        return self.steps, \
            self.training_ae_losses, self.training_ucc_losses, self.training_rcc_losses, self.training_losses, self.training_ucc_accuracies, self.training_rcc_accuracies, \
            self.val_ae_losses, self.val_ucc_losses, self.val_rcc_losses, self.val_losses, self.val_ucc_accuracies, self.val_rcc_accuracies

    def save_model_checkpoint_hook(self, step):
        # set it to train mode to save the weights (but doesn't matter apparently!)
        self.rcc_model.train()

        # create the directory if it doesn't exist
        model_save_directory = os.path.join(self.save_dir, self.name)
        os.makedirs(model_save_directory, exist_ok=True)

        # Checkpoint the model at the end of each epoch
        checkpoint_path = os.path.join(model_save_directory, f'model_epoch_{step + 1}.pt')
        torch.save(
            {
                'rcc_model_state_dict': self.rcc_model.state_dict(),
                'rcc_optimizer_state_dict': self.rcc_optimizer.state_dict(),
                'steps': self.steps,
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
        print(f"Saved the model checkpoint for experiment {self.name} for epoch {step + 1}")

    def test_model(self):
        # class level init
        self.eval_ucc_correct_predictions = 0
        self.eval_ucc_total_batches = 0

        self.eval_rcc_correct_predictions = 0
        self.eval_rcc_total_batches = 0

        test_loss = []
        test_ae_loss = []
        test_ucc_loss = []
        test_rcc_loss = []

        with torch.no_grad():
            # set all models to eval mode
            self.rcc_model.eval()

            for test_batch_idx, test_data in enumerate(self.test_loader):
                test_images, test_ucc_labels, test_rcc_labels = test_data

                # forward propogate through the model
                test_rcc_logits, test_ucc_logits, test_decoded = self.rcc_model(test_images)

                # calculate losses from both models for a batch of bags
                test_batch_ae_loss = self.calculate_autoencoder_loss(test_images, test_decoded)
                test_batch_ucc_loss, test_batch_ucc_accuracy = self.calculate_ucc_loss_and_acc(test_ucc_logits,
                                                                                               test_ucc_labels,
                                                                                               False)
                test_batch_rcc_loss, test_batch_rcc_accuracy = self.calculate_rcc_loss_and_acc(test_rcc_logits,
                                                                                               test_rcc_labels,
                                                                                               False)

                # calculate combined loss
                test_batch_loss = (self.ae_importance * test_batch_ae_loss) + (self.ucc_importance * test_batch_ucc_loss) + (self.rcc_importance * test_batch_rcc_loss)

                # cummulate the losses
                test_loss.append(test_batch_loss.item())
                test_ae_loss.append(test_batch_ae_loss.item())
                test_ucc_loss.append(test_batch_ucc_loss.item())
                test_rcc_loss.append(test_batch_rcc_loss.item())

        # Calculate average validation loss for the epoch
        avg_test_loss = np.mean(np.array(test_loss))
        avg_test_ucc_loss = np.mean(np.array(test_ucc_loss))
        avg_test_rcc_loss = np.mean(np.array(test_rcc_loss))
        avg_test_ae_loss = np.mean(np.array(test_ae_loss))
        avg_test_ucc_training_accuracy = self.eval_ucc_correct_predictions / self.eval_ucc_total_batches
        avg_test_rcc_training_accuracy = self.eval_rcc_correct_predictions / self.eval_rcc_total_batches

        stats = {
            "avg_test_loss": avg_test_loss,
            "avg_test_ae_loss": avg_test_ae_loss,
            "avg_test_ucc_loss": avg_test_ucc_loss,
            "avg_test_rcc_loss": avg_test_rcc_loss,
            "avg_test_ucc_training_accuracy": avg_test_ucc_training_accuracy,
            "avg_test_rcc_training_accuracy": avg_test_rcc_training_accuracy
        }
        print(stats)

        print("Now going to show a sample reconstruction")
        # show some sample predictions
        self.show_sample_reconstructions(self.test_loader)
        return stats

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

    # find the most recent file and return the path
    def get_model_checkpoint_path(self, step_num=None):
        directory = os.path.join(self.save_dir, self.name)
        if step_num == None:
            # Get a list of all files in the directory
            files = os.listdir(directory)

            # Filter out only the files (exclude directories)
            files = [f for f in files if os.path.isfile(os.path.join(directory, f))]

            # Sort the files by their modification time in descending order (most recent first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

            # Get the name of the most recently added file
            model_file = files[0] if files else None
        else:
            model_file = f"model_step_{step_num}.pt"
        return os.path.join(directory, model_file)

    # Calculate min JS Divergence
    def calculate_js_divergence(self, p, q):
        m = 0.5 * (p + q)
        log_p_over_m = np.log2(p / m)
        log_q_over_m = np.log2(q / m)
        return 0.5 * np.sum(p * log_p_over_m) + 0.5 * np.sum(q * log_q_over_m)

    def get_all_kde_distributions(self):
        self.rcc_model.eval()

        kde_distributions = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in tqdm(self.cifar_test_loader):
                imgs = imgs.unsqueeze(1).to(config.device)
                kde_dist = self.rcc_model.get_kde_distributions(imgs)
                kde_distributions.append(kde_dist.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        kde_distributions = np.concatenate(kde_distributions)
        all_labels = np.concatenate(all_labels)
        return kde_distributions, all_labels

    def compute_min_js_divergence(self):
        kde_distributions, all_labels = self.get_all_kde_distributions()
        print("Got the KDE distributions for the test dataset")

        # iterate for each class and get only those embeddings
        distribution_of_all_label_classes = []
        for i in range(config.num_classes):
            idxs = np.where(all_labels == i)
            kde_distribution_i = kde_distributions[idxs]
            kde_distribution_i = np.mean(kde_distribution_i, axis=0)
            distribution_of_all_label_classes.append(kde_distribution_i)
        distribution_of_all_label_classes = np.array(distribution_of_all_label_classes)
        print("Got the average kde distribution per label class, now computing min js divergence")

        res = np.zeros((config.num_classes, config.num_classes))
        for i in range(config.num_classes):
            p = np.clip(distribution_of_all_label_classes[i, :], 1e-12, 1)
            # p = distribution_of_all_label_classes[i, :]
            for j in range(i, config.num_classes):
                q = np.clip(distribution_of_all_label_classes[j, :], 1e-12, 1)
                # q = distribution_of_all_label_classes[j, :]

                # fill the upper triangle
                res[i, j] = self.calculate_js_divergence(p, q)

                # fill the lower triangle
                res[j, i] = res[i, j]

        # we are not interested in the identity relation anyway
        np.fill_diagonal(res, np.inf)
        print("Computed all interclass js divergence scores, the entire interclass js divergence is ")
        print(res)
        # Find the minimum value
        min_js_divergence = np.min(res)

        # Find the indices of the minimum value
        min_indices = np.argmin(res)
        min_row, min_col = np.unravel_index(min_indices, res.shape)

        print(f"Min JS Divergence is {min_js_divergence}, between classes {min_row} & {min_col}")
        return min_js_divergence


    # Calculate clustering accuracy
    def get_all_encoder_features(self):
        self.rcc_model.eval()

        all_features = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in tqdm(self.cifar_test_loader):
                imgs = imgs.unsqueeze(1).to(config.device)
                features = self.rcc_model.get_encoder_features(imgs)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_features = np.concatenate(all_features)
        all_labels = np.concatenate(all_labels)
        return all_features, all_labels

    def perform_clustering_and_get_cluster_labels(self, features):
        model = KMeans(n_clusters=10, init='k-means++', n_init=10)
        model.fit(features)
        return model.labels_

    def compute_clustering_accuracy(self):
        all_features, all_labels = self.get_all_encoder_features()
        print("Computed all the features from the encoder")
        cluster_labels = self.perform_clustering_and_get_cluster_labels(all_features)
        print("Performed clustering and computed all the cluster labels")

        cost_matrix = np.zeros((config.num_classes, config.num_classes))
        num_samples = np.zeros(config.num_classes)

        for true_label in range(config.num_classes):
            true_label_idxs = np.where(all_labels == true_label)[0]
            num_samples[true_label] = true_label_idxs.shape[0]

            sample_preds = cluster_labels[true_label_idxs]

            for pred_label in range(config.num_classes):
                pairs = np.where(sample_preds == pred_label)[0]

                cost_matrix[true_label, pred_label] = 1 - (pairs.shape[0] / true_label_idxs.shape[0])

        print("Going to perform linear sum assignment of cost matrix")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        cost = cost_matrix[row_ind, col_ind]
        clustering_acc = ((1 - cost) * num_samples).sum() / num_samples.sum()
        print(f"Clustering accuracy is {clustering_acc}")
        return clustering_acc
