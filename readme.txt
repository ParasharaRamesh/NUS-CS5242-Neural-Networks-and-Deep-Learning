In this folder, we have provided the code for our paper "Weakly Supervised Clustering by Exploiting Unique Class Count", which is under review as a conference paper at ICLR 2020.

Folder structure:
data (the data used in experiments)
|-----camelyon
|-----cifar10
|-----cifar100_coarse
|-----mnist
|-----subsets
|
ucc (code for our $UCC$ model on MNIST and $UCC_{segment}$ model for semantic segmentation)
|----- camelyon
|      |------ extracted_features (folder that stores extracted features for each instance/patch)
|      |------ loss_data (folder that stores loss and accuracy metrics collected during training)
|      |------ patch_ground_truths (folder that stores patch ground truths)
|      |------ predicted_labels (folder that stores predicted labels for patches)
|      |------ predicted_masks (folder that stores predicted masks for images in the dataset)
|      |------ predicted_masks_post_processed (folder that stores post-processed predicted masks for images in the dataset)
|      |------ predictions (folder that stores ucc predictions)
|      |------ saved_models (folder that stores saved model weights during training)
|      |------ bag_level_confusion_matrix.py (script to obtain confusion matrix fro ucc predictions)
|      |------ calculate_statistics.py (script to calculate final statistics)
|      |------ cluster_patches.py (script to cluster patches)
|      |------ create_predicted_masks.py (script to create predicted masks for images)
|      |------ dataset.py (script to organize dataset during training)
|      |------ evaluate_model.sh (script to call other scripts in sequence to obtain results in the paper)
|      |------ extract_patch_features.py (script to extract patch features)
|      |------ image_level_statistics.py (script to obtain image level statistics)
|      |------ model.py (script to construct our neural network models)
|      |------ obtain_patch_truths.py (script to obtain the patch ground truths)
|      |------ post_processing.py (script to post process predicted masks)
|      |------ test.py (script to test a trained model)
|      |------ train.py (script to train a new model)
|
|----- mnist
       |------ clustering (folder that stores clustering results)
       |------ distributions (folder that stores obtained distributions)
       |------ evaluate_model.sh (script to call other scripts in sequence to obtain results in the paper)
       |------ extracted_features (folder that stores extracted features for each instance)
       |------ generated_digits (folder that stores generated digits)
       |------ loss_data (folder that stores loss and accuracy metrics collected during training)
       |------ predictions (folder that stores ucc predictions)
       |------ saved_models (folder that stores saved model weights during training)
       |------ calculate_clustering_accuracy.py (script to calculate clustering accuracy)
       |------ calculate_js_divergence.py (script to calculate inter-class JS divergence values)
       |------ cluster.py (script to cluster instances in the dataset)
       |------ dataset.py (script to organize dataset during training)
       |------ dataset_test.py (script to organize dataset during testing)
       |------ extract_features.py (script to extract features of instances)
       |------ generate_digits.py (script to generate digits by using autoencoder branch with mean feature values for each class)
       |------ model.py (script to construct our neural network models)
       |------ obtain_clustering_labels.py (script to obtain clustering labels for each patch)
       |------ obtain_distributions.py (script to obtain extracted feature distributions)
       |------ test.py (script to test a trained model)
       |------ train.py (script to train a new model)
       |------ visualize_distributions.py (script to visualize obtained distributions)


##### Reproducing the results in the paper #####
- $UCC_{segment}$ model on semantic segmentation dataset: run "ucc/camelyon/evaluate_model.sh"
- $UCC$ model on MNIST dataset: run "ucc/mnist/evaluate_model.sh"

##### Training a new model #####
- $UCC_{segment}$ model on semantic segmentation dataset: run "ucc/camelyon/train.py"
- $UCC$ model on MNIST dataset: run "ucc/mnist/train.py"


