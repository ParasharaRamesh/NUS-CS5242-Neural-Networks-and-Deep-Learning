import torch
from torch.utils.data import TensorDataset

'''
TODO.x 

Can perhaps use this as a wrapper to TensorDataset and remove the image from a sub dataset when we pick it and also keep track of it.

And we can use that information to remove picking a class all together if that class is empty

'''
class CustomTensorDataset(TensorDataset):
    def __init__(self, *data_tensors):
        super(CustomTensorDataset, self).__init__(*data_tensors)

    def remove_item(self, index):
        """
        Remove an item from the dataset at the specified index.

        Args:
        index (int): Index of the item to be removed.
        """
        if index >= len(self):
            raise IndexError("Index out of bounds")

        # Create a list of data tensors after excluding the specified index
        updated_data_tensors = [data_tensor for i, data_tensor in enumerate(self.tensors) if i != index]

        # Update the tensors in the dataset
        self.tensors = updated_data_tensors