import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import numpy as np
import sys
sys.path.insert(1, '3_model/src')
from models import Model


class SequenceDataset(Dataset):
    """
    Custom Dataset class for sequences that include dynamic and static inputs,
    and multiple outputs

    __len__ returns the number of sequences
    
    __getitem__ returns dynamic input features, static input features, and
    temperature observations at all depths for one sequence

    Example:

    >>> npz = np.load("2_process/out/mntoha/train.npz")
    >>> n_dynamic = len(npz['dynamic_features_all'])
    >>> n_static = len(npz['static_features_all'])
    >>> n_depths = len(npz['depths'])
    >>> data = torch.from_numpy(npz['data'])
    >>> dataset = SequenceDataset(data, n_depths, n_dynamic, n_static)                            
    >>> # number of sequences
    >>> len(dataset)
    19069
    >>> # dynamic features, static features, and temperatures at depths 
    >>> # for first sequence
    >>> for e in dataset[0]:
    >>>     print(e.shape)
    >>> 
    torch.Size([400, 9])
    torch.Size([400, 4])
    torch.Size([400, 49])

    """
    def __init__(self, sequences, n_outputs, n_dynamic, n_static):
        """
        :param sequences: An array of sequences with shape (# sequences, sequence_length, # depths + # input features)
        :param n_outputs: Number of outputs (# depths)
        :param n_dynamic: Number of dynamic input features
        :param n_static: Number of static input features

        """
        # The number of sequences is the size of the first dimension
        self.len = sequences.shape[0]
        self.n_outputs = n_outputs
        self.n_dynamic = n_dynamic
        self.n_static = n_static
        self.sequences = sequences

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # sequence is organized as temperatures at multiple depths, then dynamic features, then static features
        temperatures = self.sequences[idx, :, :self.n_outputs]
        dynamic_features = self.sequences[idx, :, self.n_outputs:self.n_outputs + self.n_dynamic]
        static_features = self.sequences[idx, :, -self.n_static:]
        return dynamic_features, static_features, temperatures


def get_dataloader(npz_filepath,
                   dynamic_features_use,
                   static_features_use,
                   depths_use):
    """
    Get dataloader from npz file

    :param npz_filepath: Name and path to .npz data file
    :param dynamic_features_use: List of dynamic features to include
    :param static_features_use: List of static features to include
    :param depths_use: List of depth values to include

    """

    # Training data
    data_npz = np.load(npz_filepath)

    # Determine which features to use

    # depths to strings
    depths_all_str = [f"depth_{depth}" for depth in data_npz['depths_all']]
    depths_use_str = [f"depth_{depth}" for depth in depths_use]

    # The first `len(depths)` elements in the third dimension are temperature observations.
    # Input features are drivers, clarity, ice flags, and static attributes, in that order.
    features_all = (depths_all_str 
                    + data_npz['dynamic_features_all'].tolist()
                    + data_npz['static_features_all'].tolist())

    depths_use_idx = [features_all.index(feature) for feature in depths_use_str]
    dynamic_features_use_idx = [features_all.index(feature) for feature in dynamic_features_use]
    static_features_use_idx = [features_all.index(feature) for feature in static_features_use]
    idx_to_use = depths_use_idx + dynamic_features_use_idx + static_features_use_idx

    n_dynamic = len(dynamic_features_use)
    n_static = len(static_features_use)
    n_depths = len(depths_use)

    # shape of data is (# sequences, sequence_length, # depths + # input features)
    dataset = SequenceDataset(
        torch.from_numpy(data_npz['data'][:,:,idx_to_use]),
        n_depths,
        n_dynamic,
        n_static
    )                            
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
