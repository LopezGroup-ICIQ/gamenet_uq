"""This module contains functions used for the whole workflow of the project, from
data preparation to model training and evaluation."""

import math
from copy import copy, deepcopy

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch
from torch_geometric.data import InMemoryDataset


def split_percentage(splits: int, test: bool=True) -> tuple[int]:
    """Return split percentage of the train, validation and test sets.
    One split represent the test set, one the validation set and the rest the train set.
    Args:
        split (int): number of initial splits of the entire initial dataset

    Returns:
        a, b, c: train, validation, test percentage of the sets.
    Examples:
        >>> split_percentage(5) # 5 splits
        (60, 20, 20)  # 60% of data for training, 20% for validation and 20% for testing
        >>> split_percentage(5, test=False) # 5 splits
        (80, 20, 0)  # 80% of data for training, 20% for validation and 0% for testing
    """
    if test:
        a = int(100 - 200 / splits)
        b = math.ceil(100 / splits)
        return a, b, b
    else:
        return int((1 - 1/splits) * 100), math.ceil(100 / splits)
    

def create_loaders(dataset: InMemoryDataset,
                   split: int=5,
                   batch_size: int=32,
                   test: bool=True, 
                   balance_func: callable=None) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset (tuple): tuple containing the HetGraphDataset susbsets.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    train_loader, val_loader, test_loader = [], [], []
    n_items = len(dataset)
    sep = n_items // split
    dataset = dataset.shuffle()
    if test:
        test_loader += (dataset[:sep])
        val_loader += (dataset[sep:sep*2])
        train_loader += (dataset[sep*2:])
    else:
        val_loader += (dataset[:sep])
        train_loader += (dataset[sep:])
    if balance_func != None:
        train_loader = balance_func(train_loader)
    train_n = len(train_loader)
    val_n = len(val_loader)
    test_n = len(test_loader)
    total_n = train_n + val_n + test_n
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)
    if test:
        test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)
        a, b, c = split_percentage(split)
        print("Data split (train/val/test): {}/{}/{} %".format(a, b, c))
        print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
        return (train_loader, val_loader, test_loader)
    else:
        print("Data split (train/val): {}/{} %".format(int(100*(split-1)/split), int(100/split)))
        print("Training data = {} Validation data = {} (Total = {})".format(train_n, val_n, total_n))
        return (train_loader, val_loader, None)


def scale_target(train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader=None,
                 mode: str='std',
                 verbose: bool=True,
                 test: bool=True):
    """
    Apply target scaling to the whole dataset using training and validation sets.
    Args:
        train_loader (torch_geometric.loader.DataLoader): training dataloader 
        val_loader (torch_geometric.loader.DataLoader): validation dataloader
        test_loader (torch_geometric.loader.DataLoader): test dataloader
    Returns:
        train, val, test: dataloaders with scaled target values
        mean_tv, std_tv: mean and std (standardization)
        min_tv, max_tv: min and max (normalization)
    """
    # 1) Get target scaling coefficients from train and validation sets
    y_list = []
    for graph in train_loader.dataset:
        y_list.append(graph.target.item())
    for graph in val_loader.dataset:
        y_list.append(graph.target.item())
    y_tensor = torch.tensor(y_list)
    # Standardization
    mean_tv = y_tensor.mean(dim=0, keepdim=True)  
    std_tv = y_tensor.std(dim=0, keepdim=True)
    # Normalization
    max_tv = y_tensor.max()
    min_tv = y_tensor.min()
    delta_norm = max_tv - min_tv
    # 2) Apply Scaling
    for graph in train_loader.dataset:
        if mode == "std":
            graph.y = (graph.target - mean_tv) / std_tv
        elif mode == "norm":
            graph.y = (graph.target - min_tv) / (max_tv - min_tv)
        else:
            pass
    for graph in val_loader.dataset:
        if mode == "std":
            graph.y = (graph.target - mean_tv) / std_tv
        elif mode == "norm":
            graph.y = (graph.target - min_tv) / delta_norm
        else:
            pass
    if test:
        for graph in test_loader.dataset:
            if mode == "std":
                graph.y = (graph.target - mean_tv) / std_tv
            elif mode == "norm":
                graph.y = (graph.target - min_tv) / delta_norm
            else:
                pass
    if mode == "std":
        if verbose:
            print("Target Scaling (Standardization) applied successfully")
            print("(Train+Val) mean: {:.2f} eV".format(mean_tv.item()))
            print("(Train+Val) standard deviation: {:.2f} eV".format(std_tv.item()))
        if test:
            return train_loader, val_loader, test_loader, mean_tv.item(), std_tv.item()
        else:
            return train_loader, val_loader, None, mean_tv.item(), std_tv.item()
    elif mode == "norm": 
        if verbose:
            print("Target Scaling (Normalization) applied successfully")
            print("(Train+Val) min: {:.2f} eV".format(min_tv.item()))
            print("(Train+Val) max: {:.2f} eV".format(max_tv.item()))
        if test:
            return train_loader, val_loader, test_loader, min_tv.item(), max_tv.item()
        else:
            return train_loader, val_loader, None, min_tv.item(), max_tv.item()
    else:
        print("Target Scaling not applied")
        return train_loader, val_loader, test_loader, 0, 1


def train_loop(model,
               device:str,
               train_loader: DataLoader,
               optimizer,
               loss_fn):
    """
    Run training iteration (epoch) 
    For each batch in the epoch, the following actions are performed:
    1) Move the batch to the training device
    2) Forward pass through the GNN model and compute loss
    3) Compute gradient of loss function wrt model parameters
    4) Update model parameters
    Args:
        model(): GNN model object.
        device(str): device on which training is performed.
        train_loader(): Training dataloader.
        optimizer(): optimizer used during training.
        loss_fn(): Loss function used for the training.
    Returns:
        loss_all, mae_all (tuple[float]): Loss function and MAE of the whole epoch.   
    """
    model.train()  
    loss_all, mae_all = 0, 0
    for batch in train_loader:  # batch-wise
        batch = batch.to(device)
        optimizer.zero_grad()                     # Set gradients of all tensors to zero
        loss = loss_fn(model, batch)
        mae = F.l1_loss(model(batch).mean.squeeze(), batch.y)    # For comparison with val/test data
        loss.backward()                           # Get gradient of loss function wrt parameters
        loss_all += loss.item() * batch.num_graphs
        mae_all += mae.item() * batch.num_graphs
        optimizer.step()                          # Update model parameters
    loss_all /= len(train_loader.dataset)
    mae_all /= len(train_loader.dataset)
    return loss_all, mae_all


def test_loop(model,
              loader: DataLoader,
              device: str,
              std: float,
              mean: float=None, 
              scaled_graph_label: bool= True) -> float:
    """
    Run test or validation iteration (epoch).
    For each batch in the validation/test set, the following steps are performed:
    1) Set the GNN model in evaluation mode
    2) Move the batch to the training device (CPU or GPU)
    3) Compute the Mean Absolute Error (MAE)
    Args:
        model (): GNN model object.
        loader (Dataloader object): Dataset for validation/testing.
        device (str): device on which training is performed.
        std (float): standard deviation of the training+validation datasets [eV]
        mean (float): mean of the training+validation datasets [eV]
        scaled_graph_label (bool): whether the graph labels are in eV or in a scaled format.
        verbose (int): 0=no printing info 1=printing information
    Returns:
        error(float): Mean Absolute Error (MAE) of the test loader.
    """
    model.eval()   
    error = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            error += (model(batch).mean * std - batch.y * std).abs().sum().item()   
    return error / len(loader.dataset) 


def get_mean_std_from_model(path:str) -> tuple[float]:
    """Get mean and standard deviation used for scaling the target values 
       from the selected trained model.

    Args:
        model_name (str): GNN model path.
    
    Returns:
        mean, std (tuple[float]): mean and standard deviation for scaling the targets.
    """
    file = open("{}/performance.txt".format(path), "r")
    lines = file.readlines()
    for line in lines:
        if "(train+val) mean" in line:
            mean = float(line.split()[-2])
        if "(train+val) standard deviation" in line:
            std = float(line.split()[-2])
    return mean, std


def get_graph_conversion_params(path: str) -> tuple:
    """Get the hyperparameters for geometry->graph conversion algorithm.
    Args:
        path (str): path to directory containing the GNN model.
    Returns:
        tuple: voronoi tolerance (float), scaling factor (float), metal nearest neighbours inclusion (bool)
    """
    file = open("{}/performance.txt".format(path), "r")
    lines = file.readlines()
    for line in lines:
        if "Voronoi" in line:
            voronoi_tol = float(line.split()[-2])
        if "scaling factor" in line:
            scaling_factor = float(line.split()[-1])
        if "Second order" in line:
            if line.split()[-1] == "True":
                second_order_nn = True
            else:
                second_order_nn = False
    return voronoi_tol, scaling_factor, second_order_nn


def split_list(a: list, n: int):
    """
    Split a list into n chunks (for nested cross-validation)
    Args:
        a(list): list to split
        n(int): number of chunks
    Returns:
        (list): list of chunks
    """
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def create_loaders_nested_cv(dataset: InMemoryDataset, 
                             split: int, 
                             batch_size: int):
    """
    Create dataloaders for training, validation and test sets for nested cross-validation.
    Args:
        datasets(tuple): tuple containing the HetGraphDataset objects.
        split(int): number of splits to generate train/val/test sets
        batch(int): batch size    
    Returns:
        (tuple): tuple with dataloaders for training, validation and testing.
    """
    # Create list of lists, where each list contains the datasets for a split.
    chunk = [[] for _ in range(split)]
    
    dataset.shuffle()
    iterator = split_list(dataset, split)
    for index, item in enumerate(iterator):
        chunk[index] += item
    chunk = sorted(chunk, key=len)
    # Create dataloaders for each split.    
    for index in range(len(chunk)):
        proxy = copy(chunk)
        test_loader = DataLoader(proxy.pop(index), batch_size=batch_size, shuffle=False)
        for index2 in range(len(proxy)):  # length is reduced by 1 here
            proxy2 = copy(proxy)
            val_loader = DataLoader(proxy2.pop(index2), batch_size=batch_size, shuffle=False)
            flatten_training = [item for sublist in proxy2 for item in sublist]  # flatten list of lists
            train_loader = DataLoader(flatten_training, batch_size=batch_size, shuffle=True)
            yield deepcopy((train_loader, val_loader, test_loader))
