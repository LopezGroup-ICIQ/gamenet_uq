"""This module contains functions used for the whole workflow of the project, from
data preparation to model training and evaluation."""

from itertools import product
import math
from subprocess import Popen, PIPE
from copy import copy, deepcopy

from sklearn.preprocessing import OneHotEncoder
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
import numpy as np
from scipy.spatial import Voronoi
from ase.io.vasp import read_vasp
from ase import Atoms
from networkx import Graph, set_node_attributes, set_edge_attributes, is_connected, connected_components
from torch_geometric.data import InMemoryDataset
 
from gamenet_uq.constants import CORDERO


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


def structure_to_graph(contcar_file: str,
                       voronoi_tolerance: float,
                       scaling_factor: dict,
                       second_order: bool, 
                       one_hot_encoder: OneHotEncoder, 
                       molecule_elements: list[str]) -> Data:
    """Create Pytorch Geometric graph from VASP chemical structure file (CONTCAR/POSCAR).

    Args:
        contcar_file (str): Path to CONTCAR/POSCAR file.
        voronoi_tolerance (float): Tolerance applied during the graph conversion.
        scaling_factor (float): Scaling factor applied to metal radius of metals.
        second_order (bool): whether 2nd-order metal atoms are included.
        one_hot_encoder (optional): One-hot encoder.

    Returns:
        graph (torch_geometric.data.Data): PyG graph representing the system under study.
    """
    atoms = read_vasp(contcar_file)
    nx_graph = atoms_to_nxgraph(atoms, voronoi_tolerance, scaling_factor, second_order, molecule_elements)
    species_list = [nx_graph.nodes[node]['element'] for node in nx_graph.nodes]
    edge_tails = [edge[0] for edge in nx_graph.edges] + [edge[1] for edge in nx_graph.edges]
    edge_heads = [edge[1] for edge in nx_graph.edges] + [edge[0] for edge in nx_graph.edges]
    elem_array = np.array(species_list).reshape(-1, 1)
    elem_enc = one_hot_encoder.transform(elem_array).toarray()
    edge_index = torch.tensor([edge_tails, edge_heads], dtype=torch.long)
    x = torch.tensor(elem_enc, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)




def get_graph_sample(path: str, 
                     surface_path: str,
                     voronoi_tolerance: float, 
                     scaling_factor: dict, 
                     second_order: bool,
                     encoder: OneHotEncoder,
                     molecule_elements: list[str],
                     gas_mol: bool=False,
                     family: str=None, 
                     surf_multiplier: int=None, 
                     from_poscar: bool=False) -> Data:
    """ 
    Create labelled Pytroch Geometric graph from VASP calculation.
    Args: 
        path (str): path to the VASP directory of the calculation. OUTCAR and CONTCAR/POSCAR files are required.
        surface_path (str): path to the VASP calculation of the empty metal slab. OUTCAR is required.
        voronoi_tolerance (float): tolerance applied during the conversion to graph
        scaling_factor (float): scaling parameter for the atomic radii of metals
        second_order (bool): Inclusion of 2-hop metal neighbours
        encoder (OneHotEncoder): one-hot encoder used to represent atomic elements   
        gas_mol (bool): Whether the system is a gas molecule
        family (str): Family the system belongs to (e.g. "aromatics")
        surf_multiplier (int): Number of times the surface provided is repeated in the supercell (e.g. 2 for 2x2 surface)
        from_poscar (bool): Whether to read the geometry from the POSCAR file (True) or the CONTCAR file (False)
    Returns: 
        pyg_graph (Data): Labelled graph in Pytorch Geometric format
    """
    # Select from which file to read the geometry
    vasp_geometry_file = "POSCAR" if from_poscar else "CONTCAR"
    # Convert the structure to a graph
    pyg_graph = structure_to_graph("{}/{}".format(path, vasp_geometry_file),
                             voronoi_tolerance=voronoi_tolerance, 
                             scaling_factor=scaling_factor,
                             second_order=second_order, 
                             one_hot_encoder=encoder, 
                             molecule_elements=molecule_elements)
    # Label the graph with the energy of the system 
    p1 = Popen(["grep", "energy  w", "{}/OUTCAR".format(path)], stdout=PIPE)
    p2 = Popen(["tail", "-1"], stdin=p1.stdout, stdout=PIPE)
    p3 = Popen(["awk", "{print $NF}"], stdin=p2.stdout, stdout=PIPE)
    pyg_graph.y = float(p3.communicate()[0].decode("utf-8"))
    if gas_mol == False:
        ps1 = Popen(["grep", "energy  w", "{}/OUTCAR".format(surface_path)], stdout=PIPE)
        ps2 = Popen(["tail", "-1"], stdin=ps1.stdout, stdout=PIPE)
        ps3 = Popen(["awk", "{print $NF}"], stdin=ps2.stdout, stdout=PIPE)
        surf_energy = float(ps3.communicate()[0].decode("utf-8"))
        if surf_multiplier is not None:
            surf_energy *= surf_multiplier
        pyg_graph.y -= surf_energy  
    pyg_graph.family = family if family is not None else "None"
    return pyg_graph


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

def weighted_MAE(model, batch, weights):
    """
    Compute weighted MAE for a given model and dataloader.
    Args:
        model (nn.Module): model to evaluate
        loader (DataLoader): dataloader for the dataset to evaluate
        device (torch.device): device to use for the evaluation
    Returns:
        (float): weighted MAE
    """
    model.eval()
    # compute weights for MAE based on the frequency of each batch.family in the dataset
    weights = torch.zeros(len(batch.y))
    for index, item in enumerate(batch.family):
        weights[index] = 1 / batch.family.count(item)    
    return torch.sum(weights * torch.abs(model(batch).squeeze() - batch.y)) / torch.sum(weights)