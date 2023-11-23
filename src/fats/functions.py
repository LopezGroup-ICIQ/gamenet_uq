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
 
from fats.constants import CORDERO


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
    

def get_voronoi_neighbourlist(atoms: Atoms, 
                              tolerance: float, 
                              scaling_factor: float, 
                              molecule_elements: list[str]) -> np.ndarray:
    """
    Get connectivity list from Voronoi analysis, considering periodic boundary conditions.
    Assumption: The catalyst surface does not contain elements present in the adsorbate.
    
    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-metal system.
        tolerance (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the metal atoms.
        
    Returns:
        np.ndarray: connectivity list of the system. Each row represents a pair of connected atoms.
    """
    
    # First condition for two atoms to be connected: They must share a Voronoi facet
    coords_arr = np.repeat(np.expand_dims(np.copy(atoms.get_scaled_positions()), axis=0), 27, axis=0)
    mirrors = np.repeat(np.expand_dims(np.asarray(list(product([-1, 0, 1], repeat=3))), 1), coords_arr.shape[1], axis=1)
    corrected_coords = np.reshape(coords_arr + mirrors, (coords_arr.shape[0] * coords_arr.shape[1], coords_arr.shape[2]))
    corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    pairs_corr = np.delete(pairs_corr, np.argwhere(pairs_corr[:, 0] == pairs_corr[:, 1]), axis=0)

    # Second condition for two atoms to be connected: Their distance must be smaller than the sum of their covalent radii plus a tolerance.
    # NB surface atoms covalent radii are scaled by a factor corresponding to scaling_factor.
    pairs_lst = []
    for pair in pairs_corr:
        distance = atoms.get_distance(pair[0], pair[1], mic=True)
        threshold = CORDERO[atoms[pair[0]].symbol] + CORDERO[atoms[pair[1]].symbol] + tolerance + \
                    (scaling_factor - 1.0) * ((atoms[pair[0]].symbol not in molecule_elements) * CORDERO[atoms[pair[0]].symbol] + \
                                              (atoms[pair[1]].symbol not in molecule_elements) * CORDERO[atoms[pair[1]].symbol])
        if distance <= threshold:
            pairs_lst.append(pair)

    return np.sort(np.array(pairs_lst), axis=1)

def detect_broken_bond(atoms: Atoms,
                       adsorbate_elements: list[str]) -> tuple[int]:
    """
    Detect bond broken in the TS and identify atoms involved in the bond.
    Given a TS geometry of an adsorbate on a metal surface, this function detects
    the bond broken in the TS and returns the indices of the two atoms involved in the bond.
    """
    adsorbate_indices = {atom.index for atom in atoms if atom.symbol in adsorbate_elements}
    neighbour_list = get_voronoi_neighbourlist(atoms, 0.25, 1.0, adsorbate_elements)
    neighbour_list = neighbour_list[np.isin(neighbour_list[:, 0], list(adsorbate_indices)) & np.isin(neighbour_list[:, 1], list(adsorbate_indices))]
    nx_graph = Graph()
    nx_graph.add_nodes_from(adsorbate_indices)
    nx_graph.add_edges_from(neighbour_list)
    components = list(connected_components(nx_graph))
    if len(components) != 2:
        return None
    else:
        dist_dict = {}
        for node1 in components[0]:
            for node2 in components[1]:
                dist_dict[(node1, node2)] = atoms.get_distance(node1, node2, mic=True)
        return min(dist_dict, key=dist_dict.get)


def atoms_to_nxgraph(atoms: Atoms, 
                     voronoi_tolerance: float, 
                     scaling_factor: float, 
                     adsorbate_elements: list[str]) -> Graph:
    """
    Convert ASE Atoms object to NetworkX graph, representing the adsorbate-surface system.

    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-metal system.
        voronoi_tolerance (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the surface atoms.
        adsorbate_elements (list[str]): list of elements present in the adsorbate.
    Returns:
        Graph: NetworkX graph representing the adsorbate-metal system.
    """
    # 1) Detect broken bond in the TS
    adsorbate_indices = [atom.index for atom in atoms if atom.symbol in adsorbate_elements]
    neighbour_list = get_voronoi_neighbourlist(atoms, voronoi_tolerance, scaling_factor, adsorbate_elements)
    broken_bond_indices = detect_broken_bond(atoms, adsorbate_elements)
    broken_bond_indices_reindexed = [adsorbate_indices.index(broken_bond_indices[0]), adsorbate_indices.index(broken_bond_indices[1])]
    print("Broken bond: {}".format(broken_bond_indices_reindexed))        

    # 2) Get surface atoms that are neighbours of the adsorbate
    surface_neighbours_indices = {
        pair[1] if pair[0] in adsorbate_indices else pair[0] 
        for pair in neighbour_list 
        if (pair[0] in adsorbate_indices and atoms[pair[1]].symbol not in adsorbate_elements) or 
           (pair[1] in adsorbate_indices and atoms[pair[0]].symbol not in adsorbate_elements)
    }
        
    # 3) Construct graph with the atoms in the ensemble
    ensemble = Atoms(atoms[list(adsorbate_indices) + list(surface_neighbours_indices)], pbc=atoms.pbc, cell=atoms.cell)
    nx_graph = Graph()
    nx_graph.add_nodes_from(range(len(ensemble)))
    set_node_attributes(nx_graph, {i: ensemble[i].symbol for i in range(len(ensemble))}, "element")
    ensemble_neighbour_list = get_voronoi_neighbourlist(ensemble, voronoi_tolerance, scaling_factor, adsorbate_elements)
    nx_graph.add_edges_from(ensemble_neighbour_list)
    set_edge_attributes(nx_graph, {(pair[0], pair[1]): 0 for pair in ensemble_neighbour_list}, "ts_edge")
    # Add TS edge
    if broken_bond_indices_reindexed != None:
        nx_graph.add_edge(broken_bond_indices_reindexed[0], broken_bond_indices_reindexed[1])
        nx_graph.edges[broken_bond_indices_reindexed[0], broken_bond_indices_reindexed[1]]['ts_edge'] = 1
    # print(nx_graph.edges(data=True))
    return nx_graph, list(surface_neighbours_indices), broken_bond_indices


def create_loaders(dataset: InMemoryDataset,
                   split: int,
                   batch_size: int,
                   test: bool=True, 
                   balance_func: callable=None) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        datasets (tuple): tuple containing the HetGraphDataset susbsets.
        split (int): number of splits to generate train/val/test sets.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate train/val/test loaders or just train/val.    
    Returns:
        (tuple): tuple with dataloaders for training, validation and test.
    """
    train_loader, val_loader, test_loader = [], [], []
    n_items = len(dataset)
    sep = n_items // split
    #dataset = dataset.shuffle()
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


def scale_feature(train_loader: DataLoader,
                 val_loader: DataLoader,
                 features: list[str],
                 test_loader: DataLoader=None,
                 test: bool=True):
    """
    Apply feature scaling to the whole dataset using training and validation sets.
    Args:
        train_loader (torch_geometric.loader.DataLoader): training dataloader 
        val_loader (torch_geometric.loader.DataLoader): validation dataloader
        test_loader (torch_geometric.loader.DataLoader): test dataloader
    Returns:
        train, val, test: dataloaders with scaled target values
        mean_tv, std_tv: mean and std (standardization)
        min_tv, max_tv: min and max (normalization)
    """
    pass


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
        loss = loss_fn(model(batch), batch.y)
        mae = F.l1_loss(model(batch), batch.y)    # For comparison with val/test data
        loss.backward()                           # Get gradient of loss function wrt parameters
        loss_all += loss.item() * batch.num_graphs
        mae_all += mae.item() * batch.num_graphs
        optimizer.step()                          # Update model parameters
    loss_all /= len(train_loader.dataset)
    mae_all /= len(train_loader.dataset)
    return loss_all, mae_all


def train_loop_uq(model,
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
        for batch in loader:  # batch-wise
            batch = batch.to(device)
            if scaled_graph_label == False:  # label in eV
                error += (model(batch) * std + mean - batch.y).abs().sum().item()
            else:  #  Scaled label (unitless)
                error += (model(batch) * std - batch.y * std).abs().sum().item()   
    return error / len(loader.dataset) 


def test_loop_uq(model,
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
        for batch in loader:  # batch-wise
            batch = batch.to(device)
            if scaled_graph_label == False:  # label in eV
                error += (model(batch).mean * std + mean - batch.y).abs().sum().item()
            else:  #  Scaled label (unitless)
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

def atoms_to_pyggraph(atoms: Atoms,
                      voronoi_tolerance: float,
                      scaling_factor: dict,
                      one_hot_encoder: OneHotEncoder, 
                      adsorbate_elements: list[str]) -> Data:
    """
    Create Pytorch Geometric graph from ASE Atoms object representing TS system.
    Skeletal graph is generated with this function, i.e. no features are added to the nodes, except
    for the one-hot encoding of the atomic species.

    Args:
        atoms (Atoms): ASE Atoms object.
        voronoi_tolerance (float): Tolerance applied during the graph conversion.
        scaling_factor (float): Scaling factor applied to metal radius of metals.
        include_surf_atoms_2hops (bool): whether 2nd-order surface atoms are included.
        one_hot_encoder (optional): One-hot encoder.
        adsorbate_elements (list[str]): list of elements present in the adsorbate.
    Returns:
        graph (torch_geometric.data.Data): PyG graph representing the system under study.
    """
    nx_graph, surface_neighbors, bb_idxs = atoms_to_nxgraph(atoms, voronoi_tolerance, scaling_factor, adsorbate_elements)
    species_list = (nx_graph.nodes[node]['element'] for node in nx_graph.nodes)
    edge_tails_heads = [(edge[0], edge[1]) for edge in nx_graph.edges]
    edge_tails = [x for x, y in edge_tails_heads] + [y for x, y in edge_tails_heads]
    edge_heads = [y for x, y in edge_tails_heads] + [x for x, y in edge_tails_heads]
    edge_attrs = [nx_graph.edges[edge]['ts_edge'] for edge in nx_graph.edges] + [nx_graph.edges[edge]['ts_edge'] for edge in nx_graph.edges]
    elem_array = np.array(list(species_list)).reshape(-1, 1)
    elem_enc = one_hot_encoder.transform(elem_array).toarray()
    edge_index = torch.tensor([edge_tails, edge_heads], dtype=torch.long)
    x = torch.from_numpy(elem_enc).float()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) 
    graph = Data(x, edge_index, edge_attr)
    print(graph)
    return graph, surface_neighbors, bb_idxs


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