""" Module containing the class for the generation of the PyG dataset from the ASE database."""

import os, sys
from typing import Union
import multiprocessing as mp

from torch_geometric.data import InMemoryDataset, Data
from torch import zeros, where, cat, load, save, tensor
import torch
from ase.db import connect
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from ase.atoms import Atoms
from ase.io import read
import networkx as nx
import pandas as pd

from fats.graph_filters import adsorption_filter, H_connectivity_filter, C_connectivity_filter, single_fragment_filter, ase_adsorption_filter
from fats.graph_tools import extract_adsorbate
from fats.functions import atoms_to_pyggraph, get_voronoi_neighbourlist


def pyg_dataset_id(ase_database_path: str, 
                   graph_params: dict) -> str:
    """
    Provide dataset string identifier based on the provided graph parameters.
    
    Args:
        ase_database_path (str): Path to the ASE database containing the adsorption data.
        graph_params (dict): Dictionary containing the information for the graph generation 
                             in the format:
                            {"structure": {"tolerance": float, "scaling_factor": float, "second_order_nn": int},
                             "features": {"encoder": OneHotEncoder, "adsorbate": bool, "ring": bool, "aromatic": bool, "radical": bool, "valence": bool, "facet": bool}}
    Returns:
        dataset_id (str): PyG dataset identifier.
    """
    id = ase_database_path.split("/")[-1].split(".")[0]
    # extract graph structure conversion params
    structure_params = graph_params["structure"]
    tolerance = str(structure_params["tolerance"]).replace(".", "")
    scaling_factor = str(structure_params["scaling_factor"]).replace(".", "")
    metal_hops = str(structure_params["second_order_nn"])
    # extract node features parameters
    features_params = graph_params["features"]
    adsorbate = str(features_params["adsorbate"])
    radical = str(features_params["radical"])
    valence = str(features_params["valence"])
    gcn = str(features_params["gcn"])
    mag = str(features_params["magnetization"])
    target = graph_params["target"]
    # id convention: database name + target + all features. float values converted to strings and "." is removed
    dataset_id = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(id, target, tolerance, scaling_factor, metal_hops, adsorbate, radical, valence, gcn, mag)
    return dataset_id


def get_gcn(atoms_obj: Atoms, 
            adsorbate_elements: list[str]) -> dict[int, tuple[float, float]]:
    """
    Return the (normalized) generalized coordination number (gcn) for each atom in the ASE atoms object.
    gcn is defined as the sum of the coordination numbers of the neighbours divided by the maximum coordination number.
    gcn=0 atom alone; gcn=1 bulk atom; 0<gcn<1=surface atom.

    Args:
        atoms_obj (Atoms): ASE atoms object containing a slab with an adsorbate
        adsorbate_elements (list[str]): list of symbols of the adsorbate elements

    Returns:
        dict[int, tuple]: dictionary with the generalized coordination number (gcn) and traditional cn for each atom
    """

    y = get_voronoi_neighbourlist(atoms_obj, 0.25, 1.2, adsorbate_elements)
    neighbour_dict = {}
    for atom_index, atom in enumerate(atoms_obj):
        coordination_number = 0
        neighbour_list = []
        for row in y:
            if atom_index in row:
                neighbour_index = row[0] if row[0] != atom_index else row[1]
                if atoms_obj[neighbour_index].symbol in adsorbate_elements:
                    coordination_number += 0  # Consider slab only (no adsorbate)
                else:
                    coordination_number += 1
                    neighbour_list.append((atoms_obj[neighbour_index].symbol, neighbour_index, atoms_obj[neighbour_index].position[2])) 
            else:
                continue
        neighbour_dict[atom_index] = (coordination_number, atom.symbol, neighbour_list)
    max_coordination_number = max([neighbour_dict[i][0] for i in neighbour_dict.keys()])
    gcn_dict = {}
    for atom_index in neighbour_dict.keys():
        if atoms_obj[atom_index].symbol in adsorbate_elements:
            gcn_dict[atom_index] = (None, neighbour_dict[atom_index][0])
            continue
        cn_sum = 0.0
        for neighbour in neighbour_dict[atom_index][2]:
            cn_sum += neighbour_dict[neighbour[1]][0]
        gcn_dict[atom_index] = (cn_sum / max_coordination_number ** 2, neighbour_dict[atom_index][0])
    return gcn_dict


def get_radical_atoms(atoms_obj: Atoms, 
                      adsorbate_elements: list[str]) -> list[int]:
    """
    Detect atoms in the molecule which are radicals with RDKit.

    Args:
        atoms_obj (ase.Atoms): ASE atoms object of the adsorption structure
        adsorbate_elements (list[str]): List of elements in the adsorbates (e.g. ["C", "H", "O", "N", "S"])
    
    Returns:
        radical_atoms (list[int]): List of indices of the radical atoms in the atoms_obj.
    """

    molecule_atoms_obj = Atoms()
    molecule_atoms_obj.set_cell(atoms_obj.get_cell())
    molecule_atoms_obj.set_pbc(atoms_obj.get_pbc())
    for atom in atoms_obj:
        if atom.symbol in adsorbate_elements:
            molecule_atoms_obj.append(atom)
    atomic_symbols = molecule_atoms_obj.get_chemical_symbols()
    coordinates = molecule_atoms_obj.get_positions()
    xyz = '\n'.join(f'{symbol} {x} {y} {z}' for symbol, (x, y, z) in zip(atomic_symbols, coordinates))
    xyz = "{}\n\n{}".format(len(molecule_atoms_obj), xyz)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    conn_mol = Chem.Mol(rdkit_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    Chem.SanitizeMol(conn_mol, Chem.SANITIZE_FINDRADICALS  ^ Chem.SANITIZE_SETHYBRIDIZATION)
    radical_atoms = [atom.GetIdx() for atom in conn_mol.GetAtoms() if atom.GetNumRadicalElectrons() > 0]
    return radical_atoms


def get_atom_valence(atoms_obj: Atoms,
                     adsorbate_elements: list[str]) -> list[float]:
    """
    For each atom in the adsorbate, calculate the valence.
    Valence is defined as (x_max - x) / x_max, where x is the degree of the atom,
    and x_max is the maximum degree of the atom in the molecule. Bond order is not taken into account.
    valence=0 atom alone; valence=1 fully saturated atom.

    Ref: https://doi.org/10.1103/PhysRevLett.99.016105

    Args:
        atoms_obj (ase.Atoms): ASE atoms object.
        molecule_elements (list[str]): List of elements in the adsorbates (e.g. ["C", "H", "O", "N", "S"])
    Returns:
        valence (list[float]): List of valences for each atom in the molecule.
    """
    molecule_atoms_obj = Atoms()
    molecule_atoms_obj.set_cell(atoms_obj.get_cell())
    molecule_atoms_obj.set_pbc(atoms_obj.get_pbc())
    for atom in atoms_obj:
        if atom.symbol in adsorbate_elements:
            molecule_atoms_obj.append(atom)
    atomic_symbols = molecule_atoms_obj.get_chemical_symbols()
    coordinates = molecule_atoms_obj.get_positions()
    xyz = '\n'.join(f'{symbol} {x} {y} {z}' for symbol, (x, y, z) in zip(atomic_symbols, coordinates))
    xyz = "{}\n\n{}".format(len(molecule_atoms_obj), xyz)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    conn_mol = Chem.Mol(rdkit_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    Chem.SanitizeMol(conn_mol, Chem.SANITIZE_FINDRADICALS  ^ Chem.SANITIZE_SETHYBRIDIZATION)
    degree_vector = np.vectorize(lambda x: x.GetDegree())
    max_degree_vector = np.vectorize(lambda x: Chem.GetPeriodicTable().GetDefaultValence(x.GetAtomicNum()))
    atom_array = np.array([i for i in conn_mol.GetAtoms()]).reshape(-1, 1)
    degree_array = degree_vector(atom_array) / max_degree_vector(atom_array)  # valence = x / x_max in order to have the same trend as gcn 
    valence = degree_array.reshape(-1, 1)
    return valence


def isomorphism_test(graph: Data, 
                     graph_list: list[Data], 
                     eps: float=0.01) -> bool:
    """
    Perform isomorphism test for the input graph before including it in the final dataset.
    Test based on graph formula and energy difference.

    Args:
        graph (Data): Input graph.
        graph_list (list[Data]): graph list against which the input graph is tested.
        eps (float): tolerance value for the energy difference in eV. Default to 0.01 eV.
        grwph: data graph as input
    Returns:
        (bool): Whether the graph passed the isomorphism test.
    """
    if len(graph_list) == 0:
        return True 
    if graph == None:
        return False
    for rival_graph in graph_list:
        c1 = graph.num_edges == rival_graph.num_edges
        c2 = graph.num_nodes == rival_graph.num_nodes
        c3 = graph.formula == rival_graph.formula
        c4 = np.abs(graph.y - rival_graph.y) < eps
        c5 = graph.facet == rival_graph.facet
        if c1 and c2 and c3 and c4 and c5:
            return False
        else:
            continue
    return True


class TransitionStateGraphDataset(InMemoryDataset):
    """
    Generate graph dataset representing molecules adsorbed on transition metal surfaces.
    It generates the graphs from the provided ASE database and conversion settings.
    Graphs are stored in the torch_geometric.data.Data type.
    When the dataset object is instantiated for the first time, two different files are created:
    1) a `processed` directory containing the additional information about the dataset
    2) a zip file containing the graphs in the torch_geometric.data.Data format. The name of the zip file is
        dependent on the conversion settings.

    Args:
        ase_database_name (str): Path to the ase database containing the adsorption data.
        graph_dataset_dir (str): Path to the directory where the graph dataset files are stored.
        graph_params (dict): Dictionary containing the information for the graph generation in the format:
                            {"structure": {"tolerance": float, "scaling_factor": float, "metal_hops": int},
                             "features": {"adsorbate": bool,
                                          "ring": bool,
                                           "aromatic": bool, 
                                           "radical": bool, 
                                           "valence": bool, 
                                           "facet": bool, 
                                           "gcn": bool}, 
                             "target": str}
        database_key (str): Key to access specific items of the ase database. Default to "calc_type=adsorption".
        
    Notes:
        - "target" in graph_params must be a key of the ASE database.
        - Each graph object has two labels: graph.y and graph.target. Originally they are the same, 
          but during the trainings graph.target represents the 
          original value (adsorption energy in eV), while graph.y is the scaled value (
          unitless scaled adsorption energy).

    Example:
        Generate graph dataset containing only adsorption systems on Pt(111) surface, 
        with adsorbate, ring, aromatic, radical and facet features, and e_ads_dft as target.
        >>> graph_params = {"structure": {"tolerance": 0.5, "scaling_factor": 1.5, "metal_hops": False},
                            "features": {"adsorbate": True, "ring": True, "aromatic": True, "radical": True, "valence": False, "facet": True},
                            "target": "e_ads_dft"}
        >>> ase_database_path = "path/to/ase/database"
        >>> graph_dataset_dir = "path/to/graph/dataset"
        >>> dataset = AdsorptionGraphDataset(ase_database_path, graph_dataset_dir, graph_params, "calc_type=adsorption,facet=fcc(111),metal=Pt")
    """

    def __init__(self,
                 ase_database_path: str,
                 graph_dataset_dir: str,
                 graph_params: dict[str, Union[dict, str]], 
                 database_key: str="calc_type=transition_state"):        
        self.dataset_id = pyg_dataset_id(ase_database_path, graph_params)
        self.ase_database_path = ase_database_path
        self.root = os.path.dirname(ase_database_path)
        self.graph_structure_params = graph_params["structure"]
        self.graph_features_params = graph_params["features"]    
        self.target = graph_params["target"]
        self.database_key = database_key
        self.output_path = os.path.join(os.path.abspath(graph_dataset_dir), self.dataset_id)
        # Construct OHEs for elements and surface orientation (based on the selected data defined by database_key)
        db = connect(self.ase_database_path)
        self.elements_list, self.surface_orientation_list = [], []
        for row in db.select(database_key):
            chemical_symbols = set(row.toatoms().get_chemical_symbols())    
            for element in chemical_symbols:
                if element not in self.elements_list:
                    self.elements_list.append(element)
        self.molecule_elements = [elem for elem in self.elements_list if elem in ["C", "H", "O", "N", "S"]]
        self.ohe_elements = OneHotEncoder().fit(np.array(self.elements_list).reshape(-1, 1)) 
        # Initialize counters
        self.graph_dataset_size = 0 
        # Filter counters
        self.counter_isomorphism = 0
        self.counter_H_filter = 0
        self.counter_C_filter = 0
        self.counter_fragment_filter = 0
        self.counter_adsorption_filter = 0
        self.counter_ase_filter = 0
        # Filter bins
        self.bin_isomorphism = []
        self.bin_H_filter = []
        self.bin_C_filter = []
        self.bin_fragment_filter = []
        self.bin_adsorption_filter = []
        self.bin_ase_filter = []
        self.bin_unconverted_atoms_objects = [] 
        # Node features
        self.node_feature_list = list(self.ohe_elements.categories_[0])
        self.node_dim = len(self.node_feature_list)
        if graph_params["features"]["adsorbate"]:
            self.node_dim += 1
            self.node_feature_list.append("Adsorbate")
        if graph_params["features"]["radical"]:
            self.node_dim += 1
            self.node_feature_list.append("Radical")
        if graph_params["features"]["valence"]:
            self.node_dim += 1
            self.node_feature_list.append("Valence")
        if graph_params["features"]["gcn"]:
            self.node_dim += 1
            self.node_feature_list.append("gcn")
        if graph_params["features"]["magnetization"]:
            self.node_dim += 1
            self.node_feature_list.append("Magnetization")
        super().__init__(root=os.path.abspath(graph_dataset_dir))
        self.data, self.slices = load(self.processed_paths[0])    

    @property
    def raw_file_names(self): 
        return self.ase_database_path
    
    @property
    def processed_file_names(self): 
        """
        Return the name of the processed file containing the PyG data objects.
        """
        return self.output_path
    
    def download(self):
        pass
    
    def process(self):  
        db = connect(self.ase_database_path)    
        args_list = []
        for row in db.select(self.database_key):  # multiprocessing
            args_list.append((row, self.graph_structure_params, self.ohe_elements, self.molecule_elements, self.graph_features_params, self.target))
        with mp.Pool(os.cpu_count()) as pool:
            data_list = pool.starmap(gen_pyggraph, args_list)
        print(len(data_list))
        data_list = [graph for graph in data_list if graph != None]
        graph_dataset = []
        for graph in data_list:
            if isomorphism_test(graph, graph_dataset, 0.02):
                graph_dataset.append(graph)
                print("{} added to dataset".format(graph.formula))
            else:
                continue
        print("Graph dataset size: {}".format(len(graph_dataset)))
        data, slices = self.collate(data_list)
        save((data, slices), self.processed_paths[0])


def gen_pyggraph(row,
                 graph_structure_params: dict[str, Union[float, int, bool]],
                 ohe_elements: OneHotEncoder,
                 molecule_elements: list[str],
                 graph_features_params: dict[str, bool],
                 target: str) -> Data:
    """
    Generate PyG Data object from ASE database row.
    Used for multiprocessing.

    Args:

    Returns:
        graph (Data): PyG Data object.
    """
    elements_list = list(ohe_elements.categories_[0]) 
    molecule_elements_indices = [elements_list.index(element) for element in molecule_elements]
    atoms_obj = row.toatoms()
    calc_type = row.get("calc_type")
    formula = row.get("formula")
    metal = row.get("metal")
    facet = row.get("facet")
    # 1) PRIMITIVE GRAPH STRUCTURE GENERATION
    try:
        graph, surface_neighbours, bb_idxs = atoms_to_pyggraph(atoms_obj, 
                              graph_structure_params["tolerance"], 
                              graph_structure_params["scaling_factor"], 
                              ohe_elements, 
                              molecule_elements)
        print("Graph generated for {}".format(formula))
    except:
        print("FUUUUUUUCKKK")
        return None
    
    # 2) GRAPH LABELLING AND INITIAL FILTERING
    y = tensor(float(row.get(target)), dtype=torch.float)
    graph.target, graph.y = y, y
    graph.formula, graph.type, graph.metal, graph.facet = formula, calc_type, metal, facet
    graph.atoms_obj = atoms_obj
    graph.bb_idxs = bb_idxs

    # if not ase_adsorption_filter(graph, molecule_elements):
    #     return None
    # if not adsorption_filter(graph, ohe_elements, molecule_elements):
    #     return None
    # if not H_connectivity_filter(graph, ohe_elements, molecule_elements):
    #     return None
    # if not C_connectivity_filter(graph, ohe_elements, molecule_elements):
    #     return None
    if not single_fragment_filter(graph, ohe_elements, molecule_elements):
        return None
    
    # 3) NODE FEATURIZATION
    if graph_features_params["adsorbate"]:
        x_adsorbate = zeros((graph.x.shape[0], 1))
        for i, node in enumerate(graph.x):
            index = where(node == 1)[0][0].item()
            x_adsorbate[i, 0] = 1 if index in molecule_elements_indices else 0
        graph.x = cat((graph.x, x_adsorbate), dim=1)
    if graph_features_params["radical"]:
        x_radical = torch.zeros((graph.x.shape[0], 1))
        radical_atoms = get_radical_atoms(atoms_obj, molecule_elements)
        for index, node in enumerate(graph.x):
            if index in radical_atoms:
                x_radical[index, 0] = 1
        graph.x = torch.cat((graph.x, x_radical), dim=1)
    if graph_features_params["valence"]:
        try:
            x_valence = torch.zeros((graph.x.shape[0], 1))
            scaled_degree_vector = get_atom_valence(atoms_obj, molecule_elements)
            for index, node in enumerate(scaled_degree_vector):
                x_valence[index, 0] = scaled_degree_vector[index, 0]
            graph.x = torch.cat((graph.x, x_valence), dim=1)
        except:
            return None
    if graph_features_params["gcn"]:
        x_generalized_coordination_number = torch.zeros((graph.x.shape[0], 1))
        cn = get_gcn(atoms_obj, molecule_elements)
        counter = 0
        for i, node in enumerate(graph.x):
            index = where(node == 1)[0][0].item()
            if index not in molecule_elements_indices:
                x_generalized_coordination_number[i, 0] = cn[surface_neighbours[counter]][0]
                counter += 1
        graph.x = torch.cat((graph.x, x_generalized_coordination_number), dim=1)
    if graph_features_params["magnetization"]:
        if graph.metal in ("Fe", "Co", "Ni"):
            graph.x = torch.cat((graph.x, torch.ones((graph.x.shape[0], 1))), dim=1)
        else:
            graph.x = torch.cat((graph.x, torch.zeros((graph.x.shape[0], 1))), dim=1)
    return graph
    




    


def atoms_to_data(structure: Union[Atoms, str], 
                  graph_params: dict[str, Union[float, int, bool]], 
                  model_elems: list[str], 
                  calc_type: str='adsorption') -> Data:
    """
    Convert ASE atoms object to PyG Data object based on the graph parameters.
    The implementation is similar to the one in the ASE to PyG converter class, but it is not a class method and 
    is used for inference. Target values are not included in the Data object.

    Args:
        structure (Atoms): ASE atoms object or file to POSCAR/CONTCAR file.
        graph_params (dict): Dictionary containing the information for the graph generation in the format:
                            {"tolerance": float, "scaling_factor": float, "metal_hops": int, "second_order_nn": bool}
        model_elems (list): List of chemical elements that can be processed by the model.
    Returns:
        graph (Data): PyG Data object.
    """
    
    if isinstance(structure, str):  
        structure = read(structure)
    elif not isinstance(structure, Atoms):
        raise TypeError("Structure must be of type ASE Atoms or POSCAR/CONTCAR file path.")
    
    # Get list of elements in the structure
    elements_list = list(set(structure.get_chemical_symbols()))
    if not all(elem in model_elems for elem in elements_list):
        raise ValueError("Not all chemical elements in the structure can be processed by the model.")
    
    # Read graph conversion parameters
    graph_structure_params = graph_params["structure"]
    graph_features_params = graph_params["features"]
    formula = structure.get_chemical_formula()

    # Construct one-hot encoder for elements
    adsorbate_elements = ["C", "H", "O", "N", "S"]  # hard-coded for now
    ohe_elements = OneHotEncoder().fit(np.array(model_elems).reshape(-1, 1)) 
    elements_list = list(ohe_elements.categories_[0])
    node_features_list = list(ohe_elements.categories_[0]) 
    # append to node_features_list the key features whose value is True, in uppercase
    for key, value in graph_features_params.items():
        if value:
            node_features_list.append(key.upper())
    adsorbate_elements_indices = [elements_list.index(element) for element in adsorbate_elements]
    graph, surface_neighbours = atoms_to_pyggraph(structure, 
                                                  graph_structure_params["tolerance"], 
                                                  graph_structure_params["scaling_factor"],
                                                  graph_structure_params["second_order_nn"], 
                                                  ohe_elements, 
                                                  adsorbate_elements)
    graph.type = calc_type
    if not adsorption_filter(graph, ohe_elements, adsorbate_elements):
        raise ValueError("Adsorption filter failed for {}".format(formula))
    if not H_connectivity_filter(graph, ohe_elements, adsorbate_elements):
        raise ValueError("H connectivity filter failed for {}".format(formula))
    if not C_connectivity_filter(graph, ohe_elements, adsorbate_elements):
        raise ValueError("C connectivity filter failed for {}".format(formula))
    if not single_fragment_filter(graph, ohe_elements, adsorbate_elements):
        raise ValueError("Single fragment filter failed for {}".format(formula))
    # node featurization
    if graph_features_params["adsorbate"]:
        x_adsorbate = zeros((graph.x.shape[0], 1))  # 1=adsorbate, 0=metal
        for i, node in enumerate(graph.x):
            index = where(node == 1)[0][0].item()
            x_adsorbate[i, 0] = 1 if index in adsorbate_elements_indices else 0
        graph.x = cat((graph.x, x_adsorbate), dim=1)
    if graph_features_params["radical"]:
        x_radical = torch.zeros((graph.x.shape[0], 1))  # 1=radical, 0=no radical/ metal
        radical_atoms = get_radical_atoms(structure, adsorbate_elements)
        for index, node in enumerate(graph.x):
            if index in radical_atoms:
                x_radical[index, 0] = 1
        graph.x = torch.cat((graph.x, x_radical), dim=1)
    if graph_features_params["valence"]:                
        try:
            x_valence = torch.zeros((graph.x.shape[0], 1))
            scaled_degree_vector = get_atom_valence(structure, adsorbate_elements)
            for index, node in enumerate(scaled_degree_vector):
                x_valence[index, 0] = scaled_degree_vector[index, 0]
            graph.x = torch.cat((graph.x, x_valence), dim=1)
        except:
            raise ValueError("{}: Error in valence detection.".format(formula))               
    if graph_features_params["gcn"]:
        x_generalized_coordination_number = torch.zeros((graph.x.shape[0], 1))
        cn = get_gcn(structure, adsorbate_elements)
        counter = 0
        for i, node in enumerate(graph.x):
            index = where(node == 1)[0][0].item()
            if index not in adsorbate_elements_indices:
                x_generalized_coordination_number[i, 0] = cn[surface_neighbours[counter]][0]
                counter += 1
        graph.x = torch.cat((graph.x, x_generalized_coordination_number), dim=1)
    
    graph.formula = formula
    graph.node_feats = node_features_list
    return graph