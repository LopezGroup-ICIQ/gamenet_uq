""" Module containing the class for the generation of the PyG dataset from the ASE database."""

import os
from typing import Union, Optional
from copy import deepcopy
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))


from torch_geometric.data import InMemoryDataset, Data
from torch import zeros, where, cat, load, save, tensor
import torch
import torch.multiprocessing as mp
# mp.set_forkserver_preload(["torch", "torch_geometric"])
from ase.db import connect
from ase.db.core import AtomsRow
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from ase.atoms import Atoms
from ase.io import read

from gamenet_uq.graph_filters import adsorption_filter, H_filter, C_filter, fragment_filter, ase_adsorption_filter
from gamenet_uq.graph import atoms_to_pyg
from gamenet_uq.node_featurizers import get_gcn, get_radical_atoms, get_atom_valence, adsorbate_node_featurizer, get_magnetization

METALS = ["Ag", "Au", "Cd", "Co", "Cu", "Fe", "Ir", "Ni", "Os", "Pd", "Pt", "Rh", "Ru", "Zn"]
ADSORBATE_ELEMS = ["C", "H", "O", "N", "S"]
OHE_ELEMENTS = OneHotEncoder().fit(np.array(ADSORBATE_ELEMS + METALS).reshape(-1, 1))


def pyg_dataset_id(ase_database_path: str, 
                   graph_params: dict) -> str:
    """
    Provide dataset identifier based on the provided graph conversion settings.
    
    Args:
        ase_database_path (str): Path to the ASE database containing the adsorption data.
        graph_params (dict): Dictionary containing the information for the graph generation 
                             in the format:
                            {"structure": {"tolerance": float, "scaling_factor": float, "second_order": int},
                             "features": {"encoder": OneHotEncoder, "adsorbate": bool, "ring": bool, "aromatic": bool, "radical": bool, "valence": bool, "facet": bool}}
    Returns:
        dataset_id (str): PyG dataset identifier.
    """
    id = ase_database_path.split("/")[-1].split(".")[0]
    # extract graph structure conversion params
    structure_params = graph_params["structure"]
    tolerance = str(structure_params["tolerance"]).replace(".", "")
    scaling_factor = str(structure_params["scaling_factor"]).replace(".", "")
    second_order_nn = str(structure_params["second_order"])
    # extract node features parameters
    features_params = graph_params["features"]
    adsorbate = str(features_params["adsorbate"])
    radical = str(features_params["radical"])
    valence = str(features_params["valence"])
    gcn = str(features_params["gcn"])
    mag = str(features_params["magnetization"])
    target = graph_params["target"]
    # id convention: database name + target + all features. float values converted to strings and "." is removed
    dataset_id = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(id, target, tolerance, scaling_factor, second_order_nn, adsorbate, radical, valence, gcn, mag)
    return dataset_id


class AdsorptionGraphDataset(InMemoryDataset):
    """
    Generate graph dataset representing transition states and intermediates on metal surfaces.
    Graphs are generated starting from the structures stored in an ASE database and conversion settings.
    Graphs are stored as torch_geometric.data.Data.
    When the dataset object is instantiated for the first time, two different files are created:
    1) a `processed` directory containing the additional information about the dataset
    2) a zip file containing the graphs in the torch_geometric.data.Data format. The name of the zip file is
        dependent on the conversion settings.

    Args:
        ase_database_name (str): Path to the ASE database.
        graph_dataset_dir (str): Path to the directory where the graph dataset files are stored.
        graph_params (dict): Dictionary containing the information for the graph generation in the format:
                            {"structure": {"tolerance": float, "scaling_factor": float, "second_order": bool},
                             "features": {"adsorbate": bool,
                                          "ring": bool,
                                           "valence": bool, 
                                           "facet": bool, 
                                           "gcn": bool}, 
                             "target": str}
        database_key (str): Key to access specific items of the ase database. Example could be "metal=Pd,nC=2" for selecting
                            only adsorbates with 2 C atoms on Pd surfaces.
        
    Notes:
        - "target" in graph_params must be a key of the ASE database. Check available keys with `ase db *.db`.
        - Each graph has two labels: graph.y and graph.target. Originally they are the same, 
          but during the trainings graph.target represents the 
          original value (energy in eV), while graph.y is the scaled value (unitless scaled energy).

    Example:
        Generate graph dataset containing only adsorption systems on Pt(111) surface, 
        with adsorbate, radical and facet features, and e_ads_dft as target.
        >>> graph_params = {"structure": {"tolerance": 0.5, "scaling_factor": 1.5},
                            "features": {"adsorbate": True, "radical": True, "valence": False, "gcn": False, "magnetization": False},
                            "target": "scaled_energy"}
        >>> ase_database_path = "path/to/ase/database"
        >>> graph_dataset_dir = "path/to/graph/dataset"
        >>> dataset = AdsorptionGraphDataset(ase_database_path, graph_dataset_dir, graph_params, "calc_type=ts,facet=fcc(111),metal=Pt")
    """

    def __init__(self,
                 ase_database_path: str,
                 graph_dataset_dir: str,
                 graph_params: dict[str, Union[dict[str, bool | float], str]], 
                 db_key: str, 
                 ncores: int=os.cpu_count()):     
        self.dataset_id = pyg_dataset_id(ase_database_path, graph_params)
        self.db_key = db_key
        self.ase_database_path = ase_database_path
        self.root = os.path.dirname(ase_database_path)
        self.graph_structure_params = graph_params["structure"]
        self.node_feats_params = graph_params["features"]    
        self.target = graph_params["target"]
        self.output_path = os.path.join(os.path.abspath(graph_dataset_dir), self.dataset_id)
        self.ncores = ncores
        self.adsorbate_elems = ADSORBATE_ELEMS
        self.elements_list = ADSORBATE_ELEMS + METALS
        self.ohe_elements = OneHotEncoder().fit(np.array(self.elements_list).reshape(-1, 1)) 
        self.node_feature_list = list(self.ohe_elements.categories_[0])
        self.node_dim = len(self.node_feature_list)
        # self.duplicates = []
        for key, value in graph_params["features"].items():
            if value:
                self.node_dim += 1
                self.node_feature_list.append(key.upper())
        super().__init__(root=os.path.abspath(graph_dataset_dir))
        self.data, self.slices = load(self.processed_paths[0])    

    @property
    def raw_file_names(self): 
        return self.ase_database_path
    
    @property
    def processed_file_names(self): 
        return self.output_path
    
    def download(self):
        pass
    
    def process(self):  
        db = connect(self.ase_database_path)    
        args = []
        for row in db.select(self.db_key):
            args.append(row)

        def process_batch(batch_args):
            with mp.Pool(mp.cpu_count()) as pool:
                return pool.map(self.row_to_data, batch_args)

        batch_size = 2000  # Adjust based on your memory constraints
        data_list = []
        for i in range(0, len(args), batch_size):
            print("Processing batch {} to {} ...".format(i, i + batch_size))
            batch_args = args[i:i + batch_size]
            data_list.extend(deepcopy(process_batch(batch_args)))

        # Isomorphism test
        print("Removing duplicated data ...")
        dataset = []
        for graph in data_list:
            if graph != None:
                is_duplicate, iso_idx = self.is_duplicate(graph, dataset)
                if not is_duplicate:
                    dataset.append(graph)
                # else:
                #     self.duplicates.append((graph, dataset[iso_idx]))  # for testing purposes

            else:
                continue
        print("Graph dataset size: {}".format(len(dataset)))
        data, slices = self.collate(dataset)
        save((data, slices), self.processed_paths[0])

    def is_duplicate(self, 
                     graph: Data, 
                     graph_list: list[Data], 
                     eps: float=0.01) -> tuple:
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
            return False, None
        else:
            for rival in graph_list:
                if graph.type != rival.type:
                    continue
                if graph.num_edges != rival.num_edges:
                    continue
                if graph.num_nodes != rival.num_nodes:
                    continue
                if graph.formula != rival.formula:
                    continue
                if np.abs(graph.y - rival.y) > eps:
                    continue
                if graph.facet != rival.facet:
                    continue
                if graph.metal != rival.metal:
                    continue
                if graph.bb_type != rival.bb_type: # for TSs
                    continue
                print("Isomorphism detected for {}".format(graph.formula))
                return True, graph_list.index(rival)
            return False, None

    def row_to_data(self,
                    row: AtomsRow,
                    ohe_elements: OneHotEncoder = OHE_ELEMENTS,
                    target: str = "scaled_energy",
                    adsorbate_elements: list[str] = ADSORBATE_ELEMS) -> Optional[Data]:
        """
        Generate PyG Data object from ASE database row.
        Used for multiprocessing.

        Args:
            row (AtomsRow): ASE database row.
            ohe_elements (OneHotEncoder): One-hot encoder for chemical elements.
            target (str): Target value for the graph.
            adsorbate_elements (list): List of adsorbate elements.

        Returns:
            graph (Data): PyG Data object.
        """
        # GRAPH STRUCTURE GENERATION
        atoms = row.toatoms()
        formula = atoms.get_chemical_formula(mode='metal')
        calc_type = row.get("calc_type")
        if not ase_adsorption_filter(atoms, adsorbate_elements):
            return None
        
        try:
            graph, surf_atoms, bb_idxs = atoms_to_pyg(atoms,
                                                    calc_type,
                                                    self.graph_structure_params["tolerance"], 
                                                    self.graph_structure_params["scaling_factor"],
                                                    self.graph_structure_params["second_order"], 
                                                    ohe_elements, 
                                                    adsorbate_elements)
        except:
            print("Error in graph generation for {}\n".format(formula))
            return None
        graph.target, graph.y = tensor(float(row.get(target)), dtype=torch.float), tensor(float(row.get(target)), dtype=torch.float)
        graph.bb_idxs = bb_idxs if bb_idxs != None else 'None'
        graph.formula = formula
        graph.type = calc_type
        graph.metal = row.get("metal")
        graph.facet = row.get("facet")
        if bb_idxs != None:
            bb_type = [atoms[bb_idxs[0]].symbol, atoms[bb_idxs[1]].symbol]
            graph.bb_type = "-".join(sorted(bb_type))        
            try:
                graph.img_freqs = row.note.split()[0]
            except ValueError:
                graph.img_freqs = "N/A"
        else: 
            graph.bb_type = 'None'
            graph.img_freqs = "None" 
        graph.calc_path = row.get("calc_path")
        graph.node_feats = list(ohe_elements.categories_[0])
        graph.edge_feats = ["ts"]
        graph.e_mol = row.get("e_mol")
        for filter in [adsorption_filter, H_filter, C_filter, fragment_filter]:
            if not filter(graph, adsorbate_elements):
                return None 
        
        # NODE FEATURIZATION
        try:
            if self.node_feats_params["adsorbate"]:
                graph = adsorbate_node_featurizer(graph, adsorbate_elements)
            if self.node_feats_params["radical"]:
                graph = get_radical_atoms(graph, adsorbate_elements)
            if self.node_feats_params["valence"]:
                graph = get_atom_valence(graph, adsorbate_elements)
            if self.node_feats_params["gcn"]:
                graph = get_gcn(graph, atoms, adsorbate_elements, surf_atoms)
            if self.node_feats_params["magnetization"]:
                graph = get_magnetization(graph, adsorbate_elements, self.node_feats_params)
            return graph
        except:
            print("Error in node featurization for {}\n".format(formula))
            return None
    

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
    graph, surface_neighbours = atoms_to_pyg(structure, 
                                                  graph_structure_params["tolerance"], 
                                                  graph_structure_params["scaling_factor"],
                                                  graph_structure_params["second_order_nn"], 
                                                  ohe_elements, 
                                                  adsorbate_elements)
    graph.type = calc_type
    if not adsorption_filter(graph, ohe_elements, adsorbate_elements):
        raise ValueError("Adsorption filter failed for {}".format(formula))
    if not H_filter(graph, ohe_elements, adsorbate_elements):
        raise ValueError("H connectivity filter failed for {}".format(formula))
    if not C_filter(graph, ohe_elements, adsorbate_elements):
        raise ValueError("C connectivity filter failed for {}".format(formula))
    if not fragment_filter(graph, ohe_elements, adsorbate_elements):
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