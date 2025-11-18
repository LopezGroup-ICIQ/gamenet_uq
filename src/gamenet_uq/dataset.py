""" Module containing the class for the generation of the PyG dataset from the ASE database."""

from collections import defaultdict
import os
from typing import Union, Optional
from copy import deepcopy
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
from tqdm import tqdm


from torch_geometric.data import InMemoryDataset, Data
from torch import load, save, tensor
import torch
import torch.multiprocessing as mp
from ase.db import connect
from ase.db.core import AtomsRow
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from ase.atoms import Atoms
from ase.io import read

from gamenet_uq.constants import ADSORBATE_ELEMS, METALS, OHE_ELEMENTS
from gamenet_uq.graph_filters import H_filter, C_filter, fragment_filter, ase_adsorption_filter, is_ring
from gamenet_uq.graph import atoms_to_pyg
from gamenet_uq.node_featurizers import get_gcn, get_radical_atoms, get_atom_valence, adsorbate_node_featurizer, get_magnetization


def pyg_dataset_id(ase_database_path: str, 
                   graph_params: dict) -> str:
    """
    Return dataset identifier based on the graph conversion settings.
    
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
    surface_order = str(structure_params["surface_order"])
    # extract node features parameters
    features_params = graph_params["features"]
    adsorbate = str(features_params["adsorbate"])
    radical = str(features_params["radical"])
    valence = str(features_params["valence"])
    gcn = str(features_params["gcn"])
    mag = str(features_params["magnetization"])
    target = graph_params["target"]
    # id convention: database name + target + all features. float values converted to strings and "." is removed
    dataset_id = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(id, target, tolerance, scaling_factor, surface_order, adsorbate, radical, valence, gcn, mag)
    return dataset_id


class AdsorptionGraphDataset(InMemoryDataset):
    """
    Graph dataset representing transition state structures and stable intermediates on surfaces.
    Graphs are generated starting from ASE Atoms objects stored in the input ASE database and conversion settings.
    Graphs are stored as torch_geometric.data.Data.
    When the dataset object is instantiated for the first time, two different files are created:
    1) a `processed` directory containing the additional information about the dataset
    2) a zip file containing the graphs in the torch_geometric.data.Data format. The name of the zip file is
        dependent on the conversion settings.

    Args:
        ase_database_path (str): Path to the ASE database.
        graph_dataset_dir (str): Path to the directory where the graph dataset files are stored.
        graph_params (dict): Dictionary containing the information for the graph generation in the format:
                            {"structure": {"tolerance": float, "scaling_factor": float, "surface_order": int},
                             "features": {"adsorbate": bool,
                                          "ring": bool,
                                           "valence": bool, 
                                           "facet": bool, 
                                           "gcn": bool}, 
                             "target": str}
        db_key (str): Key to access specific items of the ase database. Example could be "metal=Pd,nC=2" for selecting
                            adsorbates with 2 C atoms on Pd surfaces.
        ncores (int): Number of cores used for multiprocessing. Default to the number of available cores.
        
    Notes:
        - "target" in graph_params must be a key of the ASE database. Check available keys with `ase db *.db`.
        - Each graph has two labels: graph.y and graph.target. Originally they are the same, 
          but during the trainings graph.target represents the 
          original value (energy in eV), while graph.y is the scaled value (unitless scaled energy).
        - Limitation of the graph representation used here is that surface and adsorbate cannot share the same element. 
            For instance, H2O on oxides is not supported.

    Example:
        Generate graph dataset containing only adsorption systems on Pt(111) surface, 
        with adsorbate, radical and facet features, and e_ads_dft as target.
        >>> graph_params = {"structure": {"tolerance": 0.5, "scaling_factor": 1.5, "surface_order": 2},
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
        print("Graph dataset output path: {}".format(self.output_path))
        self.ncores = ncores
        self.adsorbate_elems = ADSORBATE_ELEMS
        self.elements_list = ADSORBATE_ELEMS + METALS
        self.ohe_elements = OHE_ELEMENTS
        self.node_feature_list = list(self.ohe_elements.categories_[0])
        self.node_dim = len(self.node_feature_list)
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

        batch_size = 2500  # Adjust based on your memory constraints
        data_list = []
        for i in range(0, len(args), batch_size):
            print("Processing batch {} to {} ...".format(i, i + batch_size))
            batch_args = args[i:i + batch_size]
            data_list.extend(deepcopy(process_batch(batch_args)))  # deepcopy to avoid memory issues
        # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

        print("Removing duplicated data ...")
        data_list = [g for g in data_list if g is not None]    
        grouped_graphs = defaultdict(list)
        def key_fn(graph: Data):
            return (graph.formula, graph.facet, graph.metal, graph.type,
                    graph.num_nodes, graph.num_edges)
        dataset = []
        for graph in tqdm(data_list):
            key = key_fn(graph)
            is_dup = False
            for rival in grouped_graphs[key]:
                if np.abs(graph.y - rival.y) > 0.01:
                    continue
                if getattr(graph, 'bb_type', None) != getattr(rival, 'bb_type', None):
                    continue
                is_dup = True
                break
            if not is_dup:
                grouped_graphs[key].append(graph)
                dataset.append(graph)
        print("Graph dataset size: {}".format(len(dataset)))
        data, slices = self.collate(dataset)
        save((data, slices), self.processed_paths[0])

    def row_to_data(self,
                    row: AtomsRow,
                    ohe_elements: OneHotEncoder = OHE_ELEMENTS,
                    target: str = "scaled_energy",
                    adsorbate_elements: list[str] = ADSORBATE_ELEMS) -> Optional[Data]:
        """
        Generate PyG graph from ASE database row.
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
            graph = atoms_to_pyg(atoms,
                            calc_type,
                            self.graph_structure_params["tolerance"], 
                            self.graph_structure_params["scaling_factor"],
                            self.graph_structure_params["surface_order"], 
                            ohe_elements, 
                            adsorbate_elements)
        except:
            return None
        graph.target, graph.y = tensor(float(row.get(target)), dtype=torch.float), tensor(float(row.get(target)), dtype=torch.float)
        graph.formula = formula
        graph.metal = row.get("metal")
        graph.facet = row.get("facet")
        graph.path = row.get("path")
        if graph.bb_idxs != 'None':
            bb_type = [atoms[graph.bb_idxs[0]].symbol, atoms[graph.bb_idxs[1]].symbol]
            graph.bb_type = "-".join(sorted(bb_type))        
            try:
                graph.img_freqs = row.note.split()[0]
            except ValueError:
                graph.img_freqs = "N/A"
        else: 
            graph.bb_type = 'None'
            graph.img_freqs = "None" 
        graph.note = row.note
        graph.node_feats = list(ohe_elements.categories_[0])
        graph.edge_feats = ["ts"]
        graph.e_mol = row.get("e_mol")
        graph.has_ring = is_ring(graph, adsorbate_elements)  # adsorbate with ring
        graph.author = row.get("author")
        try:
            graph.slab_size = int(row.get("nslab"))
            graph.mag = float(row.get("mag"))
        except:
            graph.slab_size = 0
            graph.mag = 0
        for filter in [H_filter, C_filter, fragment_filter]:
            if not filter(graph, adsorbate_elements):
                return None 
        
        # NODE FEATURIZATION
        if self.node_feats_params["adsorbate"]:
            graph = adsorbate_node_featurizer(graph, adsorbate_elements)
        if self.node_feats_params["radical"]:
            graph = get_radical_atoms(graph, adsorbate_elements)
        if self.node_feats_params["valence"]:
            graph = get_atom_valence(graph, adsorbate_elements)
        if self.node_feats_params["gcn"]:
            gcn = get_gcn(atoms, adsorbate_elements)
            gcn_col = torch.zeros((graph.x.shape[0], 1))
            for i, _ in enumerate(graph.x):
                gcn_col[i] = gcn[graph.ase_indices[i]][0]
            graph.x = torch.cat((graph.x, gcn_col), dim=1)
            graph.node_feats.append("gcn")
        if self.node_feats_params["magnetization"]:
            graph = get_magnetization(graph, adsorbate_elements, self.node_feats_params)

        # include graph only if magnetization is available or reasonable
        mag_dict = {"Fe": 2.2, "Co": 1.7, "Ni": 0.6}
        if graph.metal in ["Fe", "Ni", "Co"]:
            if graph.mag == 0:
                return None
            else:
                if abs(graph.mag/graph.slab_size - mag_dict[graph.metal]) / mag_dict[graph.metal] > 0.25:
                    return None
                else:
                    return graph
        return graph
    

def atoms_to_data(structure: Union[Atoms, str], 
                  graph_params: dict[str, Union[float, int, bool]], 
                  model_elems: list[str] = ADSORBATE_ELEMS + METALS, 
                  calc_type: str='int', 
                  adsorbate_elements =ADSORBATE_ELEMS) -> Data:
    """
    Convert ASE objects to PyG graphs for inference.
    (target values are not included in the Data object).

    Args:
        structure (Atoms): ASE atoms object or POSCAR/CONTCAR file.
        graph_params (dict): Dictionary containing the information for the graph generation in the format:
                            {"tolerance": float, "scaling_factor": float, "metal_hops": int, "second_order_nn": bool}
        model_elems (list): List of chemical elements that can be processed by the model.
        calc_type (str): Type of calculation. "int" for intermediates, "ts" for transition states.
        adsorbate_elements (list): List of adsorbate elements. Default to ["C", "H", "O", "N", "S"].
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
    ohe_elements = OneHotEncoder().fit(np.array(model_elems).reshape(-1, 1)) 
    elements_list = list(ohe_elements.categories_[0])
    node_features_list = list(ohe_elements.categories_[0]) 
    # append to node_features_list the key features whose value is True, in uppercase
    for key, value in graph_features_params.items():
        if value:
            node_features_list.append(key.upper())
    graph, surf_atoms, _ = atoms_to_pyg(structure, 
                                        calc_type,
                                        graph_structure_params["tolerance"], 
                                        graph_structure_params["scaling_factor"],
                                        graph_structure_params["second_order_nn"], 
                                        ohe_elements, 
                                        adsorbate_elements)
    graph.type = calc_type    
    graph.formula = formula
    graph.node_feats = node_features_list
    # node featurization
    if graph_features_params["adsorbate"]:
        graph = adsorbate_node_featurizer(graph, adsorbate_elements)
    if graph_features_params["radical"]:
        graph = get_radical_atoms(graph, adsorbate_elements)
    if graph_features_params["valence"]:
        graph = get_atom_valence(graph, adsorbate_elements)
    if graph_features_params["gcn"]:
        gcn = get_gcn(structure, adsorbate_elements)
        gcn_col = torch.zeros((graph.x.shape[0], 1))
        for i, _ in enumerate(graph.x):
            gcn_col[i] = gcn[graph.ase_indices[i]][0]
        graph.x = torch.cat((graph.x, gcn_col), dim=1)
        graph.node_feats.append("gcn")
    if graph_features_params["magnetization"]:
        graph = get_magnetization(graph)

    for filter in [H_filter, C_filter, fragment_filter]:
        if not filter(graph, adsorbate_elements):
            return None 
    return graph