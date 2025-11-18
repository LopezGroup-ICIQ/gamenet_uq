"""
This module contains functions and classes for creating, manipulating and analyzying graphs
from ASE Atoms objects to PyG Data format.
"""

from itertools import product

from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
import numpy as np
import torch 
from scipy.spatial import Voronoi
from ase import Atoms
from networkx import Graph, set_node_attributes, connected_components, get_node_attributes
 
from gamenet_uq.constants import CORDERO, ADSORBATE_ELEMS, ELEMENT_DOMAIN, OHE_ELEMENTS
from gamenet_uq.graph_filters import H_filter, C_filter, fragment_filter

def get_voronoi_neighbourlist(atoms: Atoms, 
                              tol: float, 
                              scaling_factor: float, 
                              adsorbate_elems: list[str], 
                              mic: bool=True) -> np.ndarray:
    """
    Get connectivity list from Voronoi analysis, considering periodic boundary conditions.
    Assumption: The surface does not contain elements present in the adsorbate.
    
    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-metal system.
        tol (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the metal atoms.
        adsorbate_elems (list[str]): list of elements present in the adsorbate.
        mic(bool): If True, apply Minimum Image Convention to get distances between atoms.
        
    Returns:
        np.ndarray: connectivity list of the system. Each row represents a pair of connected atoms.

    Notes:
        Each connection is represented once, i.e. if atom A is connected to atom B, the pair (A, B) will be present in the list,
        but not the pair (B, A).
    """

    if len(atoms) == 0:
        return np.array([])
    num_adsorbate_atoms = len([atom for atom in atoms if atom.symbol in adsorbate_elems])
    
    # First necessary condition for two atoms to be linked: Sharing a Voronoi facet
    coords_arr = np.repeat(np.expand_dims(np.copy(atoms.get_scaled_positions()), axis=0), 27, axis=0)
    mirrors = np.repeat(np.expand_dims(np.asarray(list(product([-1, 0, 1], repeat=3))), 1), coords_arr.shape[1], axis=1)
    corrected_coords = np.reshape(coords_arr + mirrors, (coords_arr.shape[0] * coords_arr.shape[1], coords_arr.shape[2]))
    corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    pairs_corr = np.delete(pairs_corr, np.argwhere(pairs_corr[:, 0] == pairs_corr[:, 1]), axis=0)

    increment = 0
    pairs = []
    while True:
        pairs.clear()
        for pair in pairs_corr:
            atom1, atom2 = atoms[pair[0]].symbol, atoms[pair[1]].symbol
            threshold = CORDERO[atom1] + CORDERO[atom2] + tol
            distance = atoms.get_distance(pair[0], pair[1], mic=mic)
            if atom1 in adsorbate_elems and atom2 not in adsorbate_elems:
                threshold += max(scaling_factor + increment - 1.0, 0) * CORDERO[atom2]
            if atom1 not in adsorbate_elems and atom2 in adsorbate_elems:
                threshold += max(scaling_factor + increment - 1.0, 0) * CORDERO[atom1]

            if distance <= threshold:
                pairs.append(pair)
        if num_adsorbate_atoms == 0 or num_adsorbate_atoms == len(atoms):
            return np.sort(np.array(pairs), axis=1)
        for i, j in pairs:
            in_ads_i = atoms[i].symbol in adsorbate_elems
            in_ads_j = atoms[j].symbol in adsorbate_elems
            if in_ads_i ^ in_ads_j:
                return np.sort(np.array(pairs, dtype=int), axis=1)
        else:
            increment += 0.2


def detect_ts(atoms: Atoms,
              adsorbate_elems: list[str], 
              tol: float) -> tuple[int]:
    """
    Return indices of the adsorbate atoms involved in the bond-breaking/forming reaction.

    Args:
        atoms (Atoms): ASE Atoms object representing the transition state image.
        adsorbate_elems (list[str]): list of elements present in the adsorbate.
    Returns:
        tuple[int]: indices of the atoms of the adsorbate involved in the bond-breaking/forming.
    """
    adsorbate_idxs = {atom.index for atom in atoms if atom.symbol in adsorbate_elems}
    neighbourlist = get_voronoi_neighbourlist(atoms, tol, 1.0, adsorbate_elems)
    neighbourlist = [(pair[0], pair[1]) for pair in neighbourlist if (pair[0] in adsorbate_idxs) and (pair[1] in adsorbate_idxs)]    
    nx = Graph()
    nx.add_nodes_from(adsorbate_idxs)  # Fundamental to initialize nodes! e.g. single atom would be neglected as has no edges
    nx.add_edges_from(neighbourlist)
    components = list(connected_components(nx))
    if len(components) != 2:  # more than two components in the graph
        return None
    dist_dict = {}
    for node1 in components[0]:
        for node2 in components[1]:
            dist_dict[(node1, node2)] = atoms.get_distance(node1, node2, mic=True)
    return min(dist_dict, key=dist_dict.get)
    
def atoms_to_nx(
    atoms: Atoms,
    voronoi_tolerance: float,
    scaling_factor: float,
    surface_order: int,
    adsorbate_elements: list[str],
    mode: str,
) -> tuple[Graph, dict]:
    """
    Convert ASE Atoms object to NetworkX graph, representing the adsorbate-surface system.

    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-metal system.
        voronoi_tolerance (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the surface atoms.
        surface_order (int): order of the surface neighbours to be included in the graph. If set to -1,
                            all surface slab is included.
        adsorbate_elements (list[str]): list of elements present in the adsorbate.
        mode (str): whether the graph is created for the TS or the reactant/product. Default to 'ts'.
                    In case of 'ts', the graph will include an edge feature representing the broken bond.
    Returns:
        Graph: NetworkX graph representing the adsorbate-metal system.
    """
    neighbour_list = get_voronoi_neighbourlist(atoms, voronoi_tolerance, scaling_factor, adsorbate_elements)
    adsorption_ensemble = {atom.index for atom in atoms if atom.symbol in adsorbate_elements}
    if mode == "int":
        neighbour_list_adsorbate = [
            (pair[0], pair[1])
            for pair in neighbour_list
            if (pair[0] in adsorption_ensemble) and (pair[1] in adsorption_ensemble)
        ]
        ads_graph = Graph()
        ads_graph.add_edges_from(neighbour_list_adsorbate)
        ads_graph.add_nodes_from(adsorption_ensemble)
        components = list(connected_components(ads_graph))
        while len(components) != 1:  # safeguard for inference
            dist_dict = {}
            for node1 in components[0]:
                for node2 in components[1]:
                    dist_dict[(node1, node2)] = atoms.get_distance(
                        node1, node2, mic=True
                    )
            missing_edge = min(dist_dict, key=dist_dict.get)
            neighbour_list = np.vstack((neighbour_list, missing_edge))
            ads_graph.add_edge(missing_edge[0], missing_edge[1])
            components = list(connected_components(ads_graph))
    if surface_order == -1:
        surface_order = 100
    surf_hops = {0: list(adsorption_ensemble)}
    for _ in range(surface_order):
        surface_ensemble = {
            pair[1] if pair[0] in adsorption_ensemble else pair[0]
            for pair in neighbour_list
            if (pair[0] in adsorption_ensemble and pair[1] not in adsorption_ensemble)
            or (pair[1] in adsorption_ensemble and pair[0] not in adsorption_ensemble)
        }
        surf_hops[_ + 1] = list(surface_ensemble)
        adsorption_ensemble = adsorption_ensemble.union(surface_ensemble)
        if len(adsorption_ensemble) == len(atoms):
            break
    graph = Graph()
    graph.add_nodes_from(list(adsorption_ensemble))
    set_node_attributes(graph, {i: atoms[i].symbol for i in graph.nodes()}, "elem")
    ensemble_neighbour_list = [
        pair
        for pair in neighbour_list
        if pair[0] in graph.nodes() and pair[1] in graph.nodes()
    ]
    graph.add_edges_from(ensemble_neighbour_list, ts_edge=0)
    if mode == "ts":
        broken_bond_idxs = detect_ts(atoms, adsorbate_elements, 0.25)
        graph.add_edge(broken_bond_idxs[0], broken_bond_idxs[1], ts_edge=1)
    return graph


def atoms_to_pyg(atoms: Atoms,
                calc_type: str,
                voronoi_tol: float,
                scaling_factor: float,
                surface_order: int,
                one_hot_encoder: OneHotEncoder = OHE_ELEMENTS, 
                adsorbate_elems: list[str] = ADSORBATE_ELEMS) -> Data:
    """
    Convert ASE Atoms object to PyG Data object, representing the adsorbate-surface system.   

    Args:
        atoms (Atoms): ASE Atoms object.
        calc_type (str): type of calculation performed on the system.
                         "int": intermediate, "ts": transition state.                         
        voronoi_tol (float): Tolerance applied during the graph conversion.
        scaling_factor (float): Scaling factor applied to metal radius of metals.
        one_hot_encoder (OneHotEncoder): One-hot encoder.
        adsorbate_elems (list[str]): list of elements present in the adsorbate.
    Returns:
        graph (torch_geometric.data.Data): graph representation of the transition state.

    Notes:
        The graph is constructed as follows:
        - Nodes: one-hot encoded elements of the adsorbate and the surface atoms in contact with it.
        - Edges: connectivity list of the adsorbate-surface system.
        - Edge features: 1 if the edge corresponds to the broken bond in the TS, 0 otherwise.
    """
    if calc_type not in ["int", "ts"]:
        raise ValueError("calc_type must be either 'int' or 'ts'.")
    nx = atoms_to_nx(atoms, voronoi_tol, scaling_factor, surface_order, adsorbate_elems, calc_type)
    elem_list = list(get_node_attributes(nx, "elem").values())
    elem_array = np.array(elem_list).reshape(-1, 1)
    elem_enc = one_hot_encoder.transform(elem_array).toarray()
    x = torch.from_numpy(elem_enc).float()
    nodes_list = list(nx.nodes)
    edge_tails_heads = [(nodes_list.index(edge[0]), nodes_list.index(edge[1])) for edge in nx.edges]
    edge_tails = [x for x, _ in edge_tails_heads] + [y for _, y in edge_tails_heads]
    edge_heads = [y for _, y in edge_tails_heads] + [x for x, _ in edge_tails_heads]    
    edge_index = torch.tensor([edge_tails, edge_heads], dtype=torch.long)
    edge_attr = torch.zeros(edge_index.shape[1], 1)
    bb_idxs = "None"
    if calc_type == 'ts':
        for i in range(edge_index.shape[1]):
            edge_tuple = (nodes_list[edge_index[0, i].item()], nodes_list[edge_index[1, i].item()])
            if nx.edges[edge_tuple]['ts_edge'] == 1:
                edge_attr[i, 0] = 1  # As the nxgraph is undirected, the edge attribute is repeated twice
        bb_idxs = [(edge[0], edge[1]) for edge in nx.edges if nx.edges[edge]['ts_edge'] == 1][0]
    g = Data(x, 
             edge_index, 
             edge_attr, 
             elem=elem_list, 
             ase_indices=list(nx.nodes.keys()), 
             bb_idxs=bb_idxs, 
             type=calc_type, 
             node_feats=ELEMENT_DOMAIN, 
             formula = atoms.get_chemical_formula())
    return g


def atoms_to_data(
    structure: Atoms, 
    surface_order: int = 2,
    filter: bool = True
) -> Data:
    """
    Convert stable structures to PyG Data graph based on the input parameters.
    In CARE, this function is used only for intermediate species, not for transition states.
    The implementation is similar to the one in the ASE to PyG converter class, but it is not a class method and
    is used for inference. Target values are not included in the Data object.

    Args:
        structure (Atoms): ASE atoms object.
        surface_order (int): order of the surface neighbours to be included in the graph. If set to -1,
                            all surface slab is included.
    Returns:
        graph (Data): PyG Data object.
    """
    from gamenet_uq.node_featurizers import get_gcn

    if not isinstance(structure, Atoms):
        raise TypeError("Structure type must be ase.Atoms")

    # GRAPH STRUCTURE GENERATION
    graph = atoms_to_pyg(
        structure,
        "int",
        0.25,
        1.25,
        surface_order
    )

    # GRAPH FILTERING
    if filter:
        if not H_filter(graph, ADSORBATE_ELEMS):
            raise ValueError("{}: Wrong H connectivity in the adsorbate.".format(graph.formula))
        if not C_filter(graph, ADSORBATE_ELEMS):
            raise ValueError("{}: Wrong C connectivity in the adsorbate".format(graph.formula))
        if not fragment_filter(graph, ADSORBATE_ELEMS):
            raise ValueError("{}: Fragmented adsorbate.".format(graph.formula))

    # NODE FEATURIZATION
    gcn = get_gcn(structure)
    gcn_col = torch.zeros((graph.x.shape[0], 1))
    for i in range(graph.num_nodes):
        if graph.elem[i] not in ADSORBATE_ELEMS:
            gcn_col[i] = gcn[graph.ase_indices[i]].item()
    graph.x = torch.cat((graph.x, gcn_col), dim=1)
    graph.node_feats.append("gcn")
    return graph
