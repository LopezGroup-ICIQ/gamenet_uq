"""
Module containing a set of filter functions for adsorption graphs in the Geometric PyTorch format.
These filters are applied before the inclusion of the graphs in the Dataset.
These filters work with the graph representation of GAME-Net-UQ only.
"""

from torch_geometric.utils import to_networkx
from ase import Atoms
from torch_geometric.data import Data
from torch import tensor
from networkx import is_connected, cycle_basis

from gamenet_uq.constants import ADSORBATE_ELEMS


def extract_adsorbate(graph: Data,
                     adsorbate_elems: list[str] = ADSORBATE_ELEMS) -> bool:
    """Extract adsorbate from the adsorption graph (adsorbate+surface).
    
    Args:
        graph(Data): Adsorption graph. It must contain a 'elem' attribute, which is a list
            of atomic elements in the graph.
        adsorbate_elems(list[str]): List of atomic elements that adsorbates can contain.
        
    Returns:
        (Data): Adsorbate graph."""
    
    assert graph.elem is not None, "elem should not be None"
    adsorbate_nodes = [node_idx for node_idx in range(graph.num_nodes) if graph.elem[node_idx] in adsorbate_elems]
    new_graph = graph.subgraph(tensor(adsorbate_nodes))
    new_elem = [graph.elem[i] for i in range(len(graph.elem)) if i in adsorbate_nodes]
    new_graph.elem = new_elem
    return new_graph


def fragment_filter(graph: Data, 
                    adsorbate_elems: list[str] = ADSORBATE_ELEMS) -> bool:
    """Check adsorbate fragmentation in the graph.
    Args:
        graph(Data): Adsorption graph.
        adsorbate_elems(list[str]): List of atomic elements in the adsorbate.

    Returns:
        (bool): True = Adsorbate is not fragmented
                False = Adsorbate is fragmented
    """

    assert graph.x is not None, "x should not be None"
    assert graph.num_nodes is not None, "num_nodes should not be None"
    adsorbate = extract_adsorbate(graph, adsorbate_elems)
    graph_nx = to_networkx(adsorbate, to_undirected=True)
    if adsorbate.num_nodes != 1 and adsorbate.num_edges != 0:
        if is_connected(graph_nx):
            return True
        else:
            return False
    else:
        return True
    

def is_ring(graph: Data, 
            adsorbate_elems: list[str] = ADSORBATE_ELEMS) -> bool:
    """Check if the adsorbate molecule contains a ring."""
    adsorbate = extract_adsorbate(graph, adsorbate_elems)
    graph_nx = to_networkx(adsorbate, to_undirected=True)
    cycles = list(cycle_basis(graph_nx))
    ring_nodes = set(node for cycle in cycles for node in cycle)
    if len(ring_nodes) > 0:
        return True
    else:
        return False


def H_filter(graph: Data, 
             adsorbate_elems: list[str] = ADSORBATE_ELEMS) -> bool:
    """
    Graph filter that checks the connectivity of H atoms whithin the adsorbate.
    Each H atoms must be connected to maximum one atom within the adsorbate.
    Args:
        graph(torch_geometric.data.Data): Graph object representation
        adsorbate_elems(list[str]): List of atomic elements in the adsorbate
    Returns:
        (bool): True = Correct connectivity for all H atoms in the adsorbate
                False = Bad connectivity for at least one H atom in the adsorbate
    """
    H_nodes_indices = [i for i, elem in enumerate(graph.elem) if elem == "H"]
    for node_index in H_nodes_indices:
        counter = 0
        for j in range(graph.num_edges):
            if graph.edge_index[0, j] == node_index:
                counter += 1 if graph.elem[graph.edge_index[1, j]] in adsorbate_elems else 0
        if counter > 1:
            return False
    return True


def C_filter(graph: Data, 
             adsorbate_elems: list[str] = ADSORBATE_ELEMS) -> bool:
    """
    Graph filter that checks the connectivity of C atoms whithin the adsorbate.
    Each C atom must be connected to maximum 4 atoms within the molecule.
    Args:
        graph(torch_geometric.data.Data): Graph object representation
        adsorbate_elems(list[str]): List of atomic elements in the molecule
    Returns:
        (bool): True = Correct connectivity for all C atoms in the molecule
                False = Bad connectivity for at least one C atom in the molecule
    """
    C_nodes_indices = [i for i, elem in enumerate(graph.elem) if elem == "C"]
    for node_index in C_nodes_indices:
        counter = 0
        for j in range(graph.num_edges):
            if graph.edge_index[0, j] == node_index:
                counter += 1 if graph.elem[graph.edge_index[1, j]] in adsorbate_elems else 0
        if counter > 4:
            return False
    return True

    
def adsorption_filter(graph: Data,  
                      adsorbate_elems: list[str] = ADSORBATE_ELEMS) -> bool:
    """
    Check presence of surface atoms in the adsorption graph.

    Args:
        graph(torch_geometric.data.Data): Graph object representation
        adsorbate_elems(list[str]): List of atomic elements in the molecule
    Returns:
        (bool): True = Metal catalyst present in the adsorption graph
                False = No metal catalyst in the adsorption graph
    """
    if graph.metal == 'N/A' and graph.facet == 'N/A':
        return True
    else:
        return False if all([elem in adsorbate_elems for elem in graph.elem]) else True
    

def ase_adsorption_filter(atoms: Atoms,
                          adsorbate_elems: list[str] = ADSORBATE_ELEMS) -> bool:
    """
    Check that the adsorbate has not been incorporated in the bulk.

    Args:
        graph (Data): Input adsorption/molecular graph.
        adsorbate_elems (list[str]): List of atomic elements in the molecule
    Returns:
        (bool): True = Adsorbate is not incorporated in the bulk
                False = Adsorbate is incorporated in the bulk
    """
    if all([atom.symbol in adsorbate_elems for atom in atoms]):
        return True
    else:
        z_adsorbate = [atom.position[2] for atom in atoms if atom.symbol in adsorbate_elems]
        z_surface = [atom.position[2] for atom in atoms if atom.symbol not in adsorbate_elems]
        if len(z_adsorbate) == 0 or len(z_surface) == 0:
            print(f"{atoms.get_chemical_formula(mode='metal')}: No adsorbate or surface atoms found.")
            return False
        min_adsorbate_z = min(z_adsorbate)
        max_surface_z = max(z_surface)
        if min_adsorbate_z < 0.75 * max_surface_z:
            print(f"{atoms.get_chemical_formula(mode='metal')}: Adsorbate incorporated in the bulk.")
            return False
        else:
            return True
