from io import StringIO

from ase.io import write
from ase import Atoms
import networkx as nx
from rdkit import Chem

from gamenet_uq.graph import get_voronoi_neighbourlist

def ase2inchikey(atoms: Atoms, 
                 adsorbate_elems: list[str]=[ "C", "H", "O", "N", "S"], 
                 tol=0.25):
    """
    Get InchiKey for a given ASE atoms object representing a chemical species.
    For gas phase and adsorbates on surfaces.
    Args:
        atoms (Atoms): ASE Atoms object representing a chemical species.
    Returns:
        str: InchiKey of the chemical species.

    Note:
    - It works for gas phase and single adsorbates on surfaces. If more than one 
        adsorbate is present, it will return "N/A".
    """
    
    atoms_CHONS = atoms.copy()
    atoms_CHONS = atoms_CHONS[[atom.symbol in adsorbate_elems for atom in atoms_CHONS]]
    nC, nH, nO, nN, nS = [atoms_CHONS.get_chemical_symbols().count(symbol) for symbol in ["C", "H", "O", "N", "S"]]
    if len(atoms_CHONS) != 1:
        atoms_CHONS *= (2, 2, 1)  # needed if the adsorbate crosses the periodic boundary
        nl = get_voronoi_neighbourlist(atoms_CHONS, tol, 1.0, ["C", "H", "O", "N", "S"], False)
        g = nx.Graph()
        for i, atom in enumerate(atoms_CHONS):
            g.add_node(i, element=atom.symbol)
        for pair in nl:
            g.add_edge(pair[0], pair[1])
        connected_components = list(nx.connected_components(g))
        largest_component = max(connected_components, key=len)
        gC, gH, gO, gN, gS = [0, 0, 0, 0, 0]
        for node in largest_component:
            element = g.nodes[node]["element"]
            if element == "C":
                gC += 1
            elif element == "H":
                gH += 1
            elif element == "O":
                gO += 1
            elif element == "N":
                gN += 1
            elif element == "S":
                gS += 1

        if gC != nC or gH != nH or gO != nO or gN != nN or gS != nS:
            return "N/A"

        idxs_largest_component = [i for i, _ in enumerate(atoms_CHONS) if i in largest_component]
        atoms_CHONS = atoms_CHONS[idxs_largest_component]  

    buffer = StringIO()
    write(buffer, atoms_CHONS, format='proteindatabank')
    buffer.seek(0)
    pdb_string = buffer.read()
    rdkit_mol = Chem.MolFromPDBBlock(pdb_string, removeHs=False)
    inchikey = Chem.inchi.MolToInchiKey(rdkit_mol, options='-DoNotAddH')
    return inchikey