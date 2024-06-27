from io import StringIO

from ase.io import read, write
from ase import Atoms, Atom
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

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


def ase2inchikey2(atoms: Atoms, 
                 adsorbate_elems: list[str]=[ "C", "H", "O", "N", "S"], 
                 tol=0.25):
    """
    Get InchiKey for a given ASE atoms object representing a chemical species.
    For gas phase and adsorbates on surfaces.

    Steps: 
    1. Extract adsorbate atoms.
    2. Perform force-field geometry optimization.
    3. Get InchiKey from the optimized structure.

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

    
    # Perform force-field geometry optimization
    rdkit_mol = Chem.AddHs(rdkit_mol)
    num_conformers = 50 * nC + 10 * nO + 5*nN + 5*nS

    if rdkit_mol.GetNumAtoms() > 2:
        AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=num_conformers)
        confs = AllChem.MMFFOptimizeMoleculeConfs(rdkit_mol)
        conf_energies = [item[1] for item in confs]
        lowest_conf = int(np.argmin(conf_energies))
        # get rd_kit with lowest energy
        rdkit_mol = Chem.Mol(rdkit_mol, lowest_conf)
        
    else:
        AllChem.EmbedMolecule(rdkit_mol, AllChem.ETKDG())

    # Get InchiKey

    return Chem.inchi.MolToInchiKey(rdkit_mol, options='-DoNotAddH')


def rdkit_to_ase(self) -> Atoms:
    """
    Generate an ASE Atoms object from an RDKit molecule.

    """
    rdkit_molecule = self.molecule

    # If there are no atoms in the molecule, return an empty ASE Atoms object (Surface)
    if rdkit_molecule.GetNumAtoms() == 0:
        return Atoms()

    # Generate 3D coordinates for the molecule
    rdkit_molecule = Chem.AddHs(
        rdkit_molecule
    )  # Add hydrogens if not already added

    num_C = sum([1 for atom in rdkit_molecule.GetAtoms() if atom.GetSymbol() == "C"])
    num_O = sum([1 for atom in rdkit_molecule.GetAtoms() if atom.GetSymbol() == "O"])
    num_conformers = 50 * num_C + 10 * num_O

    # If the molecule has more than 1 atom, generate multiple conformers and optimize them
    if rdkit_molecule.GetNumAtoms() > 2:
        AllChem.EmbedMultipleConfs(rdkit_molecule, numConfs=num_conformers)
        confs = AllChem.MMFFOptimizeMoleculeConfs(rdkit_molecule)
        conf_energies = [item[1] for item in confs]
        lowest_conf = int(np.argmin(conf_energies))
        xyz_coordinates = AllChem.MolToXYZBlock(rdkit_molecule,confId=lowest_conf)
        
        # Generating the ASE atoms object from the XYZ coordinates string
        ase_atoms = read(StringIO(xyz_coordinates), format="xyz")
    else:
        AllChem.EmbedMolecule(rdkit_molecule, AllChem.ETKDG())

        # Get the number of atoms in the molecule
        num_atoms = rdkit_molecule.GetNumAtoms()

        # Initialize lists to store positions and symbols
        positions = []
        symbols = []

        # Extract atomic positions and symbols
        for atom_idx in range(num_atoms):
            atom_position = rdkit_molecule.GetConformer().GetAtomPosition(atom_idx)
            atom_symbol = rdkit_molecule.GetAtomWithIdx(atom_idx).GetSymbol()
            positions.append(atom_position)
            symbols.append(atom_symbol)

        # Create an ASE Atoms object
        ase_atoms = Atoms(
            [
                Atom(symbol=symbol, position=position)
                for symbol, position in zip(symbols, positions)
            ]
        )

    ase_atoms.set_pbc(True)

    return ase_atoms
