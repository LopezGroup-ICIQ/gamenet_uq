"""
Script to generate adsorption graph dataset from ASE database.
"""

import argparse
import os
import sys
sys.path.append('../src')

from gamenet_uq.dataset import AdsorptionGraphDataset

def main():
    parser = argparse.ArgumentParser(description="Generate adsorption graph dataset from ASE database.")
    parser.add_argument("--ase_db_path", type=str, default="../data/fg.db", help="Path to ASE database.")
    parser.add_argument("--graph_dataset_path", type=str, default="../data", help="Path to save graph dataset.")
    parser.add_argument("--db_key", type=str, default="", help="Keys to select specific instances in the ASE database.")
    parser.add_argument("--tol", type=float, default=0.25, help="Tolerance in Angstrom for detection of bonds between atoms during graph construction.")
    parser.add_argument("--scaling_factor", type=float, default=1.25, help="Scaling factor applied to atomic radii from Cordero et al. of surface atoms when detecting adsorbate-surface bonds.")
    parser.add_argument("--second_order", action="store_true", help="Include second-order surface neighbors in adsorption graph.")
    parser.add_argument("--target", type=str, default="scaled_energy", help="Target property to predict. It must be included in the ASE database as column.")
    parser.add_argument("--cores", type=int, default=1, help="Number of cores to use for parallel processing.")
    args = parser.parse_args()
    print(args)

    STRUCTURE_DICT = {"tolerance": args.tol, "scaling_factor": args.scaling_factor, "second_order": args.second_order}
    NODE_ATTRIBUTES = {"adsorbate": False, "radical": False, "valence": False, "gcn": True, "magnetization": False}
    GRAPH_PARAMS = {"structure": STRUCTURE_DICT, "features": NODE_ATTRIBUTES, "target": args.target}
    dataset = AdsorptionGraphDataset(ase_database_path=args.ase_db_path, 
                                     graph_dataset_dir=args.graph_dataset_path, 
                                     graph_params=GRAPH_PARAMS, 
                                     db_key=args.db_key, 
                                     ncores=args.cores)
    print("Dataset generated successfully. Stored as {}".format(os.path.abspath(dataset.output_path)))
    print("Graph dataset size: {}".format(len(dataset)))


if __name__ == "__main__":
    main()
